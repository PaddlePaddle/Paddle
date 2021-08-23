/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/optimizers/lars_momentum_op.h"

#define BLOCK_SIZE 256

namespace paddle {
namespace operators {

template <typename T>
using MultiPrecisionType = typename details::MPTypeTrait<T>::Type;

/*
Two stages are set up to deal with grid-level reduction sum:
    1. Do block-reduce in each block and acquire the partial sum.
    2. Merge the partial sum to get the final sum.
while __syncthreads() can only sync all threads within a block,
it cannot sync all blocks. Once lanuching stage 2, it is necessary
to sync all blocks before operatiion. Function below is made for this.
*/
__device__ bool SyncAllBlock(int* counter) {
  int last;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *counter = 0;
  }
  __threadfence();  // Ensure that partial result of each block is visible by
                    // all blocks
  if (threadIdx.x == 0) {
    last = atomicAdd(counter, 1);
  }
  return __syncthreads_or(last < gridDim.x);
}

template <typename T, typename CastT>
__device__ CastT L2NormCalculation(const T* __restrict__ data,
                                   CastT* tmp_buffer, int* counter, int tid,
                                   const int64_t num,
                                   const CastT rescale_grad = 1) {
  int stride = BLOCK_SIZE * BLOCK_SIZE;
  int reduce_times = (gridDim.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int rest_block = gridDim.x;
  int numel = num;
  int limiTblock;
  CastT tmp_val = 0.f;

  __shared__ CastT buffer;
  buffer = 0.f;

  if (reduce_times == 1) {
    limiTblock = gridDim.x;
    if (tid < numel) {
      tmp_val = static_cast<CastT>(data[tid]);
    }
    buffer += math::blockReduceSum<CastT>(tmp_val * tmp_val * rescale_grad,
                                          FINAL_MASK);
  } else {
    limiTblock = BLOCK_SIZE;
    for (int i = 0; i < reduce_times - 1; ++i) {
      numel -= stride;
      rest_block -= BLOCK_SIZE;
      if (tid < stride) {
        tmp_val = static_cast<CastT>(data[tid + stride * i]);
        buffer += math::blockReduceSum<CastT>(tmp_val * tmp_val * rescale_grad,
                                              FINAL_MASK);
      }
      __syncthreads();
    }
    CastT val;
    if (tid < numel) {
      val = static_cast<CastT>(data[tid + stride * (reduce_times - 1)]);
    }
    if (blockIdx.x < rest_block) {
      buffer +=
          math::blockReduceSum<CastT>(val * val * rescale_grad, FINAL_MASK);
    }
  }
  __syncthreads();

  if (blockIdx.x < limiTblock && threadIdx.x == 0) {
    tmp_buffer[blockIdx.x] = buffer;
  }
  if (SyncAllBlock(counter)) {
    CastT tmp_value = threadIdx.x < limiTblock ? tmp_buffer[threadIdx.x] : 0;
    __syncthreads();
    buffer = math::blockReduceSum<CastT>(tmp_value, FINAL_MASK);
    return std::sqrt(buffer);
  }
}

template <typename T, typename MT>
__global__ void MomentumLarsKernel(
    const T* __restrict__ p, const T* __restrict__ g, const MT* __restrict__ v,
    const MT* learning_rate, const MT mu, const int64_t num,
    const MT lars_coeff, const MT lars_weight_decay, MT* l2_tmp_buffer,
    int* l2_tmp_counter, T* p_out, MT* v_out, const MT epsilon,
    const MT* master_p, MT* master_p_out, const MT rescale_grad) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  MT p_n = L2NormCalculation<T, MT>(p, l2_tmp_buffer, l2_tmp_counter, tid, num);
  MT g_n = L2NormCalculation<T, MT>(g, l2_tmp_buffer, l2_tmp_counter, tid, num,
                                    rescale_grad);
  const MT lr = learning_rate[0];
  MT local_lr = lr;

  if (lars_weight_decay > static_cast<MT>(0) && p_n > static_cast<MT>(0) &&
      g_n > static_cast<MT>(0)) {
    local_lr =
        lr * lars_coeff * p_n / (g_n + lars_weight_decay * p_n + epsilon);
  }

  CUDA_KERNEL_LOOP(i, num) {
    MT grad = static_cast<MT>(g[tid]) * static_cast<MT>(rescale_grad);
    MT param = master_p ? master_p[tid] : static_cast<MT>(p[tid]);

    MT v_new = v[tid] * mu + local_lr * (grad + lars_weight_decay * param);
    MT p_new = param - v_new;

    v_out[tid] = v_new;
    p_out[tid] = static_cast<T>(p_new);
    if (master_p_out) master_p_out[tid] = p_new;
  }
}

template <typename DeviceContext, typename T>
class LarsMomentumOpCUDAKernel : public framework::OpKernel<T> {
  using MT = MultiPrecisionType<T>;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const bool multi_precision = ctx.Attr<bool>("multi_precision");
    InnerCompute(ctx, multi_precision);
  }

 private:
  void InnerCompute(const framework::ExecutionContext& ctx,
                    const bool multi_precision) const {
    auto param_out = ctx.Output<framework::LoDTensor>("ParamOut");
    auto velocity_out = ctx.Output<framework::LoDTensor>("VelocityOut");
    auto param = ctx.Input<framework::LoDTensor>("Param");
    auto velocity = ctx.Input<framework::LoDTensor>("Velocity");
    auto grad = ctx.Input<framework::LoDTensor>("Grad");
    auto learning_rate = ctx.Input<framework::LoDTensor>("LearningRate");

    const framework::Tensor* master_param = nullptr;
    framework::Tensor* master_param_out = nullptr;
    if (multi_precision) {
      bool has_master =
          ctx.HasInput("MasterParam") && ctx.HasOutput("MasterParamOut");
      PADDLE_ENFORCE_EQ(has_master, true,
                        platform::errors::InvalidArgument(
                            "The Input(MasterParam) and Output(MasterParamOut) "
                            "should not be null when "
                            "the attr `multi_precision` is true"));
      master_param = ctx.Input<framework::Tensor>("MasterParam");
      master_param_out = ctx.Output<framework::Tensor>("MasterParamOut");
    }

    const MT* master_p = multi_precision ? master_param->data<MT>() : nullptr;
    MT* master_p_out = multi_precision
                           ? master_param_out->mutable_data<MT>(ctx.GetPlace())
                           : nullptr;

    T* p_out = param_out->mutable_data<T>(ctx.GetPlace());
    MT* v_out = velocity_out->mutable_data<MT>(ctx.GetPlace());

    MT mu = static_cast<MT>(ctx.Attr<float>("mu"));
    MT lars_coeff = static_cast<MT>(ctx.Attr<float>("lars_coeff"));
    MT lars_weight_decay =
        static_cast<MT>(ctx.Attr<float>("lars_weight_decay"));
    MT epsilon = static_cast<MT>(ctx.Attr<float>("epsilon"));
    MT rescale_grad = static_cast<MT>(ctx.Attr<float>("rescale_grad"));

    auto* p = param->data<T>();
    auto* g = grad->data<T>();
    auto* v = velocity->data<MT>();
    auto* lr = learning_rate->data<MT>();
    int grid = (param->numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int l2_tmp_buffer_size = grid < BLOCK_SIZE ? grid : BLOCK_SIZE;
    framework::Tensor l2_tmp_buffer_t, l2_tmp_counter_t;
    auto* l2_tmp_buffer_data =
        l2_tmp_buffer_t.mutable_data<MT>({l2_tmp_buffer_size}, ctx.GetPlace());
    int* l2_tmp_counter_data =
        l2_tmp_counter_t.mutable_data<int>({1}, ctx.GetPlace());

    MomentumLarsKernel<
        T, MT><<<grid, BLOCK_SIZE, 0, ctx.cuda_device_context().stream()>>>(
        p, g, v, lr, mu, param->numel(), lars_coeff, lars_weight_decay,
        l2_tmp_buffer_data, l2_tmp_counter_data, p_out, v_out, epsilon,
        master_p, master_p_out, rescale_grad);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    lars_momentum,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::LarsMomentumOpCUDAKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>);
