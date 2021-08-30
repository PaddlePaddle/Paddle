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
#ifdef __NVCC__
#include <cooperative_groups.h>
#endif

#ifdef __HIPCC__
#define LARS_BLOCK_SIZE 256
#else
#define LARS_BLOCK_SIZE 512
#endif

namespace paddle {
namespace operators {

template <typename T>
using MultiPrecisionType = typename details::MPTypeTrait<T>::Type;

__device__ __forceinline__ float square_root(float x) { return sqrtf(x); }
__device__ __forceinline__ double square_root(double x) { return sqrt(x); }
__device__ __forceinline__ float fma_root(float x, float y, float z) {
  return fmaf(x, y, z);
}
__device__ __forceinline__ double fma_root(double x, double y, double z) {
  return fma(x, y, z);
}

#if CUDA_VERSION >= 9000
template <typename T, typename MT>
__device__ MT L2NormCalculation(const cooperative_groups::grid_group& cg,
                                const T* __restrict__ data, MT* tmp_buffer,
                                int tid, const int64_t numel,
                                const MT rescale_grad = static_cast<MT>(1)) {
  int stride = gridDim.x * LARS_BLOCK_SIZE;
  int reduce_times = (numel + stride - 1) / stride;
  MT rescale_grad_pow = rescale_grad * rescale_grad;

  __shared__ MT s_buffer;
  s_buffer = static_cast<MT>(0);

  MT tmp_val = static_cast<MT>(0);
  if (reduce_times == 1) {
    if (tid < numel) {
      tmp_val = static_cast<MT>(data[tid]);
    }
    s_buffer += math::blockReduceSum<MT>(tmp_val * tmp_val, FINAL_MASK);
  } else {
    for (int i = 0; i < reduce_times - 1; ++i) {
      if (tid < numel) {
        tmp_val = static_cast<MT>(data[tid]);
      }
      tid += stride;
      s_buffer += math::blockReduceSum<MT>(tmp_val * tmp_val, FINAL_MASK);
      __syncthreads();
    }
    MT val = static_cast<MT>(0);
    if (tid < numel) {
      val = static_cast<MT>(data[tid]);
    }
    s_buffer += math::blockReduceSum<MT>(val * val, FINAL_MASK);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    tmp_buffer[blockIdx.x] = s_buffer;
  }
  cg.sync();
  MT val = threadIdx.x < gridDim.x ? tmp_buffer[threadIdx.x] : 0;
  MT result = math::blockReduceSum<MT>(val, FINAL_MASK);
  return square_root(rescale_grad_pow * result);
}
#endif

template <typename T, typename MT>
__global__ void MomentumLarsKernel(
    const T* __restrict__ p, const T* __restrict__ g, const MT* __restrict__ v,
    const MT* __restrict__ learning_rate, const MT mu, const int64_t numel,
    const MT lars_coeff, const MT lars_weight_decay, T* p_out, MT* v_out,
    const MT epsilon, const MT* __restrict__ master_p, MT* master_p_out,
    const MT rescale_grad, MT* tmp_buffer, MT* tmp_buffer_2 = nullptr) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if CUDA_VERSION >= 9000
  const cooperative_groups::grid_group cg = cooperative_groups::this_grid();
  MT p_n = L2NormCalculation<T, MT>(cg, p, tmp_buffer, tid, numel);
  MT g_n =
      L2NormCalculation<T, MT>(cg, g, tmp_buffer, tid, numel, rescale_grad);
#else
  const MT p_n = tmp_buffer[0];
  const MT g_n = tmp_buffer_2[0];
#endif
  const MT lr = learning_rate[0];
  MT local_lr = lr;
  if (lars_weight_decay > static_cast<MT>(0) && p_n > static_cast<MT>(0) &&
      g_n > static_cast<MT>(0)) {
    local_lr = lr * lars_coeff * p_n /
               (fma_root(lars_weight_decay, p_n, g_n) + epsilon);
  }

  if (master_p) {
    for (int i = tid; i < numel; i += blockDim.x * gridDim.x) {
      MT grad = static_cast<MT>(g[i]) * rescale_grad;
      MT param = master_p[i];
      MT v_new = fma_root(v[i], mu,
                          local_lr * fma_root(lars_weight_decay, param, grad));
      MT p_new = param - v_new;
      v_out[i] = v_new;
      p_out[i] = static_cast<T>(p_new);
      master_p_out[i] = p_new;
    }
  } else {
    for (int i = tid; i < numel; i += blockDim.x * gridDim.x) {
      MT grad = static_cast<MT>(g[i]) * rescale_grad;
      MT param = static_cast<MT>(p[i]);
      MT v_new = fma_root(v[i], mu,
                          local_lr * fma_root(lars_weight_decay, param, grad));
      MT p_new = param - v_new;
      v_out[i] = v_new;
      p_out[i] = static_cast<T>(p_new);
    }
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
    const MT* master_p = nullptr;
    MT* master_p_out = nullptr;

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
      master_p = master_param->data<MT>();
      master_p_out = master_param_out->mutable_data<MT>(ctx.GetPlace());
    }

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
    int64_t numel = param->numel();
    int grid = (numel + LARS_BLOCK_SIZE - 1) / LARS_BLOCK_SIZE;

#if CUDA_VERSION >= 9000
    int num_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
                                                  MomentumLarsKernel<T, MT>,
                                                  LARS_BLOCK_SIZE, sizeof(MT));
    int sm_num = ctx.cuda_device_context().GetSMCount();
    int grid_real = std::min(sm_num * num_blocks_per_sm, grid);
    framework::Tensor tmp_buffer_t;
    auto* tmp_buffer =
        tmp_buffer_t.mutable_data<MT>({grid_real}, ctx.GetPlace());

    void* cuda_param[] = {reinterpret_cast<void*>(&p),
                          reinterpret_cast<void*>(&g),
                          reinterpret_cast<void*>(&v),
                          reinterpret_cast<void*>(&lr),
                          reinterpret_cast<void*>(&mu),
                          reinterpret_cast<void*>(&numel),
                          reinterpret_cast<void*>(&lars_coeff),
                          reinterpret_cast<void*>(&lars_weight_decay),
                          reinterpret_cast<void*>(&p_out),
                          reinterpret_cast<void*>(&v_out),
                          reinterpret_cast<void*>(&epsilon),
                          reinterpret_cast<void*>(&master_p),
                          reinterpret_cast<void*>(&master_p_out),
                          reinterpret_cast<void*>(&rescale_grad),
                          reinterpret_cast<void*>(&tmp_buffer),
                          nullptr};
    cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(MomentumLarsKernel<T, MT>), grid_real,
        LARS_BLOCK_SIZE, cuda_param, 0, ctx.cuda_device_context().stream());
#else
    auto eigen_p = framework::EigenVector<T>::Flatten(*param);
    auto eigen_g = framework::EigenVector<T>::Flatten(*grad);
    // calculate norms using eigein and launch the kernel.
    framework::Tensor p_norm_t, g_norm_t;
    p_norm_t.Resize({1});
    g_norm_t.Resize({1});
    auto* p_norm_data = p_norm_t.mutable_data<MT>(ctx.GetPlace());
    auto* g_norm_data = g_norm_t.mutable_data<MT>(ctx.GetPlace());
    auto ep_norm = framework::EigenScalar<MT>::From(p_norm_t);
    auto eg_norm = framework::EigenScalar<MT>::From(g_norm_t);
    auto* place = ctx.template device_context<DeviceContext>().eigen_device();
    // eigen unsupport fp16 l2-norm
    ep_norm.device(*place) = eigen_p.template cast<MT>().square().sum().sqrt();
    eg_norm.device(*place) =
        (eigen_g.template cast<MT>() * rescale_grad).square().sum().sqrt();

    MomentumLarsKernel<
        T,
        MT><<<grid, LARS_BLOCK_SIZE, 0, ctx.cuda_device_context().stream()>>>(
        p, g, v, lr, mu, numel, lars_coeff, lars_weight_decay, p_out, v_out,
        epsilon, master_p, master_p_out, rescale_grad, p_norm_data,
        g_norm_data);
#endif
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
