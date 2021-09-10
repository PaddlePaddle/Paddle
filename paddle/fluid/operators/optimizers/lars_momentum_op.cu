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
#include "paddle/fluid/platform/fast_divmod.h"

#ifdef __HIPCC__
#define LARS_BLOCK_SIZE 256
#else
#define LARS_BLOCK_SIZE 512
#endif

namespace paddle {
namespace operators {

template <typename T>
using MultiPrecisionType = typename details::MPTypeTrait<T>::Type;

__device__ __forceinline__ float SquareRoot(float x) { return sqrtf(x); }
__device__ __forceinline__ double SquareRoot(double x) { return sqrt(x); }
__device__ __forceinline__ float FmaRoot(float x, float y, float z) {
  return fmaf(x, y, z);
}
__device__ __forceinline__ double FmaRoot(double x, double y, double z) {
  return fma(x, y, z);
}

template <typename T, typename MT>
__device__ inline void VectorizeLarsUpdateMP(
    const T* __restrict__ g, const MT* __restrict__ v, T* __restrict__ p_out,
    MT* __restrict__ v_out, const MT* __restrict__ master_p,
    MT* __restrict__ master_p_out, const MT mu, MT local_lr,
    const MT lars_weight_decay, const MT rescale_grad, const int tid,
    const int grid_stride, const int numel) {
  // As for multiple-precision, type T and MT cannot be more than fp16 or fp32,
  // Then, the maximum data IO size could be set to 4.
  using VecType = paddle::platform::AlignedVector<T, 4>;
  using VecMType = paddle::platform::AlignedVector<MT, 4>;
  int main = numel >> 2;
  int tail_offset = main << 2;

  const VecType* __restrict__ g_vec = reinterpret_cast<const VecType*>(g);
  const VecMType* __restrict__ v_vec = reinterpret_cast<const VecMType*>(v);
  const VecMType* __restrict__ master_p_vec =
      reinterpret_cast<const VecMType*>(master_p);
  VecType* p_out_vec = reinterpret_cast<VecType*>(p_out);
  VecMType* v_out_vec = reinterpret_cast<VecMType*>(v_out);
  VecMType* master_p_out_vec = reinterpret_cast<VecMType*>(master_p_out);

  for (int i = tid; i < main; i += grid_stride) {
    VecType p_out;
    VecMType v_new, p_new;
    VecType g_data = g_vec[i];
    VecMType v_data = v_vec[i];
    VecMType p_data = master_p_vec[i];

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      MT grad = static_cast<MT>(g_data.val[j]) * rescale_grad;
      v_new.val[j] =
          FmaRoot(v_data.val[j], mu,
                  local_lr * FmaRoot(lars_weight_decay, p_data.val[j], grad));
      p_new.val[j] = p_data.val[j] - v_new.val[j];
      p_out.val[j] = static_cast<T>(p_new.val[j]);
    }
    v_out_vec[i] = v_new;
    p_out_vec[i] = p_out;
    master_p_out_vec[i] = p_new;
  }

  for (int i = tid + tail_offset; i < numel; i += grid_stride) {
    MT grad = static_cast<MT>(g[i]) * rescale_grad;
    MT param = master_p[i];
    MT v_new =
        FmaRoot(v[i], mu, local_lr * FmaRoot(lars_weight_decay, param, grad));
    MT p_new = param - v_new;
    v_out[i] = v_new;
    p_out[i] = static_cast<T>(p_new);
    master_p_out[i] = p_new;
  }
}

template <typename MT, int VesSize>
__device__ inline void VectorizeLarsUpdate(
    const MT* __restrict__ g, const MT* __restrict__ v, MT* __restrict__ p_out,
    MT* __restrict__ v_out, const MT* __restrict__ p, const MT mu, MT local_lr,
    const MT lars_weight_decay, const MT rescale_grad, const int tid,
    const int grid_stride, const int numel) {
  using VecMType = paddle::platform::AlignedVector<MT, VesSize>;
  int main = numel >> (VesSize >> 1);
  int tail_offset = main * VesSize;

  const VecMType* __restrict__ g_vec = reinterpret_cast<const VecMType*>(g);
  const VecMType* __restrict__ v_vec = reinterpret_cast<const VecMType*>(v);
  const VecMType* __restrict__ p_vec = reinterpret_cast<const VecMType*>(p);
  VecMType* p_out_vec = reinterpret_cast<VecMType*>(p_out);
  VecMType* v_out_vec = reinterpret_cast<VecMType*>(v_out);

  for (int i = tid; i < main; i += grid_stride) {
    VecMType v_new, p_new;
    VecMType g_data = g_vec[i];
    VecMType v_data = v_vec[i];
    VecMType p_data = p_vec[i];

#pragma unroll
    for (int j = 0; j < VesSize; ++j) {
      MT grad = g_data.val[j] * rescale_grad;
      v_new.val[j] =
          FmaRoot(v_data.val[j], mu,
                  local_lr * FmaRoot(lars_weight_decay, p_data.val[j], grad));
      p_new.val[j] = p_data.val[j] - v_new.val[j];
    }
    v_out_vec[i] = v_new;
    p_out_vec[i] = p_new;
  }

  for (int i = tid + tail_offset; i < numel; i += grid_stride) {
    MT grad = g[i] * rescale_grad;
    MT param = p[i];
    MT v_new =
        FmaRoot(v[i], mu, local_lr * FmaRoot(lars_weight_decay, param, grad));
    v_out[i] = v_new;
    p_out[i] = param - v_new;
  }
}

template <typename T, typename MT>
__global__ void L2NormKernel(const T* __restrict__ p_data,
                             const T* __restrict__ g_data,
                             MT* __restrict__ p_tmp_buffer,
                             MT* __restrict__ g_tmp_buffer,
                             const int64_t repeat_times, const int64_t numel) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int grid_stride = LARS_BLOCK_SIZE * gridDim.x;
  __shared__ MT s_buffer[2];
  s_buffer[0] = static_cast<MT>(0);
  s_buffer[1] = static_cast<MT>(0);
  MT p_tmp_val = static_cast<MT>(0);
  MT g_tmp_val = static_cast<MT>(0);

  if (repeat_times == 0) {
    if (tid < numel) {
      p_tmp_val = static_cast<MT>(p_data[tid]);
      g_tmp_val = static_cast<MT>(g_data[tid]);
    }
    s_buffer[0] += math::blockReduceSum<MT>(p_tmp_val * p_tmp_val, FINAL_MASK);
    s_buffer[1] += math::blockReduceSum<MT>(g_tmp_val * g_tmp_val, FINAL_MASK);
  } else {
    /* To avoid occupy too much temp buffer. Hence, slice the whole data into 2
    parts, the front of them whose quantity is excatly multiple of grid-thread
    number, and this part of data is delt in for loop, the rest of data is delt
    with another step to avoid visiting data address beyond bound. */
    for (int i = 0; i < repeat_times; ++i) {
      p_tmp_val = static_cast<MT>(p_data[tid]);
      g_tmp_val = static_cast<MT>(g_data[tid]);
      tid += grid_stride;
      s_buffer[0] +=
          math::blockReduceSum<MT>(p_tmp_val * p_tmp_val, FINAL_MASK);
      s_buffer[1] +=
          math::blockReduceSum<MT>(g_tmp_val * g_tmp_val, FINAL_MASK);
      __syncthreads();
    }
    MT p_val = 0;
    MT g_val = 0;
    if (tid < numel) {
      p_val = static_cast<MT>(p_data[tid]);
      g_val = static_cast<MT>(g_data[tid]);
    }
    s_buffer[0] += math::blockReduceSum<MT>(p_val * p_val, FINAL_MASK);
    s_buffer[1] += math::blockReduceSum<MT>(g_val * g_val, FINAL_MASK);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    p_tmp_buffer[blockIdx.x] = s_buffer[0];
    g_tmp_buffer[blockIdx.x] = s_buffer[1];
  }
}

template <typename T, typename MT>
__global__ void MomentumLarsKernel(
    const T* __restrict__ p, const T* __restrict__ g, const MT* __restrict__ v,
    T* __restrict__ p_out, MT* __restrict__ v_out,
    const MT* __restrict__ master_p, MT* __restrict__ master_p_out,
    const MT* __restrict__ learning_rate, MT* __restrict__ p_tmp_buffer,
    MT* __restrict__ g_tmp_buffer, const MT mu, const MT lars_coeff,
    const MT lars_weight_decay, const MT epsilon, const MT rescale_grad,
    const int thresh, const int64_t numel) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int grid_stride = LARS_BLOCK_SIZE * gridDim.x;
  const MT rescale_grad_pow = rescale_grad * rescale_grad;
  MT p_val = threadIdx.x < thresh ? p_tmp_buffer[threadIdx.x] : 0;
  MT g_val = threadIdx.x < thresh ? g_tmp_buffer[threadIdx.x] : 0;
  __syncthreads();
  MT p_n = SquareRoot(math::blockReduceSum<MT>(p_val, FINAL_MASK));
  MT g_n = SquareRoot(rescale_grad_pow *
                      math::blockReduceSum<MT>(g_val, FINAL_MASK));

  const MT lr = learning_rate[0];
  MT local_lr = lr;
  if (lars_weight_decay > static_cast<MT>(0)) {
    local_lr = lr * lars_coeff * p_n /
               (FmaRoot(lars_weight_decay, p_n, g_n) + epsilon);
  }

  if (master_p) {
    VectorizeLarsUpdateMP<T, MT>(g, v, p_out, v_out, master_p, master_p_out, mu,
                                 local_lr, lars_weight_decay, rescale_grad, tid,
                                 grid_stride, numel);
  } else {
    if (sizeof(T) < 64) {  // 64 stands for sizeof(double)
      VectorizeLarsUpdate<MT, 4>(
          reinterpret_cast<const MT*>(g), v, reinterpret_cast<MT*>(p_out),
          v_out, reinterpret_cast<const MT*>(p), mu, local_lr,
          lars_weight_decay, rescale_grad, tid, grid_stride, numel);
    } else {
      VectorizeLarsUpdate<MT, 2>(
          reinterpret_cast<const MT*>(g), v, reinterpret_cast<MT*>(p_out),
          v_out, reinterpret_cast<const MT*>(p), mu, local_lr,
          lars_weight_decay, rescale_grad, tid, grid_stride, numel);
    }
  }
}

template <typename DeviceContext, typename T>
class LarsMomentumOpCUDAKernel : public framework::OpKernel<T> {
  using MT = MultiPrecisionType<T>;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const bool multi_precision = ctx.Attr<bool>("multi_precision");
    auto param_out = ctx.Output<framework::LoDTensor>("ParamOut");
    auto velocity_out = ctx.Output<framework::LoDTensor>("VelocityOut");
    auto param = ctx.Input<framework::LoDTensor>("Param");
    auto velocity = ctx.Input<framework::LoDTensor>("Velocity");
    auto grad = ctx.Input<framework::LoDTensor>("Grad");
    auto learning_rate = ctx.Input<framework::LoDTensor>("LearningRate");
    auto& cuda_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    int64_t numel = param->numel();
    int grid_norm = (numel + LARS_BLOCK_SIZE - 1) / LARS_BLOCK_SIZE;
    // Determine to read 4 fp16 or float data once, but 2 double data once.
    int grid_lars =
        sizeof(T) < 64
            ? (numel + (LARS_BLOCK_SIZE << 2) - 1) / (LARS_BLOCK_SIZE << 2)
            : (numel + (LARS_BLOCK_SIZE << 1) - 1) / (LARS_BLOCK_SIZE << 1);

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
    T* p_out = param_out->mutable_data<T>(ctx.GetPlace());
    MT* v_out = velocity_out->mutable_data<MT>(ctx.GetPlace());

    /*
    Once model trainning with lars optimizer, whose principal implementation
    is achieved by following two steps:
      1. Figure out the L2 norm statistic result of grad data and param data.
      2. Update param and velocity data with usage of L2 norm statistic result.
    Orignally, these two steps were fulfilled by respective eigen function and
    cuda kernel, however the overhead of eigen function occupied much ratio in
    total, consequently affect the performance of lars op, make it necessary to
    replace the eigen with cuda kernel.
    */
    // Allocate temp buffer for L2 norm kernel.
    grid_norm = std::min(grid_norm, LARS_BLOCK_SIZE);
    framework::Tensor p_buffer_t =
        ctx.AllocateTmpTensor<MT, platform::CUDADeviceContext>({grid_norm << 1},
                                                               cuda_ctx);
    auto* p_tmp_buffer = p_buffer_t.mutable_data<MT>(ctx.GetPlace());
    auto* g_tmp_buffer = p_tmp_buffer + grid_norm;

    const int grid_stride = LARS_BLOCK_SIZE * grid_norm;
    const int repeat_times = (numel + grid_stride - 1) / grid_stride - 1;

    L2NormKernel<T, MT><<<grid_norm, LARS_BLOCK_SIZE, 0, cuda_ctx.stream()>>>(
        p, g, p_tmp_buffer, g_tmp_buffer, repeat_times, numel);

    MomentumLarsKernel<
        T, MT><<<grid_lars, LARS_BLOCK_SIZE, 0, cuda_ctx.stream()>>>(
        p, g, v, p_out, v_out, master_p, master_p_out, lr, p_tmp_buffer,
        g_tmp_buffer, mu, lars_coeff, lars_weight_decay, epsilon, rescale_grad,
        grid_norm, numel);
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
