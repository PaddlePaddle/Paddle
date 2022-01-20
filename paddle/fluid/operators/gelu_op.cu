/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/gelu_op.h"

DECLARE_bool(use_fast_math);

namespace paddle {
namespace operators {

#ifdef __NVCC__
template <bool FastMode>
static __device__ __forceinline__ float FP32FastTanh(float x) {
#if __CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000
  if (FastMode) {
    float y;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(y) : "f"(x));
    return y;
  }
#endif
  return tanhf(x);
}

template <bool FastMode>
static __device__ __forceinline__ float FP32GeluFwd(float x) {
  auto tanh_out =
      FP32FastTanh<FastMode>(0.79788456f * x * (1.0f + 0.044715f * x * x));
  return x * 0.5f * (1.0f + tanh_out);
}

template <bool FastMode>
static __device__ __forceinline__ float FP32GeluBwd(float x, float y_g) {
  auto tanh_out =
      FP32FastTanh<FastMode>(0.79788456f * x * (1.0f + 0.044715f * x * x));
  auto tmp = 0.5f * x * ((1.0f - tanh_out * tanh_out) *
                         (0.79788456f + 0.1070322243f * x * x)) +
             0.5f * (1.0f + tanh_out);
  return tmp * y_g;
}

template <int VecSize, bool FastMode>
static __global__ void FP16FastGeluFwdCUDAKernel(const __half* x, __half* y,
                                                 size_t n) {
  size_t offset =
      static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * VecSize;
  for (; offset < n; offset += stride) {
    using ArrT = platform::AlignedVector<__half, VecSize>;
    ArrT in_arr = *reinterpret_cast<const ArrT*>(x + offset);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      float tmp = __half2float(in_arr[i]);
      in_arr[i] = __float2half(FP32GeluFwd<FastMode>(tmp));
    }
    *reinterpret_cast<ArrT*>(y + offset) = in_arr;
  }
}

template <int VecSize, bool FastMode>
static __global__ void FP16FastGeluBwdCUDAKernel(const __half* x,
                                                 const __half* y_g, __half* x_g,
                                                 size_t n) {
  size_t offset =
      static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x) * VecSize;
  size_t stride = static_cast<size_t>(blockDim.x * gridDim.x) * VecSize;
  for (; offset < n; offset += stride) {
    using ArrT = platform::AlignedVector<__half, VecSize>;
    ArrT x_in_arr = *reinterpret_cast<const ArrT*>(x + offset);
    ArrT y_g_in_arr = *reinterpret_cast<const ArrT*>(y_g + offset);
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      __half2 tmp_fp16_2;
      tmp_fp16_2.x = x_in_arr[i];
      tmp_fp16_2.y = y_g_in_arr[i];
      float2 tmp_fp32_2 = __half22float2(tmp_fp16_2);
      x_in_arr[i] =
          __float2half(FP32GeluBwd<FastMode>(tmp_fp32_2.x, tmp_fp32_2.y));
    }
    *reinterpret_cast<ArrT*>(x_g + offset) = x_in_arr;
  }
}

static bool TryLaunchFP16FastGeluFwdVectorizeCUDAKernel(
    const platform::CUDADeviceContext& dev_ctx, const __half* x, __half* y,
    size_t n) {
  auto is_aligned = [](const void* p, size_t alignment) {
    return reinterpret_cast<uintptr_t>(p) % alignment == 0;
  };

#define PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL(__vec_size, __use_fast_math)      \
  do {                                                                        \
    constexpr auto kAlignment =                                               \
        alignof(platform::AlignedVector<__half, __vec_size>);                 \
    if (n % __vec_size == 0 && is_aligned(x, kAlignment) &&                   \
        is_aligned(y, kAlignment)) {                                          \
      size_t thread = std::min<size_t>(512, dev_ctx.GetMaxThreadsPerBlock()); \
      size_t block = (n / __vec_size + thread - 1) / thread;                  \
      block = std::min<size_t>(block, dev_ctx.GetCUDAMaxGridDimSize().x);     \
      VLOG(10) << "Use FP16 fast gelu fwd kernel, block = " << block          \
               << " , thread = " << thread;                                   \
      FP16FastGeluFwdCUDAKernel<                                              \
          __vec_size,                                                         \
          __use_fast_math><<<block, thread, 0, dev_ctx.stream()>>>(x, y, n);  \
      return true;                                                            \
    }                                                                         \
  } while (0)

  if (FLAGS_use_fast_math) {
    PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL(8, true);
  } else {
    PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL(8, false);
  }

#undef PD_LAUNCH_FP16_FAST_GELU_FWD_KERNEL
  return false;
}

static bool TryLaunchFP16FastGeluBwdVectorizeCUDAKernel(
    const platform::CUDADeviceContext& dev_ctx, const __half* x,
    const __half* y_g, __half* x_g, size_t n) {
  auto is_aligned = [](const void* p, size_t alignment) {
    return reinterpret_cast<uintptr_t>(p) % alignment == 0;
  };

#define PD_LAUNCH_FP16_FAST_GELU_BWD_KERNEL(__vec_size, __use_fast_math)      \
  do {                                                                        \
    constexpr auto kAlignment =                                               \
        alignof(platform::AlignedVector<__half, __vec_size>);                 \
    if (n % __vec_size == 0 && is_aligned(x, kAlignment) &&                   \
        is_aligned(x, kAlignment) && is_aligned(y_g, kAlignment) &&           \
        is_aligned(x_g, kAlignment)) {                                        \
      size_t thread = std::min<size_t>(512, dev_ctx.GetMaxThreadsPerBlock()); \
      size_t block = (n / __vec_size + thread - 1) / thread;                  \
      block = std::min<size_t>(block, dev_ctx.GetCUDAMaxGridDimSize().x);     \
      VLOG(10) << "Use FP16 fast gelu bwd kernel, block = " << block          \
               << " , thread = " << thread;                                   \
      FP16FastGeluBwdCUDAKernel<                                              \
          __vec_size,                                                         \
          __use_fast_math><<<block, thread, 0, dev_ctx.stream()>>>(x, y_g,    \
                                                                   x_g, n);   \
      return true;                                                            \
    }                                                                         \
  } while (0)

  if (FLAGS_use_fast_math) {
    PD_LAUNCH_FP16_FAST_GELU_BWD_KERNEL(8, true);
  } else {
    PD_LAUNCH_FP16_FAST_GELU_BWD_KERNEL(8, false);
  }

#undef PD_LAUNCH_FP16_FAST_GELU_BWD_KERNEL
  return false;
}
#endif

template <typename T>
struct GeluWithApproximateFunctor {
  using MPType = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x) {
    // this function is tanh approximation of gelu
    MPType x = static_cast<MPType>(arg_x);
    MPType one = static_cast<MPType>(1);
    MPType half = static_cast<MPType>(0.5);
    MPType kAlpha = static_cast<MPType>(M_2_SQRTPI * M_SQRT1_2);
    auto tanh_out =
        tanh(kAlpha * x * (one + static_cast<MPType>(GELU_CONSTANT) * x * x));
    MPType out = x * half * (one + tanh_out);
    return static_cast<T>(out);
  }
};

template <typename T>
struct GeluWithoutApproximateFunctor {
  using MPType = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x) {
    // actual gelu with approximation = false
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(x * normcdf(x));
  }
};

template <typename T>
class GeluKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    auto approximate = context.Attr<bool>("approximate");
    out->mutable_data<T>(in->place());

    std::vector<const framework::Tensor*> ins = {in};
    std::vector<framework::Tensor*> outs = {out};
    const auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    if (approximate) {
#ifdef __NVCC__
      if (std::is_same<T, platform::float16>::value) {
        size_t n = in->numel();
        const auto* in_ptr = reinterpret_cast<const __half*>(in->data<T>());
        auto* out_ptr = reinterpret_cast<__half*>(out->data<T>());
        if (TryLaunchFP16FastGeluFwdVectorizeCUDAKernel(dev_ctx, in_ptr,
                                                        out_ptr, n)) {
          return;
        }
      }
#endif
      paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary,
                                                     T, T>(
          dev_ctx, ins, &outs, 0, GeluWithApproximateFunctor<T>());
    } else {
      paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary,
                                                     T, T>(
          dev_ctx, ins, &outs, 0, GeluWithoutApproximateFunctor<T>());
    }
  }
};

template <typename T>
struct GeluWithApproximateGradFunctor {
  using MPType = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x, T arg_dout) {
    MPType x = static_cast<MPType>(arg_x);
    MPType dout = static_cast<MPType>(arg_dout);
    MPType one = static_cast<MPType>(1);
    MPType half = static_cast<MPType>(0.5);
    MPType kAlpha = static_cast<MPType>(M_2_SQRTPI * M_SQRT1_2);
    MPType kBeta =
        kAlpha * static_cast<MPType>(GELU_CONSTANT) * static_cast<MPType>(3);
    auto cube_x = x * x * x;
    auto tanh_out =
        tanh(kAlpha * ((static_cast<MPType>(GELU_CONSTANT) * cube_x) + x));
    auto ans =
        half * (one + tanh_out +
                (one - tanh_out * tanh_out) * (x * kAlpha + kBeta * cube_x));
    return static_cast<T>(ans * dout);
  }
};

template <typename T>
struct GeluWithoutApproximateGradFunctor {
  using MPType = typename details::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x, T arg_dout) {
    MPType x = static_cast<MPType>(arg_x);
    MPType dout = static_cast<MPType>(arg_dout);
    constexpr MPType kBeta = M_2_SQRTPI * M_SQRT1_2 * static_cast<MPType>(0.5);
    const MPType cdf = normcdf(x);
    const MPType pdf = exp(static_cast<MPType>(-0.5) * x * x) * kBeta;
    return static_cast<T>(dout * (cdf + x * pdf));
  }
};

template <typename T>
class GeluGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto approximate = context.Attr<bool>("approximate");
    dx->mutable_data<T>(dout->place());

    std::vector<const framework::Tensor*> ins = {x, dout};
    std::vector<framework::Tensor*> outs = {dx};
    const auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    if (approximate) {
#ifdef __NVCC__
      if (std::is_same<T, platform::float16>::value) {
        size_t n = x->numel();
        const auto* x_ptr = reinterpret_cast<const __half*>(x->data<T>());
        const auto* y_g_ptr = reinterpret_cast<const __half*>(dout->data<T>());
        auto* x_g_ptr = reinterpret_cast<__half*>(dx->data<T>());
        if (TryLaunchFP16FastGeluBwdVectorizeCUDAKernel(dev_ctx, x_ptr, y_g_ptr,
                                                        x_g_ptr, n)) {
          return;
        }
      }
#endif
      paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary,
                                                     T, T>(
          dev_ctx, ins, &outs, 0, GeluWithApproximateGradFunctor<T>());
    } else {
      paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary,
                                                     T, T>(
          dev_ctx, ins, &outs, 0, GeluWithoutApproximateGradFunctor<T>());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    gelu, ops::GeluKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GeluKernel<paddle::platform::CUDADeviceContext, double>,
    ops::GeluKernel<paddle::platform::CUDADeviceContext,
                    paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    gelu_grad, ops::GeluGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GeluGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::GeluGradKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::float16>);
