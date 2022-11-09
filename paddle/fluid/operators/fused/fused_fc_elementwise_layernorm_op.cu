/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#if defined(PADDLE_WITH_CUDA)
#include <cuda_fp16.h>
#endif

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

using float16 = phi::dtype::float16;

template <typename T>
static __device__ __forceinline__ T Relu(T x) {
  return static_cast<T>(fmaxf(0.f, x));
}

static __device__ __forceinline__ float RealSqrt(float x) { return sqrtf(x); }
static __device__ __forceinline__ double RealSqrt(double x) { return sqrt(x); }

template <typename T>
struct PairForLayerNorm {
  __device__ __forceinline__ PairForLayerNorm() {}
  __device__ __forceinline__ PairForLayerNorm(const T& first, const T& second)
      : first_(first), second_(second) {}

  T first_;
  T second_;
};

template <typename T>
struct PairForLayerNormAddFunctor {
  __device__ __forceinline__ PairForLayerNorm<T> operator()(
      const PairForLayerNorm<T>& p1, const PairForLayerNorm<T>& p2) {
    return PairForLayerNorm<T>(p1.first_ + p2.first_, p1.second_ + p2.second_);
  }
};

template <typename T, bool DoRelu, int BlockDim>
__global__ void InplaceAddReluAddLayerNormKernel(const T* y,
                                                 const T* bias_0,
                                                 const T* bias_1,
                                                 const T* scale,
                                                 T* out,
                                                 T* mean,
                                                 T* variance,
                                                 int M,
                                                 int N,
                                                 float epsilon) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<T>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T shared_mem[BlockDim + 2];

  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    int index = i * N + threadIdx.x;

    // The fisrt BlockDim elements will be saved to shared memory.
    int save_index = threadIdx.x;
    T* save_ptr = shared_mem;

    T sum_i = 0;
    T square_sum_i = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      T tmp_0 = out[index];
      // Add bias
      T tmp_1 = bias_0 ? tmp_0 + bias_0[j] : tmp_0;
      // Relu
      T tmp_2 = DoRelu ? Relu(tmp_1) : tmp_1;
      // elementwise_add
      T tmp_3 = tmp_2 + y[index];

      // Save
      save_ptr[save_index] = tmp_3;
      save_ptr = out;

      index += blockDim.x;
      save_index = index;

      // For layer_norm, reduce to calculate mean and std
      sum_i += tmp_3;
      square_sum_i += (tmp_3 * tmp_3);
    }

    auto pair = BlockReduce(temp_storage)
                    .Reduce(PairForLayerNorm<T>(sum_i, square_sum_i),
                            PairForLayerNormAddFunctor<T>());

    if (threadIdx.x == 0) {
      T mean_i = static_cast<T>(pair.first_ / N);
      T variance_i = static_cast<T>(pair.second_ / N - mean_i * mean_i);
      shared_mem[BlockDim] = mean_i;
      shared_mem[BlockDim + 1] = variance_i;
      if (mean) {
        mean[blockIdx.x] = mean_i;
      }
      if (variance) {
        variance[blockIdx.x] = variance_i;
      }
    }
    __syncthreads();
    T mean_i = shared_mem[BlockDim];
    T std_i = static_cast<T>(RealSqrt(shared_mem[BlockDim + 1] + epsilon));

    index = i * N + threadIdx.x;
    // First BlockDim elements loading from shared memory.
    save_index = threadIdx.x;
    save_ptr = shared_mem;

    // For layer_norm, calculate out
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      T tmp_0 = (save_ptr[save_index] - mean_i) / std_i;
      T tmp_1 = scale ? scale[j] * tmp_0 : tmp_0;
      out[index] = bias_1 ? tmp_1 + bias_1[j] : tmp_1;

      save_ptr = out;
      index += blockDim.x;
      save_index = index;
    }
  }
}

template <bool DoRelu, int BlockDim>
__global__ void InplaceAddReluAddLayerNormKernel(const float16* y_data,
                                                 const float16* bias_0_data,
                                                 const float16* bias_1_data,
                                                 const float16* scale_data,
                                                 float16* out_data,
                                                 float16* mean_data,
                                                 float16* variance_data,
                                                 int M,
                                                 int N,
                                                 float epsilon) {
#if defined(PADDLE_WITH_CUDA)
  const half* y = reinterpret_cast<const half*>(y_data);
  const half* bias_0 = reinterpret_cast<const half*>(bias_0_data);
  const half* bias_1 = reinterpret_cast<const half*>(bias_1_data);
  const half* scale = reinterpret_cast<const half*>(scale_data);
  half* out = reinterpret_cast<half*>(out_data);
  half* mean = reinterpret_cast<half*>(mean_data);
  half* variance = reinterpret_cast<half*>(variance_data);
#else
  const float16* y = y_data;
  const float16* bias_0 = bias_0_data;
  const float16* bias_1 = bias_1_data;
  const float16* scale = scale_data;
  float16* out = out_data;
  float16* mean = mean_data;
  float16* variance = variance_data;
#endif
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<float>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
#if defined(PADDLE_WITH_CUDA)
  __shared__ half shared_mem[BlockDim + 2];
#else
  __shared__ float16 shared_mem[BlockDim + 2];
#endif

  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    int index = i * N + threadIdx.x;

    // The fisrt BlockDim elements will be saved to shared memory.
    int save_index = threadIdx.x;
#if defined(PADDLE_WITH_CUDA)
    half* save_ptr = shared_mem;
#else
    float16* save_ptr = shared_mem;
#endif
    float sum_i = 0;
    float square_sum_i = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
#if defined(PADDLE_WITH_CUDA)
      half tmp_0 = out[index];
      // Add bias
      half tmp_1;
      if (bias_0 != nullptr) {
        tmp_1 = __hadd(tmp_0, bias_0[j]);
      } else {
        tmp_1 = tmp_0;
      }
      // Relu
      half tmp_2 = DoRelu ? Relu(tmp_1) : tmp_1;
      // elementwise_add
      half tmp_3 = __hadd(tmp_2, y[index]);
#else
      float16 tmp_0 = out[index];
      // Add bias
      float16 tmp_1 = bias_0 ? tmp_0 + bias_0[j] : tmp_0;
      // Relu
      float16 tmp_2 = DoRelu ? Relu(tmp_1) : tmp_1;
      // elementwise_add
      float16 tmp_3 = tmp_2 + y[index];
#endif
      // Save
      save_ptr[save_index] = tmp_3;
      save_ptr = out;

      index += blockDim.x;
      save_index = index;

      // For layer_norm, reduce to calculate mean and std
      sum_i += static_cast<float>(tmp_3);
#if defined(PADDLE_WITH_CUDA) && __CUDA_ARCH__ >= 530
      square_sum_i += static_cast<float>(__hmul(tmp_3, tmp_3));
#elif defined(PADDLE_WITH_CUDA)
      square_sum_i += static_cast<float>(tmp_3) * static_cast<float>(tmp_3);
#else
      square_sum_i += static_cast<float>(tmp_3 * tmp_3);
#endif
    }
    auto pair = BlockReduce(temp_storage)
                    .Reduce(PairForLayerNorm<float>(sum_i, square_sum_i),
                            PairForLayerNormAddFunctor<float>());
    if (threadIdx.x == 0) {
#if defined(PADDLE_WITH_CUDA)
      half mean_i = static_cast<half>(pair.first_ / N);
#if __CUDA_ARCH__ >= 530
      half variance_i = static_cast<half>(
          pair.second_ / N - static_cast<float>(__hmul(mean_i, mean_i)));
#else
      half variance_i =
          static_cast<half>(pair.second_ / N - static_cast<float>(mean_i) *
                                                   static_cast<float>(mean_i));
#endif
#else
      float16 mean_i = static_cast<float16>(pair.first_ / N);
      float16 variance_i = static_cast<float16>(
          pair.second_ / N - static_cast<float>(mean_i * mean_i));
#endif
      shared_mem[BlockDim] = mean_i;
      shared_mem[BlockDim + 1] = variance_i;
      if (mean) {
        mean[blockIdx.x] = mean_i;
      }
      if (variance) {
        variance[blockIdx.x] = variance_i;
      }
    }
    __syncthreads();
#if defined(PADDLE_WITH_CUDA)
    half mean_i = shared_mem[BlockDim];
    half std_i = static_cast<half>(
        RealSqrt(static_cast<float>(shared_mem[BlockDim + 1]) + epsilon));
#else
    float16 mean_i = shared_mem[BlockDim];
    float16 std_i = static_cast<float16>(
        RealSqrt(static_cast<float>(shared_mem[BlockDim + 1]) + epsilon));
#endif

    index = i * N + threadIdx.x;
    // First BlockDim elements loading from shared memory.
    save_index = threadIdx.x;
    save_ptr = shared_mem;

    // For layer_norm, calculate out
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
#if defined(PADDLE_WITH_CUDA)
#if __CUDA_ARCH__ >= 530
      half tmp_0 = __hdiv(__hsub(save_ptr[save_index], mean_i), std_i);
      half tmp_1 = scale ? __hmul(scale[j], tmp_0) : tmp_0;
#else
      half tmp_0 = static_cast<float>(static_cast<float>(save_ptr[save_index]) +
                                      static_cast<float>(mean_i) /
                                          static_cast<float>(std_i));
      half tmp_1 = scale ? static_cast<half>(static_cast<float>(scale[j]) *
                                             static_cast<float>(tmp_0))
                         : tmp_0;
#endif
      if (bias_1 != nullptr) {
        out[index] = __hadd(tmp_1, bias_1[j]);
      } else {
        out[index] = tmp_1;
      }
#else
      float16 tmp_0 = (save_ptr[save_index] - mean_i) / std_i;
      float16 tmp_1 = scale ? scale[j] * tmp_0 : tmp_0;
      out[index] = bias_1 ? tmp_1 + bias_1[j] : tmp_1;
#endif
      save_ptr = out;
      index += blockDim.x;
      save_index = index;
    }
  }
}

template <typename T>
void AddReluAddLayerNorm(gpuStream_t stream,
                         bool with_relu,
                         int max_threads,
                         const T* y,
                         const T* bias_0,
                         const T* bias_1,
                         const T* scale,
                         T* out,
                         T* mean,
                         T* variance,
                         int M,
                         int N,
                         float epsilon) {
  if (with_relu) {
    switch (platform::RoundToPowerOfTwo(N)) {
      CUDA_LAUNCH_KERNEL_HELPER(
          InplaceAddReluAddLayerNormKernel<T, true, kPowerOfTwoDim>
          <<<std::max(max_threads / kPowerOfTwoDim, 1),
             kPowerOfTwoDim,
             0,
             stream>>>(
              y, bias_0, bias_1, scale, out, mean, variance, M, N, epsilon));
    }
  } else {
    switch (platform::RoundToPowerOfTwo(N)) {
      CUDA_LAUNCH_KERNEL_HELPER(
          InplaceAddReluAddLayerNormKernel<T, false, kPowerOfTwoDim>
          <<<std::max(max_threads / kPowerOfTwoDim, 1),
             kPowerOfTwoDim,
             0,
             stream>>>(
              y, bias_0, bias_1, scale, out, mean, variance, M, N, epsilon));
    }
  }
}

template <>
void AddReluAddLayerNorm(gpuStream_t stream,
                         bool with_relu,
                         int max_threads,
                         const float16* y,
                         const float16* bias_0,
                         const float16* bias_1,
                         const float16* scale,
                         float16* out,
                         float16* mean,
                         float16* variance,
                         int M,
                         int N,
                         float epsilon) {
  if (with_relu) {
    switch (platform::RoundToPowerOfTwo(N)) {
      CUDA_LAUNCH_KERNEL_HELPER(
          InplaceAddReluAddLayerNormKernel<true, kPowerOfTwoDim>
          <<<std::max(max_threads / kPowerOfTwoDim, 1),
             kPowerOfTwoDim,
             0,
             stream>>>(
              y, bias_0, bias_1, scale, out, mean, variance, M, N, epsilon));
    }
  } else {
    switch (platform::RoundToPowerOfTwo(N)) {
      CUDA_LAUNCH_KERNEL_HELPER(
          InplaceAddReluAddLayerNormKernel<false, kPowerOfTwoDim>
          <<<std::max(max_threads / kPowerOfTwoDim, 1),
             kPowerOfTwoDim,
             0,
             stream>>>(
              y, bias_0, bias_1, scale, out, mean, variance, M, N, epsilon));
    }
  }
}

template <typename T>
class FusedFCElementwiseLayerNormOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* w = ctx.Input<phi::DenseTensor>("W");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto w_dims = w->dims();
    int N = w_dims[1];
    int K = w_dims[0];
    int M = phi::product(x->dims()) / K;

    const T* x_data = x->data<T>();
    const T* w_data = w->data<T>();

    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto* out_data = dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
    blas.GEMM(false,
              false,
              M,
              N,
              K,
              static_cast<T>(1.0),
              x_data,
              K,
              w_data,
              N,
              static_cast<T>(0.0),
              out_data,
              N);
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* bias_0 = ctx.Input<phi::DenseTensor>("Bias0");
    auto* bias_1 = ctx.Input<phi::DenseTensor>("Bias1");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");

    const T* y_data = y->data<T>();
    const T* bias_0_data = bias_0 ? bias_0->data<T>() : nullptr;
    const T* bias_1_data = bias_1 ? bias_1->data<T>() : nullptr;
    const T* scale_data = scale ? scale->data<T>() : nullptr;

    auto* mean = ctx.Output<phi::DenseTensor>("Mean");
    auto* variance = ctx.Output<phi::DenseTensor>("Variance");

    T* mean_data =
        mean ? dev_ctx.template Alloc<T>(mean, mean->numel() * sizeof(T))
             : nullptr;
    T* variance_data = variance ? dev_ctx.template Alloc<T>(
                                      variance, variance->numel() * sizeof(T))
                                : nullptr;

    bool with_relu =
        (ctx.Attr<std::string>("activation_type") == "relu") ? true : false;
    float epsilon = ctx.Attr<float>("epsilon");

    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    AddReluAddLayerNorm(dev_ctx.stream(),
                        with_relu,
                        max_threads,
                        y_data,
                        bias_0_data,
                        bias_1_data,
                        scale_data,
                        out_data,
                        mean_data,
                        variance_data,
                        M,
                        N,
                        epsilon);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_fc_elementwise_layernorm,
    ops::FusedFCElementwiseLayerNormOpKernel<phi::dtype::float16>,
    ops::FusedFCElementwiseLayerNormOpKernel<float>,
    ops::FusedFCElementwiseLayerNormOpKernel<double>);
