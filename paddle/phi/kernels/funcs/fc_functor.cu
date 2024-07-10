/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#include "paddle/phi/kernels/funcs/quant_dequant.h"
#include "paddle/phi/kernels/matmul_kernel.h"

namespace phi {
namespace funcs {

using float16 = phi::dtype::float16;

template <typename T>
struct FcTypeTraits;

template <>
struct FcTypeTraits<float> {
  typedef float4 Type;
};

template <>
struct FcTypeTraits<double> {
  typedef double4 Type;
};

#if defined(PADDLE_WITH_CUDA)
#include <cuda_fp16.h>

template <>
struct FcTypeTraits<float16> {
  typedef half2 Type;
};
#else
struct float16_4 {
  float16 x, y, z, w;
};

template <>
struct FcTypeTraits<float16> {
  typedef float16_4 Type;
};
#endif

template <typename T, bool DoRelu>
__global__ void bias_relu_v4(const int num, const T* bias, T* data, int K) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int bias_idx = tid % K;
    const T bias_ptr = bias[bias_idx];
    const T in_ptr = data[tid];
    T packed_val;
    packed_val.x = in_ptr.x + bias_ptr.x;
    packed_val.y = in_ptr.y + bias_ptr.y;
    packed_val.z = in_ptr.z + bias_ptr.z;
    packed_val.w = in_ptr.w + bias_ptr.w;
    if (DoRelu) {
      packed_val.x = fmaxf(0.f, packed_val.x);
      packed_val.y = fmaxf(0.f, packed_val.y);
      packed_val.z = fmaxf(0.f, packed_val.z);
      packed_val.w = fmaxf(0.f, packed_val.w);
    }
    data[tid] = packed_val;
  }
}

template <typename T, bool DoRelu, int BlockDim>
__global__ void InplaceAddReluKernel(const int N, const T* bias, T* data) {
  int offset = blockIdx.x * N;

  for (int i = threadIdx.x; i < N; i += BlockDim) {
    T temp;
#if defined(__HIPCC__) || __CUDA_ARCH__ >= 350
    temp = __ldg(data + offset + i) + __ldg(bias + i);
#else
    temp = data[offset + i] + bias[i];
#endif
    if (DoRelu) {
      data[offset + i] = static_cast<int>(temp > 0) * temp;
    } else {
      data[offset + i] = temp;
    }
  }
}

template <typename T>
void AddReluKernel(
    gpuStream_t stream, const int M, const int N, T* Y, const T* B, bool relu) {
  if (N % 4 == 0) {
    const int threads = 256;
    const int num = M * N / 4;
    const int blocks = (num + threads - 1) / threads;
    typedef typename FcTypeTraits<T>::Type trans_type;
    auto* bias_ptr_v4 = reinterpret_cast<const trans_type*>(B);
    auto* data_ptr_v4 = reinterpret_cast<trans_type*>(Y);
    if (relu) {
      bias_relu_v4<trans_type, true><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v4, data_ptr_v4, N / 4);
    } else {
      bias_relu_v4<trans_type, false><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v4, data_ptr_v4, N / 4);
    }
  } else {
    const int threads = 256;
    const int blocks = M;

    if (relu) {
      InplaceAddReluKernel<T, true, threads>
          <<<blocks, threads, 0, stream>>>(N, B, Y);
    } else {
      InplaceAddReluKernel<T, false, threads>
          <<<blocks, threads, 0, stream>>>(N, B, Y);
    }
  }
}

#if defined(PADDLE_WITH_CUDA)
template <bool DoRelu, int Half2VecSize>
__global__ void bias_relu_v4_half2(const int num,
                                   const half2* bias,
                                   half2* data,
                                   int K) {
  using LoadT = phi::AlignedVector<half2, Half2VecSize>;
  LoadT data_vec;
  LoadT bias_vec;
  const int32_t global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t grid_stride = gridDim.x * blockDim.x;

  for (int32_t linear_idx = global_thread_idx * Half2VecSize; linear_idx < num;
       linear_idx += grid_stride * Half2VecSize) {
    phi::Load<half2, Half2VecSize>(&data[linear_idx], &data_vec);
    const int bias_idx = linear_idx % K;
    phi::Load<half2, Half2VecSize>(&bias[bias_idx], &bias_vec);

#pragma unroll
    for (int unroll_idx = 0; unroll_idx < Half2VecSize; unroll_idx++) {
// Do biasAdd
#if __CUDA_ARCH__ >= 530
      data_vec[unroll_idx] =
          __hadd2(data_vec[unroll_idx], bias_vec[unroll_idx]);
#else
      data_vec[unroll_idx].x =
          __hadd(data_vec[unroll_idx].x, bias_vec[unroll_idx].x);
      data_vec[unroll_idx].y =
          __hadd(data_vec[unroll_idx].y, bias_vec[unroll_idx].y);
#endif

      // Do relu
      if (DoRelu) {
#if __CUDA_ARCH__ >= 800
        data_vec[unroll_idx] = __hmax2(__half2(0, 0), data_vec[unroll_idx]);
#elif __CUDA_ARCH__ >= 530
        data_vec[unroll_idx] = __hmul2(
            __hgt2(data_vec[unroll_idx], __half2(0, 0)), data_vec[unroll_idx]);
#else
        data_vec[unroll_idx].x =
            static_cast<int>(static_cast<float>(data_vec[unroll_idx].x) > 0) *
            static_cast<float>(data_vec[unroll_idx].x);
        data_vec[unroll_idx].y =
            static_cast<int>(static_cast<float>(data_vec[unroll_idx].y) > 0) *
            static_cast<float>(data_vec[unroll_idx].y);
#endif
      }
    }
    phi::Store<half2, Half2VecSize>(data_vec, &data[linear_idx]);
  }
}

template <bool DoRelu, int BlockDim>
__global__ void InplaceAddReluKernel(const int N,
                                     const half* bias,
                                     half* data) {
  int offset = blockIdx.x * N;
  for (int i = threadIdx.x; i < N; i += BlockDim) {
    half temp;
#if defined(__HIPCC__) || __CUDA_ARCH__ >= 350
    temp = __hadd(__ldg(data + offset + i), __ldg(bias + i));
#else
    temp = __hadd(data[offset + i], bias[i]);
#endif
    if (DoRelu) {
#if __CUDA_ARCH__ >= 800
      data[offset + i] = __hmax(0, temp);
#elif __CUDA_ARCH__ >= 530
      data[offset + i] = __hmul(__hgt(temp, 0), temp);
#else
      data[offset + i] = static_cast<int>(static_cast<float>(temp) > 0) *
                         static_cast<float>(temp);
#endif
    } else {
      data[offset + i] = temp;
    }
  }
}

/**
 * brief: Launch BiasAddReluKernel with relu or not.
 **/
template <int Half2VecSize>
void LaunchBiasAddReluHalf2Kernel(cudaStream_t stream,
                                  const int32_t rows,
                                  const int32_t cols,
                                  float16* Y,
                                  const float16* B,
                                  bool relu) {
  const int threads = 256;
  const int vec_num = rows * cols / (Half2VecSize * 2);
  const int half2_num = rows * cols / 2;
  const int blocks = (vec_num + threads - 1) / threads;
  // Here reinterpret_cast to half2 type.
  typedef typename FcTypeTraits<float16>::Type trans_type;
  auto* bias_half2_ptr = reinterpret_cast<const trans_type*>(B);
  auto* data_half2_ptr = reinterpret_cast<trans_type*>(Y);
  if (relu) {
    bias_relu_v4_half2<true, Half2VecSize><<<blocks, threads, 0, stream>>>(
        half2_num, bias_half2_ptr, data_half2_ptr, cols / 2);
  } else {
    bias_relu_v4_half2<false, Half2VecSize><<<blocks, threads, 0, stream>>>(
        half2_num, bias_half2_ptr, data_half2_ptr, cols / 2);
  }
}

/**
 * brief: Dispatch BiasAddReluKernel half2 type with 8 / 4 / 2 vecsize.
 **/
void DispatchBiasAddReluKernelHalf2VecSize(cudaStream_t stream,
                                           const int32_t rows,
                                           const int32_t cols,
                                           float16* Y,
                                           const float16* B,
                                           bool relu) {
  // Half Max Vecsize is 128 / 16 = 8, since we use half2 type, here
  // Half2VecSize need divide 2.
  if (cols % 8 == 0) {
    LaunchBiasAddReluHalf2Kernel<4>(stream, rows, cols, Y, B, relu);
  } else if (cols % 4 == 0) {
    LaunchBiasAddReluHalf2Kernel<2>(stream, rows, cols, Y, B, relu);
  } else {
    LaunchBiasAddReluHalf2Kernel<1>(stream, rows, cols, Y, B, relu);
  }
}

template <>
void AddReluKernel(cudaStream_t stream,
                   const int M,
                   const int N,
                   float16* Y,
                   const float16* B,
                   bool relu) {
  if (N % 2 == 0) {
    DispatchBiasAddReluKernelHalf2VecSize(stream, M, N, Y, B, relu);
  } else {
    const int threads = 256;
    const int blocks = M;
    auto* halfB = reinterpret_cast<const half*>(B);
    auto* halfY = reinterpret_cast<half*>(Y);
    if (relu) {
      InplaceAddReluKernel<true, threads>
          <<<blocks, threads, 0, stream>>>(N, halfB, halfY);
    } else {
      InplaceAddReluKernel<false, threads>
          <<<blocks, threads, 0, stream>>>(N, halfB, halfY);
    }
  }
}
#else
template <bool DoRelu, int BlockDim>
__global__ void InplaceAddReluKernel(const int N,
                                     const float16* bias,
                                     float16* data) {
  int offset = blockIdx.x * N;
  for (int i = threadIdx.x; i < N; i += BlockDim) {
    float16 temp;
    temp = data[offset + i] + bias[i];
    if (DoRelu) {
      data[offset + i] = fmaxf(0.f, temp);
    } else {
      data[offset + i] = temp;
    }
  }
}

template <>
void AddReluKernel(gpuStream_t stream,
                   const int M,
                   const int N,
                   float16* Y,
                   const float16* B,
                   bool relu) {
  if (N % 4 == 0) {
    const int threads = 256;
    const int num = M * N / 4;
    const int blocks = (num + threads - 1) / threads;
    typedef typename FcTypeTraits<float16>::Type trans_type;
    auto* bias_ptr_v4 = reinterpret_cast<const trans_type*>(B);
    auto* data_ptr_v4 = reinterpret_cast<trans_type*>(Y);
    if (relu) {
      bias_relu_v4<trans_type, true><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v4, data_ptr_v4, N / 4);
    } else {
      bias_relu_v4<trans_type, false><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v4, data_ptr_v4, N / 4);
    }
  } else {
    const int threads = 256;
    const int blocks = M;

    if (relu) {
      InplaceAddReluKernel<true, threads>
          <<<blocks, threads, 0, stream>>>(N, B, Y);
    } else {
      InplaceAddReluKernel<false, threads>
          <<<blocks, threads, 0, stream>>>(N, B, Y);
    }
  }
}
#endif

template <typename DeviceContext, typename T>
void FCFunctor<DeviceContext, T>::operator()(const DeviceContext& context,
                                             const int M,
                                             const int N,
                                             const int K,
                                             const T* X,
                                             const T* W,
                                             T* Y,
                                             const T* B,
                                             bool relu,
                                             bool padding_weights) {
  PADDLE_ENFORCE_EQ(padding_weights,
                    false,
                    errors::PermissionDenied(
                        "Weight padding in fc can not be used in GPU scope."));
  auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);
  blas.GEMM(CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            K,
            static_cast<T>(1.0),
            X,
            W,
            static_cast<T>(0.0),
            Y);
  if (B == NULL) {
    return;
  }

  // M * N
  AddReluKernel(context.stream(), M, N, Y, B, relu);
}

template class FCFunctor<GPUContext, float16>;
template class FCFunctor<GPUContext, float>;
template class FCFunctor<GPUContext, double>;

template <typename DeviceContext, typename T>
void FCInt8Functor<DeviceContext, T>::operator()(
    const DeviceContext& context,
    const int M,
    const int N,
    const int K,
    const T* X,
    const DenseTensor* w_tensor,
    T* Y,
    float scale_in,
    std::vector<float> scale_weights,
    int quant_round_type,
    float quant_max_bound,
    float quant_min_bound,
    const T* B,
    bool relu,
    bool padding_weights) {
  PADDLE_ENFORCE_EQ(padding_weights,
                    false,
                    errors::PermissionDenied(
                        "Weight padding in fc can not be used in GPU scope."));
  const int8_t* W = w_tensor->data<int8_t>();

  DenseTensor quant_x_tensor, quant_y_tensor;
  quant_x_tensor.Resize(common::make_ddim({M, K}));
  quant_y_tensor.Resize(common::make_ddim({M, N}));
  context.template Alloc<int8_t>(&quant_x_tensor,
                                 quant_x_tensor.numel() * sizeof(int8_t));
  context.template Alloc<int32_t>(&quant_y_tensor,
                                  quant_y_tensor.numel() * sizeof(int32_t));
  LaunchQuantKernelWithVecSize<T>(X,
                                  quant_x_tensor.data<int8_t>(),
                                  scale_in,
                                  M,
                                  K,
                                  quant_round_type,
                                  quant_max_bound,
                                  quant_min_bound,
                                  context.stream());

  MatmulKernel<int8_t, GPUContext>(
      context, quant_x_tensor, *w_tensor, false, false, &quant_y_tensor);

  DenseTensor scale_weights_dev;
  scale_weights_dev.Resize(common::make_ddim({N}));
  context.template Alloc<float>(&scale_weights_dev,
                                scale_weights_dev.numel() * sizeof(float));
  float* scale_weights_dev_ptr = scale_weights_dev.data<float>();
#ifdef PADDLE_WITH_HIP
  hipMemcpyAsync(scale_weights_dev_ptr,
                 scale_weights.data(),
                 N * sizeof(float),
                 hipMemcpyHostToDevice);
#else
  cudaMemcpyAsync(scale_weights_dev_ptr,
                  scale_weights.data(),
                  N * sizeof(float),
                  cudaMemcpyHostToDevice);
#endif

  phi::backends::gpu::GpuLaunchConfig config;
  if (N % DequantKernelVecSize == 0) {
    config = phi::backends::gpu::GetGpuLaunchConfig1D(
        context, M * N, DequantKernelVecSize);
  } else {
    config = phi::backends::gpu::GetGpuLaunchConfig1D(context, M * N, 1);
  }
  LaunchDequantKernelWithScaleOfInputAndWeight(quant_y_tensor.data<int32_t>(),
                                               Y,
                                               M,
                                               N,
                                               context.stream(),
                                               &config,
                                               scale_in,
                                               scale_weights_dev_ptr,
                                               quant_max_bound);

  if (B == NULL) {
    return;
  }

  // M * N
  AddReluKernel(context.stream(), M, N, Y, B, relu);
}

template class FCInt8Functor<GPUContext, float16>;
template class FCInt8Functor<GPUContext, float>;
template class FCInt8Functor<GPUContext, double>;

}  // namespace funcs
}  // namespace phi
