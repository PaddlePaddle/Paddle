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

#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"

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
template <bool DoRelu>
__global__ void bias_relu_v2(const int num,
                             const half2* bias,
                             half2* data,
                             int K) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num) {
    int bias_idx = tid % K;
    const half2 bias_ptr = bias[bias_idx];
    const half2 in_ptr = data[tid];
    half2 packed_val;
#if __CUDA_ARCH__ >= 530
    packed_val = __hadd2(bias_ptr, in_ptr);
#else
    packed_val.x = __hadd(bias_ptr.x, in_ptr.x);
    packed_val.y = __hadd(bias_ptr.y, in_ptr.y);
#endif
    if (DoRelu) {
#if __CUDA_ARCH__ >= 800
      packed_val = __hmax2(__half2(0, 0), packed_val);
#elif __CUDA_ARCH__ >= 530
      packed_val = __hmul2(__hgt2(__half2(0, 0), packed_val), packed_val);
#else
      packed_val.x = static_cast<int>(static_cast<float>(packed_val.x) > 0) *
                     static_cast<float>(packed_val.x);
      packed_val.y = static_cast<int>(static_cast<float>(packed_val.y) > 0) *
                     static_cast<float>(packed_val.y);
#endif
    }
    data[tid] = packed_val;
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

template <>
void AddReluKernel(cudaStream_t stream,
                   const int M,
                   const int N,
                   float16* Y,
                   const float16* B,
                   bool relu) {
  if (N % 2 == 0) {
    const int threads = 256;
    const int num = M * N / 2;
    const int blocks = (num + threads - 1) / threads;
    typedef typename FcTypeTraits<float16>::Type trans_type;
    auto* bias_ptr_v2 = reinterpret_cast<const trans_type*>(B);
    auto* data_ptr_v2 = reinterpret_cast<trans_type*>(Y);
    if (relu) {
      bias_relu_v2<true><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v2, data_ptr_v2, N / 2);
    } else {
      bias_relu_v2<false><<<blocks, threads, 0, stream>>>(
          num, bias_ptr_v2, data_ptr_v2, N / 2);
    }
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
  blas.GEMM(false,
            false,
            M,
            N,
            K,
            static_cast<T>(1.0),
            X,
            K,
            W,
            N,
            static_cast<T>(0.0),
            Y,
            N);
  if (B == NULL) {
    return;
  }

  // M * N
  AddReluKernel(context.stream(), M, N, Y, B, relu);
}

template class FCFunctor<GPUContext, float16>;
template class FCFunctor<GPUContext, float>;
template class FCFunctor<GPUContext, double>;

}  // namespace funcs
}  // namespace phi
