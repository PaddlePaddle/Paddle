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

#pragma once

#include <glog/logging.h>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/hostdevice.h"
#ifdef __HIPCC__
#define PADDLE_CUDA_THREAD_SIZE 256
#else
#define PADDLE_CUDA_THREAD_SIZE 512
#endif

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#ifdef PADDLE_CUDA_FP16
#include <cuda_fp16.h>
#endif
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#ifdef PADDLE_CUDA_FP16
#include <hip/hip_fp16.h>
#endif
#endif  // PADDLE_WITH_HIP

#define DIV_ERROR_INFO                                                     \
  "InvalidArgumentError: Integer division by zero encountered in divide. " \
  "Please check.\n"
namespace paddle {
namespace operators {

#define DEFINE_SIMPLE_BINARY_FUNCTOR(Func, expr)                   \
  template <typename T, class Enable = void>                       \
  struct Func##Functor {                                           \
    inline HOSTDEVICE T operator()(const T& a, const T& b) const { \
      return a expr b;                                             \
    }                                                              \
  };                                                               \
  template <typename T, class Enable = void>                       \
  struct Inverse##Func##Functor {                                  \
    inline HOSTDEVICE T operator()(const T& a, const T& b) const { \
      return b expr a;                                             \
    }                                                              \
  };

DEFINE_SIMPLE_BINARY_FUNCTOR(Add, +)
DEFINE_SIMPLE_BINARY_FUNCTOR(Sub, -)
DEFINE_SIMPLE_BINARY_FUNCTOR(Mul, *)
DEFINE_SIMPLE_BINARY_FUNCTOR(Div, /)
#undef DEFINE_SIMPLE_BINARY_FUNCTOR

// special div functor for int32/int64. check divison has a zero
template <typename T>
struct DivFunctor<T,
                  typename std::enable_if<std::is_integral<T>::value>::type> {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
    return a / b;
  }
};

#define DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR(Func, expr)                         \
  template <typename T, class Enable = void>                                  \
  struct Func##RangeFunctor {                                                 \
    Func##RangeFunctor(const T* x, const T* y, T* z) : x_(x), y_(y), z_(z) {} \
    inline HOSTDEVICE void operator()(size_t id) const {                      \
      z_[id] = x_[id] expr y_[id];                                            \
    }                                                                         \
    const T* x_;                                                              \
    const T* y_;                                                              \
    T* z_;                                                                    \
  };
DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR(Add, +)
DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR(Sub, -)
DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR(Mul, *)
DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR(Div, /)
#undef DEFINE_SIMPLE_CUDA_BINARY_FUNCTOR

// special div functor for int32/int64. check divison has a zero
template <typename T>
struct DivRangeFunctor<
    T, typename std::enable_if<std::is_integral<T>::value>::type> {
  DivRangeFunctor(const T* x, const T* y, T* z) : x_(x), y_(y), z_(z) {}
  inline HOSTDEVICE void operator()(size_t id) const {
    PADDLE_ENFORCE(y_[id] != 0, DIV_ERROR_INFO);
    z_[id] = x_[id] / y_[id];
  }
  const T* x_;
  const T* y_;
  T* z_;
};

#ifdef PADDLE_CUDA_FP16
inline DEVICE half2 half2_add(const half2& a, const half2& b) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  return __hadd2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 + b1;
  float r2 = a2 + b2;
  return __floats2half2_rn(r1, r2);
#endif
}

inline DEVICE half2 half2_sub(const half2& a, const half2& b) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  return __hsub2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 - b1;
  float r2 = a2 - b2;
  return __floats2half2_rn(r1, r2);
#endif
}

inline DEVICE half2 half2_mul(const half2& a, const half2& b) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  return __hmul2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 * b1;
  float r2 = a2 * b2;
  return __floats2half2_rn(r1, r2);
#endif
}

inline DEVICE half2 half2_div(const half2& a, const half2& b) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  return __h2div(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 / b1;
  float r2 = a2 / b2;
  return __floats2half2_rn(r1, r2);
#endif
}

#define DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Func, expr, FP16Function)             \
  inline __global__ void SameDimsElemwise##Func##CUDAKernel(                   \
      const float* __restrict__ x, const float* __restrict__ y, float* z,      \
      int64_t size) {                                                          \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int stride = gridDim.x * blockDim.x;                                       \
    int loop = size / 4;                                                       \
    int remainder = size % 4;                                                  \
    const float4* x_vec = reinterpret_cast<const float4*>(x);                  \
    const float4* y_vec = reinterpret_cast<const float4*>(y);                  \
    float4* z_vec = reinterpret_cast<float4*>(z);                              \
    float4 x_f4, y_f4;                                                         \
    for (int i = tid; i < loop; i += stride) {                                 \
      x_f4 = x_vec[i];                                                         \
      y_f4 = y_vec[i];                                                         \
      z_vec[i] = make_float4(x_f4.x expr y_f4.x, x_f4.y expr y_f4.y,           \
                             x_f4.z expr y_f4.z, x_f4.w expr y_f4.w);          \
    }                                                                          \
    if (tid == loop && remainder != 0) {                                       \
      while (remainder) {                                                      \
        int idx = size - remainder;                                            \
        remainder--;                                                           \
        z[idx] = x[idx] expr y[idx];                                           \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  inline __global__ void SameDimsElemwise##Func##CUDAKernel(                   \
      const half* __restrict__ x, const half* __restrict__ y, half* z,         \
      int64_t size) {                                                          \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                           \
    int stride = gridDim.x * blockDim.x;                                       \
    int loop = size / 8;                                                       \
    int remainder = size % 8;                                                  \
    const float4* x_vec = reinterpret_cast<const float4*>(x);                  \
    const float4* y_vec = reinterpret_cast<const float4*>(y);                  \
    float4* z_vec = reinterpret_cast<float4*>(z);                              \
    float4 x_h8, y_h8, z_h8;                                                   \
    for (int i = tid; i < loop; i += stride) {                                 \
      x_h8 = x_vec[i];                                                         \
      y_h8 = y_vec[i];                                                         \
      half2* x_h2 = reinterpret_cast<half2*>(&x_h8);                           \
      half2* y_h2 = reinterpret_cast<half2*>(&y_h8);                           \
      half2* z_h2 = reinterpret_cast<half2*>(&z_h8);                           \
      z_h2[0] = FP16Function(x_h2[0], y_h2[0]);                                \
      z_h2[1] = FP16Function(x_h2[1], y_h2[1]);                                \
      z_h2[2] = FP16Function(x_h2[2], y_h2[2]);                                \
      z_h2[3] = FP16Function(x_h2[3], y_h2[3]);                                \
      z_vec[i] = z_h8;                                                         \
    }                                                                          \
    if (tid == loop && remainder != 0) {                                       \
      while (remainder) {                                                      \
        int idx = size - remainder;                                            \
        remainder--;                                                           \
        z[idx] = __float2half(__half2float(x[idx]) expr __half2float(y[idx])); \
      }                                                                        \
    }                                                                          \
  }
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Add, +, half2_add)
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Sub, -, half2_sub)
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Mul, *, half2_mul)
DEFINE_SIMPLE_CUDA_BINARY_KERNEL(Div, /, half2_div)
#undef DEFINE_SIMPLE_CUDA_BINARY_KERNEL

#endif  // PADDLE_CUDA_FP16

}  // namespace operators
}  // namespace paddle
