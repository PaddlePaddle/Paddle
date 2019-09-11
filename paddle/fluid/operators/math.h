// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/hostdevice.h"

#include "math.h"  // NOLINT

#if defined(__CUDACC__) && CUDA_VERSION >= 7050
#define PADDLE_CUDA_FP16
#include <cuda_fp16.h>
#endif

namespace paddle {
namespace operators {

inline HOSTDEVICE platform::float16 real_exp(platform::float16 x) {
  return static_cast<platform::float16>(::expf(static_cast<float>(x)));
}

inline HOSTDEVICE float real_exp(float x) { return ::expf(x); }

inline HOSTDEVICE double real_exp(double x) { return ::exp(x); }

inline HOSTDEVICE platform::float16 real_log(platform::float16 x) {
  return static_cast<platform::float16>(::logf(static_cast<float>(x)));
}

inline HOSTDEVICE float real_log(float x) { return ::logf(x); }

inline HOSTDEVICE double real_log(double x) { return ::log(x); }

inline HOSTDEVICE float real_min(float x, float y) { return ::fminf(x, y); }

inline HOSTDEVICE double real_min(double x, double y) { return ::fmin(x, y); }

#define DEFINE_SIMPLE_BINARY_FUNCTOR(Func, expr)                   \
  template <typename T>                                            \
  struct Func##Functor {                                           \
    inline HOSTDEVICE T operator()(const T& a, const T& b) const { \
      return a expr b;                                             \
    }                                                              \
  };

DEFINE_SIMPLE_BINARY_FUNCTOR(Add, +)
DEFINE_SIMPLE_BINARY_FUNCTOR(Sub, -)
DEFINE_SIMPLE_BINARY_FUNCTOR(Mul, *)
DEFINE_SIMPLE_BINARY_FUNCTOR(Div, /)
#undef DEFINE_SIMPLE_BINARY_FUNCTOR

#if defined(PADDLE_CUDA_FP16)
inline DEVICE half2 half2_add(const half2& a, const half2& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hadd2(a, b);
#else
  float2 fa, fb, fo;
  fa = __half22float2(a);
  fb = __half22float2(b);
  fo.x = fa.x + fb.x;
  fo.y = fa.y + fb.y;
  return __float22half2_rn(fo);
#endif
}

inline DEVICE half2 half2_sub(const half2& a, const half2& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hsub2(a, b);
#else
  float2 fa, fb, fo;
  fa = __half22float2(a);
  fb = __half22float2(b);
  fo.x = fa.x - fb.x;
  fo.y = fa.y - fb.y;
  return __float22half2_rn(fo);
#endif
}

inline DEVICE half2 half2_mul(const half2& a, const half2& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hmul2(a, b);
#else
  float2 fa, fb, fo;
  fa = __half22float2(a);
  fb = __half22float2(b);
  fo.x = fa.x * fb.x;
  fo.y = fa.y * fb.y;
  return __float22half2_rn(fo);
#endif
}

inline DEVICE half2 half2_div(const half2& a, const half2& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __h2div(a, b);
#else
  float2 fa, fb, fo;
  fa = __half22float2(a);
  fb = __half22float2(b);
  fo.x = fa.x / fb.x;
  fo.y = fa.y / fb.y;
  return __float22half2_rn(fo);
#endif
}
#endif  // PADDLE_CUDA_FP16

}  // namespace operators
}  // namespace paddle
