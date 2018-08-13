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
#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#if !defined(__APPLE__) && !defined(__OSX__)
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#endif

#include <cmath>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {
template <typename DeviceContext, typename T, int Rank>
struct Transpose {
  void operator()(const DeviceContext& context, const framework::Tensor& in,
                  framework::Tensor* out, const std::vector<int>& axis);
};

template <typename DeviceContext, typename T>
struct SetConstant {
  void operator()(const DeviceContext& context, framework::Tensor* tensor,
                  T num);
};

template <typename Place>
void set_constant_with_place(const platform::DeviceContext& context,
                             framework::Tensor* tensor, float value);

void set_constant(const platform::DeviceContext& context,
                  framework::Tensor* tensor, float value);

template <typename DeviceContext, typename T>
struct RowwiseAdd {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& vec, framework::Tensor* output);
};

template <typename DeviceContext, typename T>
struct ColwiseSum {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseSum {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseMean {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

#if !defined(__APPLE__) && !defined(__OSX__)
static const unsigned int SSE_STEP_SIZE = 4;
static const unsigned int SSE_CUT_LEN_MASK = 3U;
#define __m256x __m256
#define __m128x __m128
#define _mm_load_px _mm_loadu_ps
#define _mm_load1_px _mm_load1_ps
#define _mm_store_px _mm_storeu_ps
#define _mm_add_px _mm_add_ps
#define _mm_mul_px _mm_mul_ps

template <typename T>
inline void paddle_sse_axpy(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_alpha = _mm_load1_px(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(y + jjj,
                 _mm_add_px(_mm_load_px(y + jjj),
                            _mm_mul_px(mm_alpha, _mm_load_px(x + jjj))));
  }
  for (; jjj < len; jjj++) {
    y[jjj] += alpha * x[jjj];
  }
}

template <typename T>
inline T paddle_uniform_real(T min, T max) {
  return ((T)rand() / (RAND_MAX)) * (max - min) + min;
}
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
