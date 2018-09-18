/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/cpu_info.h"
#ifdef __AVX__
#include <immintrin.h>
#endif

namespace paddle {
namespace operators {
namespace math {

// TODO(TJ): ugly workaround, clean me
template <typename T>
void lstm_compute_ctht(T* gates, const T* ct_1, T* ct, T* ht) {
  // gates: W_ch, W_ih, W_fh, W_oh
  vec_sigmoid<T, platform::jit::avx>(24, gates + 8, gates + 8);
  vec_tanh<T, platform::jit::avx>(8, gates, gates);
  const T *i = gates + 8, *f = gates + 16, *o = gates + 24;
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int d = 0; d < 8; ++d) {
    // C_t = C_t-1 * fgated + cand_gated * igated
    ct[d] = ct_1[d] * f[d] + gates[d] * i[d];
    // H_t = act_cell(C_t) * ogated
    T tmp = ct[d] * 2;
    tmp = static_cast<T>(0) - ((tmp < min) ? min : ((tmp > max) ? max : tmp));
    vec_exp<T>(1, &tmp, &tmp);
    tmp = static_cast<T>(2) / (static_cast<T>(1) + tmp) - static_cast<T>(1);
    ht[d] = tmp * o[d];
  }
}

#ifdef __AVX__
namespace detail {
namespace forward {
namespace avx {
__m256 Sigmoid(const __m256 a);
__m256 Tanh(const __m256 a);
}  // namespace avx
}  // namespace forward
}  // namespace detail

template <>
void lstm_compute_ctht<float>(float* gates, const float* ct_1, float* ct,
                              float* ht) {
  namespace act = detail::forward::avx;
  // gates: W_ch, W_ih, W_fh, W_oh
  __m256 c, i, f, o;
  c = _mm256_loadu_ps(gates);
  i = _mm256_loadu_ps(gates + 8);
  f = _mm256_loadu_ps(gates + 16);
  o = _mm256_loadu_ps(gates + 24);

  /* C_t = C_t-1 * fgated + cand_gated * igated*/
  c = _mm256_mul_ps(act::Tanh(c), act::Sigmoid(i));
  i = _mm256_loadu_ps(ct_1);
  f = _mm256_mul_ps(i, act::Sigmoid(f));
  f = _mm256_add_ps(c, f);
  _mm256_storeu_ps(ct, f);

  /* H_t = act_cell(C_t) * ogated */
  o = _mm256_mul_ps(act::Tanh(f), act::Sigmoid(o));
  _mm256_storeu_ps(ht, o);
}
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
