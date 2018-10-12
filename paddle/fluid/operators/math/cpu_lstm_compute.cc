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

#include "paddle/fluid/operators/math/cpu_lstm_compute.h"

namespace paddle {
namespace operators {
namespace math {
#ifdef __AVX__
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
