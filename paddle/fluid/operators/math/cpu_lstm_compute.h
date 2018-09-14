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

namespace paddle {
namespace operators {
namespace math {

// TODO(TJ): ugly workaround, clean me
template <typename T>
void lstm_compute_ctht(const T* gates, const T* ct_1, T* ct, T* ht) {
  // gates: W_ch, W_ih, W_fh, W_oh
  vec_sigmoid<T, platform::jit::avx>(24, gates + 8, gates + 8);
  vec_tanh<T, platform::jit::avx>(8, gates, gates);
  const T *i = gates + 8, *f = gates + 16, *o = gates + 24;
  for (int d = 0; d < 8; ++d) {
    // C_t = C_t-1 * fgated + cand_gated * igated
    ct[d] = ct_1[d] * f[d] + gates[d] * i[d];

    // H_t = act_cell(C_t) * ogated
    T tmp = ct[d] * 2;
    tmp = static_cast<T>(0) - (tmp < static_cast<T>(SIGMOID_THRESHOLD_MIN))
              ? min
              : ((tmp > static_cast<T>(SIGMOID_THRESHOLD_MAX))
                     ? static_cast<T>(SIGMOID_THRESHOLD_MAX)
                     : tmp);
    vec_exp<T>(1, &tmp, &tmp);
    tmp = static_cast<T>(2) / (static_cast<T>(1) + tmp) - static_cast<T>(1);
    ht[d] = tmp * o[d];
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
