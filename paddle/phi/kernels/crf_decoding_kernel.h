// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <limits>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/jit/kernels.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void Decode(const Context& dev_ctx,
            const phi::DenseTensor& emission_weights,
            const phi::DenseTensor& transition_weights,
            phi::DenseTensor* decoded_path) {
  auto emission_dims = emission_weights.dims();
  const size_t seq_len = emission_dims[0];
  const size_t tag_num = emission_dims[1];
  const T* x = emission_weights.data<T>();
  const T* w = transition_weights.data<T>();
  int64_t* path = decoded_path->data<int64_t>();

  // alpha is a memo table. An element alpha(k, v) records the score of the
  // best sequence of tags from position 1 to position k with v being the end
  // tag.
  phi::DenseTensor alpha;
  alpha.Resize(emission_dims);
  T* alpha_value = dev_ctx.template Alloc<T>(&alpha);
  phi::DenseTensor track;
  track.Resize(emission_dims);
  int* track_value = dev_ctx.template Alloc<int>(&track);
  auto ker = phi::jit::KernelFuncs<phi::jit::CRFDecodingTuple<T>,
                                   phi::CPUPlace>::Cache()
                 .At(tag_num);
  ker(static_cast<int>(seq_len), x, w, alpha_value, track_value, tag_num);
  T max_score = -std::numeric_limits<T>::max();
  int max_i = 0;
  for (size_t i = 0; i < tag_num; ++i) {
    T score = alpha_value[(seq_len - 1) * tag_num + i] + w[tag_num + i];
    if (score > max_score) {
      max_score = score;
      max_i = i;
    }
  }
  path[seq_len - 1] = max_i;
  for (int k = seq_len - 1; k >= 1; --k) {
    path[k - 1] = max_i = track_value[k * tag_num + max_i];
  }
}

}  // namespace phi
