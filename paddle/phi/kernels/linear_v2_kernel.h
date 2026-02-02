// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

// we don't receive 2+d tensor as weight
inline std::tuple<int64_t, int64_t, int64_t> canonicalize_dims(
    const DenseTensor& input,
    const DenseTensor& weight,
    const bool transpose_weight) {
  const auto input_dims = input.dims();
  const auto weight_dims = weight.dims();
  // We assume weight to be [K, N] if not tranasposed, [N, K] if transposed, [K]
  // if 1D
  const int64_t N = weight_dims.size() < 2 ? 1 : weight_dims[!transpose_weight];
  const int64_t K =
      weight_dims.size() < 2 ? weight_dims[0] : weight_dims[transpose_weight];

  int64_t M = input_dims.size() >= 2 ? input_dims[input_dims.size() - 2] : 1;
  if (input_dims.size() > 2) {
    // Accumulate the batch dims for input
    for (int64_t i = 0; i < input_dims.size() - 2; ++i) {
      M *= input_dims[i];
    }
  }

  return {M, N, K};
}

template <typename T, typename Context>
void LinearV2Kernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& weight,
                    const DenseTensor& bias,
                    const bool transpose_weight,
                    DenseTensor* out);
}  // namespace phi
