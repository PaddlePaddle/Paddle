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
    const DenseTensor& input, const DenseTensor& weight) {
  const auto x_dims = input.dims();
  const auto y_dims = weight.dims();
  const int64_t N = y_dims.size() < 2 ? 1 : y_dims[y_dims.size() - 1];
  const int64_t K = y_dims.size() < 2 ? y_dims[0] : y_dims[y_dims.size() - 2];

  int64_t M = x_dims.size() >= 2 ? x_dims[x_dims.size() - 2] : 1;
  if (x_dims.size() > 2) {
    // Accumulate the batch dims for input
    for (int64_t i = 0; i < x_dims.size() - 2; ++i) {
      M *= x_dims[i];
    }
  }

  return {M, N, K};
}

template <typename T, typename Context>
void LinearV2Kernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& weight,
                    const DenseTensor& bias,
                    DenseTensor* out);
}  // namespace phi
