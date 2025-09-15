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
#include <limits>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void RandomKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int64_t from,
                  int64_t to,
                  DenseTensor* out);

template <typename scalar_t>
int64_t update_from(int64_t from) {
  static_assert(std::is_floating_point<scalar_t>::value ||
                    std::is_same<scalar_t, paddle::float16>::value ||
                    std::is_same<scalar_t, paddle::bfloat16>::value,
                "scalar_t must be floating-point type");

  const auto from_plus_1 =
      static_cast<int64_t>(static_cast<scalar_t>(from + 1));
  if (from_plus_1 < from) {
    int64_t from_ = std::abs(from + 1);
    int n = 0;
    while (from_ >>= 1) ++n;
    from =
        from_plus_1 + (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return from;
}

template <typename scalar_t>
int64_t update_to(int64_t to) {
  static_assert(std::is_floating_point<scalar_t>::value ||
                    std::is_same<scalar_t, paddle::float16>::value ||
                    std::is_same<scalar_t, paddle::bfloat16>::value,
                "scalar_t must be floating-point type");

  const auto to_minus_1 = static_cast<int64_t>(static_cast<scalar_t>(to - 1));
  if (to_minus_1 >= to) {
    int64_t to_ = std::abs(to - 1);
    int n = 0;
    while (to_ >>= 1) ++n;
    to = to_minus_1 - (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return to;
}

}  // namespace phi
