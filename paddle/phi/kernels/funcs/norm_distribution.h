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

#include <random>
#include "paddle/phi/common/data_type.h"

namespace phi {

template <typename T>
inline void NormalDistribution(T* data,
                               const int64_t& size,
                               const float& mean,
                               const float& std,
                               std::shared_ptr<std::mt19937_64> engine) {
  std::normal_distribution<T> dist(mean, std);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <>
inline void NormalDistribution(phi::dtype::float16* data,
                               const int64_t& size,
                               const float& mean,
                               const float& std,
                               std::shared_ptr<std::mt19937_64> engine) {
  std::normal_distribution<float> dist(mean, std);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = static_cast<phi::dtype::float16>(dist(*engine));
  }
}

template <>
inline void NormalDistribution(phi::dtype::bfloat16* data,
                               const int64_t& size,
                               const float& mean,
                               const float& std,
                               std::shared_ptr<std::mt19937_64> engine) {
  std::normal_distribution<float> dist(mean, std);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = static_cast<phi::dtype::bfloat16>(dist(*engine));
  }
}

}  // namespace phi
