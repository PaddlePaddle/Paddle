// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "paddle/common/flags.h"

COMMON_DECLARE_bool(deterministic_rng);
COMMON_DECLARE_int32(deterministic_rng_grid);

namespace phi {
namespace funcs {

inline bool IsDeterministicRNG() { return FLAGS_deterministic_rng; }

struct RNGLaunchConfig {
  size_t grid_size;
  size_t block_size;
  uint64_t increment;
};

// Cross-device consistency requires the same FLAGS_deterministic_rng_grid.
// vec_size: elements per thread per loop iteration (kReturnsCount).
inline RNGLaunchConfig GetDeterministicRNGConfig(int64_t numel,
                                                 int vec_size = 4) {
  RNGLaunchConfig config;
  constexpr size_t kBlockSize = 256;
  size_t grid_cap = static_cast<size_t>(FLAGS_deterministic_rng_grid);
  size_t needed = (static_cast<size_t>(numel) + kBlockSize - 1) / kBlockSize;
  config.grid_size = std::min(needed, grid_cap);
  config.block_size = kBlockSize;

  size_t total_thread = config.grid_size * config.block_size;
  size_t loop_times =
      (static_cast<size_t>(numel) + vec_size * total_thread - 1) /
      (vec_size * total_thread);
  config.increment = static_cast<uint64_t>(loop_times * vec_size);

  return config;
}

}  // namespace funcs
}  // namespace phi
