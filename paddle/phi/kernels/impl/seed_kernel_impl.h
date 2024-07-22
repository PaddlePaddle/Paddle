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

#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

static int get_seed(int user_seed,
                    bool deterministic,
                    const std::string& rng_name) {
  int seed = 0;
  if (!deterministic) {
    // NOTE: fixed seed should only be used in unittest or for debug.
    // Guarantee to use random seed in training.
    if (user_seed != 0) {
      seed = user_seed;
    } else {
      std::random_device rnd;
      seed = rnd();
    }
  } else {
    std::string name = rng_name;
    auto rng = phi::GetRandomSeedGenerator(name);
    do {  // NOTE(wangxi): cpu dropout will use random seed if seed == 0
      seed = static_cast<int>(rng->Random64());
    } while (seed == 0);
  }
  return seed;
}

template <typename T, typename Context>
void CPUSeedKernel(const Context& dev_ctx,
                   int seed,
                   bool deterministic,
                   const std::string& rng_name,
                   bool force_cpu UNUSED,
                   DenseTensor* out) {
  auto* out_data = dev_ctx.template Alloc<T>(out);
  out_data[0] = get_seed(seed, deterministic, rng_name);
}
}  // namespace phi
