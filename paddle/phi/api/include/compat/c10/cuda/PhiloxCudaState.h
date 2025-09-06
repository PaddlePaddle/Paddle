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

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"

namespace at {

struct PhiloxCudaState {
  PhiloxCudaState() = default;
  // Called if graph capture is not underway
  PhiloxCudaState(uint64_t seed, uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxCudaState(int64_t* seed,
                  int64_t* offset_extragraph,
                  uint32_t offset_intragraph) {
    seed_.ptr = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  Payload seed_{};
  Payload offset_{};
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

inline PhiloxCudaState _PD_Internal_GetDefaultPhiloxCudaState(int64_t inc) {
  auto dev_ctx = phi::DeviceContextPool::Instance().Get(phi::GPUPlace());
  auto cuda_ctx = static_cast<const phi::GPUContext*>(dev_ctx);
  // auto gen = phi::GetRandomSeedGenerator("");
  auto* gen = cuda_ctx->GetGenerator();
  auto seed_offset_pair = gen->IncrementOffset(inc);
  return PhiloxCudaState(seed_offset_pair.first, seed_offset_pair.second);
}

}  // namespace at
