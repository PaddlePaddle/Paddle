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

// The file has been adapted from DeepSeek DeepGEMM project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepGEMM/blob/main/LICENSE

#pragma once

#include "utils.cuh"

namespace deep_gemm {

// TODO: move this function to other files
__device__ __forceinline__ void tma_copy(void const* desc_ptr,
                                         uint64_t* barrier_ptr,
                                         void* smem_ptr,
                                         int32_t const& crd_0,
                                         int32_t const& crd_1,
                                         uint32_t num_tma_multicast) {
  constexpr auto cache_hint =
      static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  if (num_tma_multicast == 1) {
    cute::SM90_TMA_LOAD_2D::copy(
        desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd_0, crd_1);
  } else if (cute::block_rank_in_cluster() == 0) {
    cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr,
                                           barrier_ptr,
                                           (1 << num_tma_multicast) - 1,
                                           cache_hint,
                                           smem_ptr,
                                           crd_0,
                                           crd_1);
  }
}

}  // namespace deep_gemm
