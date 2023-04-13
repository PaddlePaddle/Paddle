// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/distribution_helper.h"

namespace phi {
namespace fusion {

template <typename Context>
static inline std::vector<size_t> GetRandomCudaProp(int64_t numel,
                                                    const Context& dev_ctx) {
  constexpr int kVecSize = funcs::uniform_distribution<float>::kReturnsCount;
  auto gpu_config =
      backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, kVecSize);
  size_t grid_size = gpu_config.GetGridSize();
  size_t block_size = gpu_config.GetBlockSize();
  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  const auto& prop = phi::backends::gpu::GetDeviceProperties(device_id);
  size_t max_grid_size =
      prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / block_size;
  grid_size = std::min(grid_size, max_grid_size);
  auto offset =
      ((numel - 1) / (grid_size * block_size * kVecSize) + 1) * kVecSize;
  size_t main_offset =
      numel / (block_size * kVecSize) * (block_size * kVecSize);
  return {grid_size, block_size, offset, main_offset};
}

}  // namespace fusion
}  // namespace phi
