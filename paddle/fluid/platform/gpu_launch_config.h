/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>

#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace platform {

struct GpuLaunchConfig {
  // Number of threads per block.
  int threads;
  // Number of blocks for GPU kernel launch.
  int blocks;

  GpuLaunchConfig(int threads, int blocks) : threads(threads), blocks(blocks) {}
};

inline GpuLaunchConfig getGpuLaunchConfig(
    const int N, const framework::ExecutionContext& ctx) {
  int threads =
      std::min(1024, ctx.cuda_device_context().GetMaxThreadsPerBlock());
  int physical_thread_count =
      std::min(ctx.cuda_device_context().GetMaxPhysicalThreadCount(), N);
  int blocks = std::min((physical_thread_count + threads - 1) / threads,
                        ctx.cuda_device_context().GetSMCount());

  GpuLaunchConfig config(threads, blocks);

  return config;
}

}  // namespace platform
}  // namespace paddle
