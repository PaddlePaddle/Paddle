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

// The file has been adapted from DeepSeek DeepEP project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

#pragma once

#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"

#define DISPATCH_NUM_EXPERTS(num_experts, kNumExperts, ...) \
  if (num_experts == 64) {                                  \
    constexpr int kNumExperts = 64;                         \
    __VA_ARGS__                                             \
  } else if (num_experts == 72) {                           \
    constexpr int kNumExperts = 72;                         \
    __VA_ARGS__                                             \
  } else if (num_experts == 128) {                          \
    constexpr int kNumExperts = 128;                        \
    __VA_ARGS__                                             \
  } else if (num_experts == 192) {                          \
    constexpr int kNumExperts = 192;                        \
    __VA_ARGS__                                             \
  } else if (num_experts == 256) {                          \
    constexpr int kNumExperts = 256;                        \
    __VA_ARGS__                                             \
  } else if (num_experts == 384) {                          \
    constexpr int kNumExperts = 384;                        \
    __VA_ARGS__                                             \
  } else {                                                  \
    EP_HOST_ASSERT(false && "Unsupported num_experts");     \
  }

#define DISPATCH_NUM_WARP_GROUPS(num_warp_groups, kNumWarpGroups, ...) \
  if (num_warp_groups == 1) {                                          \
    constexpr int kNumWarpGroups = 1;                                  \
    __VA_ARGS__                                                        \
  } else if (num_warp_groups == 2) {                                   \
    constexpr int kNumWarpGroups = 2;                                  \
    __VA_ARGS__                                                        \
  } else if (num_warp_groups == 3) {                                   \
    constexpr int kNumWarpGroups = 3;                                  \
    __VA_ARGS__                                                        \
  } else if (num_warp_groups == 4) {                                   \
    constexpr int kNumWarpGroups = 4;                                  \
    __VA_ARGS__                                                        \
  } else if (num_warp_groups == 8) {                                   \
    constexpr int kNumWarpGroups = 8;                                  \
    __VA_ARGS__                                                        \
  } else {                                                             \
    EP_HOST_ASSERT(false && "Unsupported num_warp_groups");            \
  }
