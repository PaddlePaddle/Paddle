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

#define SWITCH_TYPES(case_macro)                   \
  switch (type) {                                  \
    case CUDA_R_16BF:                              \
      case_macro(nv_bfloat16);                     \
    case CUDA_R_32F:                               \
      case_macro(float);                           \
    default:                                       \
      EP_HOST_ASSERT(false && "Unsupported type"); \
  }                                                \
  while (false)

#define DISPATCH_NUM_PER_CHANNEL(num_per_channel, kNumPerChannels, ...) \
  if (num_per_channel == -1) {                                          \
    constexpr int kNumPerChannels = -1;                                 \
    __VA_ARGS__                                                         \
  } else if (num_per_channel == 128) {                                  \
    constexpr int kNumPerChannels = 128;                                \
    __VA_ARGS__                                                         \
  } else {                                                              \
    EP_HOST_ASSERT(false && "Unsupported num_per_channel");             \
  }

#define DISPATCH_NUM_TOPK(num_topk, kTopk, ...)      \
  if (num_topk == 8) {                               \
    constexpr int kTopk = 8;                         \
    __VA_ARGS__                                      \
  } else {                                           \
    EP_HOST_ASSERT(false && "Unsupported num_topk"); \
  }
