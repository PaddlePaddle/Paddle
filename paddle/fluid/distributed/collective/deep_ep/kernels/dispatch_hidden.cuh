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

#define SWITCH_HIDDEN(case_macro)                    \
  switch (hidden) {                                  \
    case 2048:                                       \
      case_macro(2048);                              \
    case 2560:                                       \
      case_macro(2560);                              \
    case 4096:                                       \
      case_macro(4096);                              \
    case 5120:                                       \
      case_macro(5120);                              \
    case 6144:                                       \
      case_macro(6144);                              \
    case 7168:                                       \
      case_macro(7168);                              \
    case 8192:                                       \
      case_macro(8192);                              \
    case 6144:                                       \
      case_macro(6144);                              \
    default:                                         \
      EP_HOST_ASSERT(false && "Unsupported hidden"); \
  }                                                  \
  while (false)

#define DISPATCH_HIDDEN_SIZE(hidden, kHidden, ...) \
  if (hidden == 1536) {                            \
    constexpr size_t kHidden = 1536;               \
    __VA_ARGS__                                    \
  } else if (hidden == 4096) {                     \
    constexpr size_t kHidden = 4096;               \
    __VA_ARGS__                                    \
  } else if (hidden == 5120) {                     \
    constexpr size_t kHidden = 5120;               \
    __VA_ARGS__                                    \
  } else if (hidden == 6144) {                     \
    constexpr size_t kHidden = 6144;               \
    __VA_ARGS__                                    \
  } else if (hidden == 7168) {                     \
    constexpr size_t kHidden = 7168;               \
    __VA_ARGS__                                    \
  } else if (hidden == 8192) {                     \
    constexpr size_t kHidden = 8192;               \
    __VA_ARGS__                                    \
  } else {                                         \
    EP_HOST_ASSERT(false && "Unsupported hidden"); \
  }
