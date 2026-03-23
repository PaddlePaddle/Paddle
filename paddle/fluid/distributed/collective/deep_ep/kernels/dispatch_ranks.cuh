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

#define SWITCH_RANKS(case_macro)                     \
  switch (num_ranks) {                               \
    case 2:                                          \
      case_macro(2);                                 \
    case 4:                                          \
      case_macro(4);                                 \
    case 8:                                          \
      case_macro(8);                                 \
    default:                                         \
      EP_HOST_ASSERT(false and "Unsupported ranks"); \
  }                                                  \
  while (false)

#define SWITCH_RDMA_RANKS(case_macro)                     \
  switch (num_ranks / NUM_MAX_NVL_PEERS) {                \
    case 2:                                               \
      case_macro(2);                                      \
    case 3:                                               \
      case_macro(3);                                      \
    case 4:                                               \
      case_macro(4);                                      \
    case 8:                                               \
      case_macro(8);                                      \
    case 16:                                              \
      case_macro(16);                                     \
    case 18:                                              \
      case_macro(18);                                     \
    case 20:                                              \
      case_macro(20);                                     \
    default:                                              \
      EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
  }                                                       \
  while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)  \
  switch (num_ranks) {                              \
    case 2:                                         \
      case_macro(dtype, 2);                         \
    case 4:                                         \
      case_macro(dtype, 4);                         \
    case 8:                                         \
      case_macro(dtype, 8);                         \
    default:                                        \
      EP_HOST_ASSERT(false && "Unsupported ranks"); \
  }                                                 \
  while (false)

#define DISPATCH_RDMA_RANKS(num_rdma_ranks, kNumRdmaRanks, ...) \
  if (num_rdma_ranks == 1) {                                    \
    constexpr int kNumRdmaRanks = 1;                            \
    __VA_ARGS__                                                 \
  } else if (num_rdma_ranks == 2) {                             \
    constexpr int kNumRdmaRanks = 2;                            \
    __VA_ARGS__                                                 \
  } else if (num_rdma_ranks == 3) {                             \
    constexpr int kNumRdmaRanks = 3;                            \
    __VA_ARGS__                                                 \
  } else if (num_rdma_ranks == 4) {                             \
    constexpr int kNumRdmaRanks = 4;                            \
    __VA_ARGS__                                                 \
  } else if (num_rdma_ranks == 8) {                             \
    constexpr int kNumRdmaRanks = 8;                            \
    __VA_ARGS__                                                 \
  } else if (num_rdma_ranks == 16) {                            \
    constexpr int kNumRdmaRanks = 16;                           \
    __VA_ARGS__                                                 \
  } else {                                                      \
    EP_HOST_ASSERT(false && "Unsupported num_rdma_ranks");      \
  }
