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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/primitive/datamover_primitives.h"

#define SWITCH_NOPE_HEAD_DIM(__dim, ...) \
  if (__dim == 32) {                     \
    constexpr int NopeSize = 32;         \
    { __VA_ARGS__ }                      \
  } else if (__dim == 64) {              \
    constexpr int NopeSize = 64;         \
    { __VA_ARGS__ }                      \
  } else if (__dim == 96) {              \
    constexpr int NopeSize = 96;         \
    { __VA_ARGS__ }                      \
  } else if (__dim == 128) {             \
    constexpr int NopeSize = 128;        \
    { __VA_ARGS__ }                      \
  } else {                               \
    constexpr int NopeSize = 0;          \
    { __VA_ARGS__ }                      \
  }

#define SWITCH_PE_HEAD_DIM(__dim, ...) \
  if (__dim == 32) {                   \
    constexpr int PeSize = 32;         \
    { __VA_ARGS__ }                    \
  } else if (__dim == 64) {            \
    constexpr int PeSize = 64;         \
    { __VA_ARGS__ }                    \
  } else if (__dim == 96) {            \
    constexpr int PeSize = 96;         \
    { __VA_ARGS__ }                    \
  } else if (__dim == 128) {           \
    constexpr int PeSize = 128;        \
    { __VA_ARGS__ }                    \
  } else {                             \
    constexpr int PeSize = 0;          \
    { __VA_ARGS__ }                    \
  }

// Note: pe_head_dim must be divisible by 2x of the vector size.
#define SWITCH_VEC_SIZE(__nope_head_dim, __pe_head_dim, ...)      \
  if (__nope_head_dim % 4 == 0 && __nope_head_dim >= 128 &&       \
      __pe_head_dim % 8 == 0 && __pe_head_dim >= 128) {           \
    constexpr int VecSize = 4;                                    \
    { __VA_ARGS__ }                                               \
  } else if (__nope_head_dim % 2 == 0 && __nope_head_dim >= 64 && \
             __pe_head_dim % 4 == 0 && __pe_head_dim >= 64) {     \
    constexpr int VecSize = 2;                                    \
    { __VA_ARGS__ }                                               \
  } else {                                                        \
    constexpr int VecSize = 1;                                    \
    { __VA_ARGS__ }                                               \
  }

#define SWITCH_ROPE_KERNEL(__nope_head_dim, __pe_head_dim, ...) \
  SWITCH_NOPE_HEAD_DIM(                                         \
      __nope_head_dim,                                          \
      SWITCH_PE_HEAD_DIM(                                       \
          __pe_head_dim,                                        \
          SWITCH_VEC_SIZE(__nope_head_dim, __pe_head_dim, {__VA_ARGS__})))

#define LOOP_WITH_SIZE_HINT(__index, __init, __size, __stride, __hint) \
  for (uint32_t __index = (__init), __offset = 0;                      \
       (__hint) > 0 ? __offset < (__hint) : __index < (__size);        \
       __index += (__stride), __offset += (__stride))                  \
    if ((__hint) == 0 || (__hint) % (__stride) == 0 ||                 \
        __offset + (__stride) < (__hint) || __index < (__size))
