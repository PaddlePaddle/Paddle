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

//  Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
//
//  This source code is licensed under the BSD license found in the
//  LICENSE file in the root directory of this source tree.

#pragma once
#include <float.h>
#include <stdio.h>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
// Debugging functions
////////////////////////////////////////////////////////////////////////////////
// Nans & inf detection
#define NANCHECK(frag)                                     \
  {                                                        \
    for (int _i = 0; _i < frag.size(); ++_i) {             \
      assert(std::isfinite(static_cast<float>(frag[_i]))); \
      assert(!std::isnan(static_cast<float>(frag[_i])));   \
    }                                                      \
  }

// Print on the first thread of the first block
#if 1
#define PRINT_WARP_ID 0
#define PRINT_LANE_ID 0
#define PRINT_B0_T0(msg, ...)                                         \
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&        \
      threadIdx.x == PRINT_LANE_ID && threadIdx.y == PRINT_WARP_ID && \
      threadIdx.z == 0) {                                             \
    printf(msg "\n", ##__VA_ARGS__);                                  \
  }
#define PRINT_T0(msg, ...)                                            \
  if (threadIdx.x == PRINT_LANE_ID && threadIdx.y == PRINT_WARP_ID && \
      threadIdx.z == 0) {                                             \
    printf(msg "\n", ##__VA_ARGS__);                                  \
  }
#define PRINT_TX_LX(msg, ...)                                                 \
  for (int bx = 0; bx < gridDim.x; ++bx) {                                    \
    for (int by = 0; by < gridDim.y; ++by) {                                  \
      for (int bz = 0; bz < gridDim.z; ++bz) {                                \
        for (int tx = 0; tx < blockDim.x; ++tx) {                             \
          for (int ty = 0; ty < blockDim.y; ++ty) {                           \
            for (int tz = 0; tz < blockDim.z; ++tz) {                         \
              __syncthreads();                                                \
              if (blockIdx.x == bx && blockIdx.y == by && blockIdx.z == bz && \
                  threadIdx.x == tx && threadIdx.y == ty &&                   \
                  threadIdx.z == tz) {                                        \
                printf("[%d,%d,%d][%d,%d,%d]" msg "\n",                       \
                       bx,                                                    \
                       by,                                                    \
                       bz,                                                    \
                       tx,                                                    \
                       ty,                                                    \
                       tz,                                                    \
                       ##__VA_ARGS__);                                        \
              }                                                               \
            }                                                                 \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }
#else
#define PRINT_B0_T0
#define PRINT_TX_LX
#endif

struct __string_view {
  char const* data;
  std::size_t size;
};
#if __cplusplus >= 201402L
template <class T>
constexpr __string_view __get_type_name() {
  char const* p = __PRETTY_FUNCTION__;
  while (*p++ != '=')
    ;  // NOLINT
  for (; *p == ' '; ++p)
    ;  // NOLINT
  char const* p2 = p;
  int count = 1;
  for (;; ++p2) {
    switch (*p2) {
      case '[':
        ++count;
        break;
      case ']':
        --count;
        if (!count) return {p, std::size_t(p2 - p)};
    }
  }
  return {};
}
#else
template <class T>
constexpr __string_view __get_type_name() {
  return {"unsupported", 11};
}
#endif

// Print a given array
#define PRINT_ACCUM8_T0_L0_START(name, accum, start)          \
  PRINT_T0_L0("%s[%d:%d] - {%f, %f, %f, %f, %f, %f, %f, %f}", \
              name,                                           \
              static_cast<int>(start),                        \
              static_cast<int>(start + 8),                    \
              static_cast<float>(accum[start + 0]),           \
              static_cast<float>(accum[start + 1]),           \
              static_cast<float>(accum[start + 2]),           \
              static_cast<float>(accum[start + 3]),           \
              static_cast<float>(accum[start + 4]),           \
              static_cast<float>(accum[start + 5]),           \
              static_cast<float>(accum[start + 6]),           \
              static_cast<float>(accum[start + 7]));
#define PRINT_ACCUM8_T0_L0(name, accum) PRINT_ACCUM8_T0_L0_START(name, accum, 0)
#define PRINT_FRAG_T0_L0(name, frag)                          \
  {                                                           \
    auto typeStr = __get_type_name<decltype(frag)>();         \
    PRINT_T0_L0("printing %s (%s)", name, typeStr.data);      \
    for (int _start = 0; _start < frag.size(); _start += 8) { \
      PRINT_ACCUM8_T0_L0_START("  ", frag, _start);           \
    }                                                         \
    /*__syncthreads();   NANCHECK(frag); */                   \
  }
#define PRINT_ARRAY_T0_L0_INCR(name, array, length, incr)                \
  {                                                                      \
    PRINT_T0_L0("printing %s (len=%d)", name, static_cast<int>(length)); \
    for (int _start = 0; _start < length; _start += incr) {              \
      PRINT_ACCUM8_T0_L0_START("  ", array, _start);                     \
    }                                                                    \
  }
#define PRINT_ARRAY_T0_L0(name, array, length) \
  PRINT_ARRAY_T0_L0_INCR(name, array, length, 8)

// Print a 4x4 matrix
#define PRINT_TENSOR4x4_T0_L0_START(name, ref, start_x, start_y)            \
  PRINT_T0_L0(                                                              \
      "%s[%d:%d, %d:%d]:\n    %f, %f, %f, %f\n    %f, %f, %f, %f\n    %f, " \
      "%f, %f, %f\n    %f, %f, %f, %f",                                     \
      name,                                                                 \
      static_cast<int>(start_x),                                            \
      static_cast<int>(start_x + 4),                                        \
      static_cast<int>(start_y),                                            \
      static_cast<int>(start_y + 4),                                        \
      static_cast<float>(ref.at({start_x + 0, start_y + 0})),               \
      static_cast<float>(ref.at({start_x + 0, start_y + 1})),               \
      static_cast<float>(ref.at({start_x + 0, start_y + 2})),               \
      static_cast<float>(ref.at({start_x + 0, start_y + 3})),               \
      static_cast<float>(ref.at({start_x + 1, start_y + 0})),               \
      static_cast<float>(ref.at({start_x + 1, start_y + 1})),               \
      static_cast<float>(ref.at({start_x + 1, start_y + 2})),               \
      static_cast<float>(ref.at({start_x + 1, start_y + 3})),               \
      static_cast<float>(ref.at({start_x + 2, start_y + 0})),               \
      static_cast<float>(ref.at({start_x + 2, start_y + 1})),               \
      static_cast<float>(ref.at({start_x + 2, start_y + 2})),               \
      static_cast<float>(ref.at({start_x + 2, start_y + 3})),               \
      static_cast<float>(ref.at({start_x + 3, start_y + 0})),               \
      static_cast<float>(ref.at({start_x + 3, start_y + 1})),               \
      static_cast<float>(ref.at({start_x + 3, start_y + 2})),               \
      static_cast<float>(ref.at({start_x + 3, start_y + 3})));
#define PRINT_TENSOR4x4_T0_L0(name, ref) \
  PRINT_TENSOR4x4_T0_L0_START(name, ref, 0, 0)

#define PRINT_PROBLEM_SIZE(name, ps)                    \
  PRINT_T0_L0("%s.problem_size: {.m=%d, .n=%d, .k=%d}", \
              name,                                     \
              static_cast<int>(ps.m()),                 \
              static_cast<int>(ps.n()),                 \
              static_cast<int>(ps.k()))

template <typename LambdaIterator, typename LaneOffsetT, typename AccumT>
CUTLASS_DEVICE void print_warp_accum(AccumT accum,
                                     LaneOffsetT lane_offset,
                                     int32_t num_rows,
                                     int32_t num_cols) {
  bool is_main = blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
                 threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
  for (int row = 0; row < num_rows; ++row) {
    for (int col = 0; col < num_cols; ++col) {
      if (col % 32 == 0) {
        if (is_main) {
          printf("\nmat[%3d, %3d:%3d]", row, col, col + 32);
        }
        __syncthreads();
      }
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {},
          [&](int accum_m, int accum_n, int idx) {
            if (row == accum_m && col == accum_n &&
                (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)) {
              printf(" %6.1f", static_cast<float>(accum[idx]));
            }
          },
          [&](int accum_m) {});
      __syncthreads();
    }
    if (is_main) {
      printf("\n");
    }
  }
}
