// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "hytlass/gemm_coord.h"

namespace ap {

constexpr int kNumConfigsHalf = 28;
constexpr int kNumConfigsFloat = 13;

template <int SwizzleFactor, bool Batched>
struct SwizzleWrapper {
  using Type =
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>;
};

#define AP_AUTOTUNE(func, stream_ptr, count, ...)                             \
  {                                                                           \
    using FuncType = decltype(func<0>);                                       \
    static int selected_config_id = -1;                                       \
    static std::vector<std::function<FuncType>> matmul_functions =            \
        []<std::size_t... Is>(std::index_sequence<Is...>) {                   \
      return std::vector<std::function<FuncType>>{func<Is>...};               \
    }                                                                         \
    (std::make_index_sequence<count>());                                      \
    if (selected_config_id == -1) {                                           \
      selected_config_id =                                                    \
          ap::ProfileBestConfig(matmul_functions, stream_ptr, ##__VA_ARGS__); \
    }                                                                         \
    matmul_functions[selected_config_id](__VA_ARGS__);                        \
  }

#define AP_AUTOTUNE_half(func, stream_ptr, ...) \
  AP_AUTOTUNE(func, stream_ptr, ap::kNumConfigsHalf, __VA_ARGS__)
#define AP_AUTOTUNE_float(func, stream_ptr, ...) \
  AP_AUTOTUNE(func, stream_ptr, ap::kNumConfigsFloat, __VA_ARGS__)
#define AP_AUTOTUNE_bfloat16(func, stream_ptr, ...) \
  AP_AUTOTUNE_half(func, stream_ptr, __VA_ARGS__)

template <typename ElementT, int SwizzleFactor, bool Batched, int Id = 0>
struct GemmTuningConfigs {
  using TShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 2;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = Id;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 1> {
  using TShape = hytlass::gemm::GemmShape<64, 128, 64>;
  using WShape = hytlass::gemm::GemmShape<32, 64, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 1;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 2> {
  using TShape = hytlass::gemm::GemmShape<64, 128, 64>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 2;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 3> {
  using TShape = hytlass::gemm::GemmShape<128, 64, 64>;
  using WShape = hytlass::gemm::GemmShape<64, 32, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 3;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 4> {
  using TShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 4;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 5> {
  using TShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 5;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 6> {
  using TShape = hytlass::gemm::GemmShape<256, 64, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 6;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 7> {
  using TShape = hytlass::gemm::GemmShape<256, 64, 64>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 7;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 8> {
  using TShape = hytlass::gemm::GemmShape<256, 128, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 8;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 9> {
  using TShape = hytlass::gemm::GemmShape<256, 128, 64>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 9;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 10> {
  using TShape = hytlass::gemm::GemmShape<128, 32, 64>;
  using WShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 4;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 10;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 11> {
  using TShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 4;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 11;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 12> {
  using TShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 4;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 12;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 13> {
  using TShape = hytlass::gemm::GemmShape<256, 64, 64>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 4;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 13;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 14> {
  using TShape = hytlass::gemm::GemmShape<256, 64, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 4;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 14;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 15> {
  using TShape = hytlass::gemm::GemmShape<32, 64, 64>;
  using WShape = hytlass::gemm::GemmShape<16, 32, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 5;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 15;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 16> {
  using TShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 5;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 16;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 17> {
  using TShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 5;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 17;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 18> {
  using TShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 5;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 18;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 19> {
  using TShape = hytlass::gemm::GemmShape<64, 128, 32>;
  using WShape = hytlass::gemm::GemmShape<32, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 6;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 19;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 20> {
  using TShape = hytlass::gemm::GemmShape<128, 64, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 32, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 6;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 20;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 21> {
  using TShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WShape = hytlass::gemm::GemmShape<32, 32, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 10;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 21;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 22> {
  using TShape = hytlass::gemm::GemmShape<128, 256, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 2;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 22;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 23> {
  using TShape = hytlass::gemm::GemmShape<128, 256, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 23;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 24> {
  using TShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WShape = hytlass::gemm::GemmShape<32, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 24;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 25> {
  using TShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WShape = hytlass::gemm::GemmShape<32, 32, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 25;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 26> {
  using TShape = hytlass::gemm::GemmShape<64, 128, 64>;
  using WShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 4;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 26;
};

template <typename ElementT, int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<ElementT, SwizzleFactor, Batched, 27> {
  using TShape = hytlass::gemm::GemmShape<128, 64, 64>;
  using WShape = hytlass::gemm::GemmShape<64, 32, 64>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 16>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 27;
};

// Specialization for float
template <int SwizzleFactor, bool Batched, int Id>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, Id> {
  using TShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using WShape = hytlass::gemm::GemmShape<32, 32, 16>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = Id;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 1> {
  using TShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WShape = hytlass::gemm::GemmShape<32, 32, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 1;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 2> {
  using TShape = hytlass::gemm::GemmShape<64, 128, 32>;
  using WShape = hytlass::gemm::GemmShape<32, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 2;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 3> {
  using TShape = hytlass::gemm::GemmShape<64, 256, 16>;
  using WShape = hytlass::gemm::GemmShape<32, 64, 16>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 3;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 4> {
  using TShape = hytlass::gemm::GemmShape<64, 256, 32>;
  using WShape = hytlass::gemm::GemmShape<32, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 4;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 5> {
  using TShape = hytlass::gemm::GemmShape<128, 64, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 32, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 5;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 6> {
  using TShape = hytlass::gemm::GemmShape<128, 128, 16>;
  using WShape = hytlass::gemm::GemmShape<32, 64, 16>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 6;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 7> {
  using TShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WShape = hytlass::gemm::GemmShape<32, 64, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 7;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 8> {
  using TShape = hytlass::gemm::GemmShape<256, 64, 16>;
  using WShape = hytlass::gemm::GemmShape<64, 32, 16>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 8;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 9> {
  using TShape = hytlass::gemm::GemmShape<256, 64, 32>;
  using WShape = hytlass::gemm::GemmShape<64, 32, 32>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 3;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 9;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 10> {
  using TShape = hytlass::gemm::GemmShape<64, 128, 16>;
  using WShape = hytlass::gemm::GemmShape<32, 64, 16>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 4;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 10;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 11> {
  using TShape = hytlass::gemm::GemmShape<128, 64, 16>;
  using WShape = hytlass::gemm::GemmShape<64, 32, 16>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 4;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 11;
};

template <int SwizzleFactor, bool Batched>
struct GemmTuningConfigs<float, SwizzleFactor, Batched, 12> {
  using TShape = hytlass::gemm::GemmShape<128, 128, 16>;
  using WShape = hytlass::gemm::GemmShape<32, 64, 16>;
  using IShape = hytlass::gemm::GemmShape<16, 16, 8>;
  static constexpr int kNumStages = 4;

  using SwizzleThreadBlock =
      typename SwizzleWrapper<SwizzleFactor, Batched>::Type;
  static constexpr int kId = 12;
};

struct DefaultConfig {
  static constexpr int kConfigId = 0;
  static constexpr int kSwizzleFactor = 1;
  static constexpr bool kBatched = false;
};

}  // namespace ap
