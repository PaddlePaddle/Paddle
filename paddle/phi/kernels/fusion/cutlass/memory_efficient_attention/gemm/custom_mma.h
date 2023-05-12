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

#include "./custom_mma_multistage.h"
#include "./custom_mma_pipelined.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"

template <typename Mma, int kMaxK>
struct MakeCustomMma;

template <typename Shape,
          typename IteratorA,
          typename SmemIteratorA,
          cutlass::arch::CacheOperation::Kind CacheOpA,
          typename IteratorB,
          typename SmemIteratorB,
          cutlass::arch::CacheOperation::Kind CacheOpB,
          typename ElementC,
          typename LayoutC,
          typename Policy,
          int Stages,
          cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
          int kMaxK>
struct MakeCustomMma<
    cutlass::gemm::threadblock::MmaMultistage<Shape,
                                              IteratorA,
                                              SmemIteratorA,
                                              CacheOpA,
                                              IteratorB,
                                              SmemIteratorB,
                                              CacheOpB,
                                              ElementC,
                                              LayoutC,
                                              Policy,
                                              Stages,
                                              SharedMemoryClear>,
    kMaxK> {
  // Reduce the number of stages if we don't need that many
  static int constexpr kStages =
      kMaxK == cutlass::platform::numeric_limits<int>::max()
          ? Stages
          : cutlass::const_min(Stages,
                               (kMaxK + static_cast<int>(Shape::kK) - 1) /
                                   static_cast<int>(Shape::kK));
  using Mma = cutlass::gemm::threadblock::CustomMmaMultistage<Shape,
                                                              IteratorA,
                                                              SmemIteratorA,
                                                              CacheOpA,
                                                              IteratorB,
                                                              SmemIteratorB,
                                                              CacheOpB,
                                                              ElementC,
                                                              LayoutC,
                                                              Policy,
                                                              kStages,
                                                              SharedMemoryClear,
                                                              kMaxK>;
};

template <typename Shape,
          typename IteratorA,
          typename SmemIteratorA,
          typename IteratorB,
          typename SmemIteratorB,
          typename ElementC,
          typename LayoutC,
          typename Policy,
          int kMaxK>
struct MakeCustomMma<cutlass::gemm::threadblock::MmaPipelined<Shape,
                                                              IteratorA,
                                                              SmemIteratorA,
                                                              IteratorB,
                                                              SmemIteratorB,
                                                              ElementC,
                                                              LayoutC,
                                                              Policy>,
                     kMaxK> {
  using Mma = cutlass::gemm::threadblock::CustomMmaPipelined<Shape,
                                                             IteratorA,
                                                             SmemIteratorA,
                                                             IteratorB,
                                                             SmemIteratorB,
                                                             ElementC,
                                                             LayoutC,
                                                             Policy>;
};
