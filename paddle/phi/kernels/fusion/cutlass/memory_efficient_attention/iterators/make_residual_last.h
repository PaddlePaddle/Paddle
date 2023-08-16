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

#include "./predicated_tile_access_iterator_residual_last.h"
#include "./predicated_tile_iterator_residual_last.h"

namespace cutlass {
namespace transform {
namespace threadblock {

template <typename BaseIterator>
struct MakeIteratorResidualLast;

template <typename Shape,
          typename Element,
          typename Layout,
          int AdvanceRank,
          typename ThreadMap,
          int AccessSize,
          bool Gather>
struct MakeIteratorResidualLast<PredicatedTileIterator<Shape,
                                                       Element,
                                                       Layout,
                                                       AdvanceRank,
                                                       ThreadMap,
                                                       AccessSize,
                                                       Gather>> {
  using Iterator = PredicatedTileIteratorResidualLast<Shape,
                                                      Element,
                                                      Layout,
                                                      AdvanceRank,
                                                      ThreadMap,
                                                      AccessSize,
                                                      Gather>;
};

template <typename Shape,
          typename Element,
          typename Layout,
          int AdvanceRank,
          typename ThreadMap,
          typename AccessType,
          bool Gather>
struct MakeIteratorResidualLast<PredicatedTileAccessIterator<Shape,
                                                             Element,
                                                             Layout,
                                                             AdvanceRank,
                                                             ThreadMap,
                                                             AccessType,
                                                             Gather>> {
  using Iterator = PredicatedTileAccessIteratorResidualLast<Shape,
                                                            Element,
                                                            Layout,
                                                            AdvanceRank,
                                                            ThreadMap,
                                                            AccessType,
                                                            Gather>;
};
}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass
