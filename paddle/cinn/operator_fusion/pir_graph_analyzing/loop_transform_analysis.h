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

#include "paddle/cinn/operator_fusion/pir_graph_analyzing/loop_axis_mapping.h"

namespace cinn::fusion {

LoopAxisMapping LoopAxisMappingMerge(const LoopAxisMapping& upstream,
                                     const LoopAxisMapping& downstream,
                                     bool upstream_is_anchor);
LoopAxisMapping TrivialSinkLoopAxisMappingMerge(
    const LoopAxisMapping& upstream, const LoopAxisMapping& downstream);
LoopAxisMapping ReducePlusTrivialLoopAxisMappingMerge(
    const LoopAxisMapping& upstream, const LoopAxisMapping& downstream);

// Try to find a valid axis transform route with specific direction between
// upstream and downstream LoopAxisMapping. The following cases are considered
// invalid:
// 1. There exists unsupported axis transform in the route.
// 2. There exists invalid delete axis transform in the route.
// 3. There exists relationship between reduce axis and non reduce axis.
// 4. Cannot transform from reduce pattern loop to trivial pattern loop.
// 5. Cannot transform between reduce patterns with different reduce axis.
std::optional<AxisTransformRoute> GetValidLoopTransformRoute(
    const LoopAxisMapping& upstream,
    const LoopAxisMapping& downstream,
    bool upstream_is_anchor);

}  // namespace cinn::fusion
