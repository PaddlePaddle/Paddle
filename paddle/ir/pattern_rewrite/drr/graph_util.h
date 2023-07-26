// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include <functional>
#include <utility>

#include "cinn/hlir/drr/graph_struct.h"

namespace cinn {
namespace hlir {
namespace drr {

template <typename NodeType>
std::pair<NodeType, int> FindCenterAndRadius(NodeType start,
                                             const GraphStruct& graph_struct) {
  TODO(thisjiang);
}

template <typename NodeType>
std::vector<std::vector<NodeType>> FindMSTFromCenterToSinks(
    NodeType center, const GraphStruct& graph_struct) {
  TODO(thisjiang);
}

}  // namespace drr
}  // namespace hlir
}  // namespace cinn
