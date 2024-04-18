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

#include "paddle/common/dfs_topo_walker.h"

namespace cinn {
namespace common {

// DFS Topological order walker.
// Try to walk in a depth first manner while ensuring topological order.
// For example:
//   Graph:
//     0 -> 1
//     2 -> 3
//     0 -> 3
//     1 -> 3
//     3 -> 4
//   Start nodes: 0, 2
//   Walking order: 0 -> 1 -> 2 -> 3 -> 4
template <typename NodeType,
          typename NodeHash = std::hash<NodeType>,
          typename NodeEqual = std::equal_to<NodeType>>
using DfsTopoWalker = ::common::DfsTopoWalker<NodeType, NodeHash, NodeEqual>;

}  // namespace common
}  // namespace cinn
