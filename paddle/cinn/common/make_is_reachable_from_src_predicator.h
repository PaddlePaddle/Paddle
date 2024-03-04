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

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "paddle/cinn/common/topo_walker.h"

namespace cinn::common {

template <typename NodeT, typename IterT>
std::function<bool(NodeT)> MakeIsReachableFromSrcPredicator(
    const TopoWalker<NodeT>& walker, IterT src_begin, IterT src_end) {
  auto nodes = std::make_shared<std::unordered_set<NodeT>>();
  nodes->insert(src_begin, src_end);
  walker(src_begin, src_end, [&](NodeT node) { nodes->insert(node); });
  return [nodes](NodeT node) { return nodes->count(node) > 0; };
}

}  // namespace cinn::common
