// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>
#include <unordered_map>
#include "paddle/ap/include/graph/adt.h"
#include "paddle/ap/include/graph/node.h"

namespace ap::graph {

template <typename NodeT>
struct NodeDescriptor;

template <typename NodeT>
struct NodeDescriptorInterface {
  std::string DebugId(const NodeT&);

  template <typename DrrGraphNodeT>
  adt::Result<bool> AttrsSatisfyIfBothAreOpsOrValues(
      const NodeT& node, const DrrGraphNodeT& drr_node);
};

}  // namespace ap::graph
