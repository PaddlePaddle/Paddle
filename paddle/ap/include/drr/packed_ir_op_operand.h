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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/graph/node_topo_cstr.h"

namespace ap::drr {

template <typename NodeT>
struct PackedIrOpOperandImpl {
  graph::Node<NodeT> node;
  std::size_t local_uid;  // not a index

  bool operator==(const PackedIrOpOperandImpl& other) const {
    return this->node == other.node && this->local_uid == other.local_uid;
  }

  graph::PackedIrOpOperandTopoCstr node_topo_cstr() const {
    return graph::PackedIrOpOperandTopoCstr{};
  }
};

template <typename NodeT>
ADT_DEFINE_RC(PackedIrOpOperand, PackedIrOpOperandImpl<NodeT>);

}  // namespace ap::drr
