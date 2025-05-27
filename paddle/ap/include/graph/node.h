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

#include <list>
#include <variant>
#include <vector>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/graph/node_list.h"
#include "paddle/ap/include/graph/tags.h"

namespace ap::graph {

template <typename T>
class NodeArena;

template <typename T>
struct Node {
  tNodeId<size_t> node_id_;
  std::weak_ptr<NodeArena<T>> node_arena_;

  const tNodeId<size_t>& node_id() const { return node_id_; }
  std::weak_ptr<NodeArena<T>> node_arena() const { return node_arena_; }

  adt::Result<std::shared_ptr<NodeArena<T>>> GetNodeArena() const {
    auto ptr = node_arena_.lock();
    if (!ptr) {
      return adt::errors::RuntimeError{"NodeArena is delete."};
    }
    return ptr;
  }

  adt::Result<T> Get() const;
  adt::Result<NodeList<T>> DownstreamNodes() const;
  adt::Result<NodeList<T>> UpstreamNodes() const;

  adt::Result<adt::Ok> ConnectTo(
      const Node& dst_node,
      const ValidListTag<std::monostate>& src_downstream_type,
      const ValidListTag<std::monostate>& dst_unstream_type) const;

  bool operator<(const Node& other) const {
    if (!(this->node_id_ == other.node_id_)) {
      return this->node_id_.value() < other.node_id_.value();
    }
    return this->node_arena_.lock() < other.node_arena_.lock();
  }

  bool operator==(const Node& other) const {
    return other.node_id_.value() == this->node_id_.value() &&
           other.node_arena_.lock() == this->node_arena_.lock();
  }

  bool operator!=(const Node& other) const { return !(*this == other); }

  std::size_t GetHashValue() const {
    return adt::hash_combine(
        this->node_id_.value(),
        std::hash<std::shared_ptr<NodeArena<T>>>()(this->node_arena_.lock()));
  }
};

}  // namespace ap::graph

namespace std {

template <typename T>
struct hash<ap::graph::Node<T>> {
  std::size_t operator()(const ap::graph::Node<T>& node) const {
    return node.GetHashValue();
  }
};

}  // namespace std
