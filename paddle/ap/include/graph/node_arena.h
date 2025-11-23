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
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/graph/tags.h"

namespace ap::graph {

template <typename T>
class NodeArena : public std::enable_shared_from_this<NodeArena<T>> {
 public:
  NodeArena() {}
  NodeArena(const NodeArena&) = delete;
  NodeArena(NodeArena&&) = delete;

  adt::Result<T> At(const tNodeId<size_t>& node_id) const {
    if (node_id.value() >= nodes_.size()) {
      return adt::errors::IndexError{"node_id out of ranges."};
    }
    return nodes_.at(node_id.value());
  }

  template <typename ConstructorT>
  const T& New(const ConstructorT& Constructor) {
    tNodeId<size_t> node_id{nodes_.size()};
    const auto& node = Constructor(Node<T>{node_id, this->shared_from_this()});
    return EmplaceBackNode(node);
  }

  template <typename ConstructorT>
  adt::Result<T> TryNew(const ConstructorT& Constructor) {
    tNodeId<size_t> node_id{nodes_.size()};
    ADT_LET_CONST_REF(node,
                      Constructor(Node<T>{node_id, this->shared_from_this()}));
    return EmplaceBackNode(node);
  }

  adt::Result<NodeList<T>> DownstreamNodes4SrcNodeId(
      const tNodeId<size_t>& src_id) {
    if (src_id.value() >= src_node_id2downstream_nodes_.size()) {
      return adt::errors::IndexError{"src node_id out of ranges."};
    }
    return src_node_id2downstream_nodes_.at(src_id.value());
  }

  adt::Result<NodeList<T>> UpstreamNodes4DstNodeId(
      const tNodeId<size_t>& dst_id) {
    if (dst_id.value() >= dst_node_id2upstream_nodes_.size()) {
      return adt::errors::IndexError{"dst node_id out of ranges."};
    }
    return dst_node_id2upstream_nodes_.at(dst_id.value());
  }

  adt::Result<adt::Ok> Connect(
      const Node<T>& src_node,
      const ValidListTag<std::monostate>& src_downstream_type,
      const Node<T>& dst_node,
      const ValidListTag<std::monostate>& dst_unstream_type) {
    const auto& src_id = src_node.node_id();
    if (src_node.node_arena().lock() != this->shared_from_this()) {
      return adt::errors::RuntimeError{
          "Connection between nodes from different arena is not supported. "};
    }
    if (src_id.value() >= this->src_node_id2downstream_nodes_.size()) {
      return adt::errors::IndexError{
          "src_id.value() is out of range "
          "this->src_node_id2downstream_nodes_."};
    }
    const auto& dst_id = dst_node.node_id();
    if (dst_node.node_arena().lock() != this->shared_from_this()) {
      return adt::errors::RuntimeError{
          "Connection between nodes from different arena is not supported. "};
    }
    if (dst_id.value() >= this->dst_node_id2upstream_nodes_.size()) {
      return adt::errors::IndexError{
          "src_id.value() is out of range this->dst_node_id2upstream_nodes_."};
    }
    ADT_LET_CONST_REF(
        downstream_nodes_data,
        GetNodeListData(&src_node_id2downstream_nodes_[src_id.value()],
                        src_downstream_type));
    downstream_nodes_data->emplace_back(
        Node<T>{dst_id, this->shared_from_this()});
    ADT_LET_CONST_REF(
        upstream_nodes_data,
        GetNodeListData(&dst_node_id2upstream_nodes_[dst_id.value()],
                        dst_unstream_type));
    upstream_nodes_data->emplace_back(
        Node<T>{src_id, this->shared_from_this()});
    return adt::Ok{};
  }

  const std::vector<T>& nodes() const { return nodes_; }

 private:
  adt::Result<adt::List<Node<T>>> GetNodeListData(
      NodeList<T>* node_list, const ValidListTag<std::monostate>& type) {
    using RetDataT = adt::List<Node<T>>;
    using RetT = adt::Result<RetDataT>;
    if (node_list->template Has<UndefinedTag<RetDataT>>()) {
      return type.Match(
          [&](const IndexedTag<std::monostate>&) -> RetT {
            IndexedTag<RetDataT> data{RetDataT{}};
            *node_list = data;
            return data.data;
          },
          [&](const UnindexedTag<std::monostate>&) -> RetT {
            UnindexedTag<RetDataT> data{RetDataT{}};
            *node_list = data;
            return data.data;
          });
    }
    const auto& pattern_match = ::common::Overloaded{
        [&](const IndexedTag<RetDataT>& l,
            const IndexedTag<std::monostate>&) -> RetT { return l.data; },
        [&](const UnindexedTag<RetDataT>& l,
            const UnindexedTag<std::monostate>&) -> RetT { return l.data; },
        [&](const auto&, const auto&) -> RetT {
          return adt::errors::TypeError{"ap graph node list type mismatch."};
        }};
    return std::visit(pattern_match, node_list->variant(), type.variant());
  }

  const T& EmplaceBackNode(const T& node) {
    nodes_.emplace_back(node);
    src_node_id2downstream_nodes_.resize(nodes_.size());
    dst_node_id2upstream_nodes_.resize(nodes_.size());
    return nodes_.at(nodes_.size() - 1);
  }

  std::vector<T> nodes_;
  std::vector<NodeList<T>> src_node_id2downstream_nodes_;
  std::vector<NodeList<T>> dst_node_id2upstream_nodes_;
};

template <typename T>
adt::Result<T> Node<T>::Get() const {
  ADT_LET_CONST_REF(arena, adt::WeakPtrLock(this->node_arena()));
  return arena->At(this->node_id());
}

template <typename T>
adt::Result<NodeList<T>> Node<T>::DownstreamNodes() const {
  ADT_LET_CONST_REF(arena, adt::WeakPtrLock(this->node_arena()));
  return arena->DownstreamNodes4SrcNodeId(this->node_id());
}

template <typename T>
adt::Result<NodeList<T>> Node<T>::UpstreamNodes() const {
  ADT_LET_CONST_REF(arena, adt::WeakPtrLock(this->node_arena()));
  return arena->UpstreamNodes4DstNodeId(this->node_id());
}

template <typename T>
adt::Result<adt::Ok> Node<T>::ConnectTo(
    const Node& dst_node,
    const ValidListTag<std::monostate>& src_downstream_type,
    const ValidListTag<std::monostate>& dst_unstream_type) const {
  ADT_LET_CONST_REF(arena, adt::WeakPtrLock(this->node_arena()));
  return arena->Connect(
      *this, src_downstream_type, dst_node, dst_unstream_type);
}

}  // namespace ap::graph
