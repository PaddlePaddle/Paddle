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

#include <memory>
#include "paddle/cinn/api/tensor_node.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/pass/fusion_helper_base.h"

namespace cinn {
namespace api {

using OpPatternKind = cinn::hlir::framework::OpPatternKind;
using Attribute = cinn::utils::Attribute;

class OpNode {
 public:
  OpNode(const hlir::framework::Node* node, const hlir::framework::Graph* graph)
      : node_(node),
        graph_(graph),
        input_tensors_(node->inlinks_in_order(), graph_),
        output_tensors_(node->outlinks_in_order(), graph_) {
    VLOG(1) << "[OpNode] node: " << node->id();
  }

  OpNode(const OpNode& other)
      : node_(other.node_),
        graph_(other.graph_),
        input_tensors_(node_->inlinks_in_order(), graph_),
        output_tensors_(node_->outlinks_in_order(), graph_) {}

  OpPatternKind kind() const {
    static const hlir::framework::OpValueType<OpPatternKind>& op_pattern_dict =
        hlir::framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
    auto kind = op_pattern_dict[node_->op()];

    if (kind == hlir::framework::kBroadcast) {
      // As binary op was defined as broadcast, actually it should be
      // element-wise.
      if (node_->op()->name != "broadcast_to") {
        return hlir::framework::kElementWise;
      }
    }
    return kind;
  }

  class InputTensorListView {
   public:
    InputTensorListView(
        const std::vector<common::Shared<common::GraphEdge>>& edges,
        const hlir::framework::Graph* graph)
        : edges_(edges), graph_(graph) {}

    InputTensorListView(const InputTensorListView& other) = delete;
    InputTensorListView(InputTensorListView&& other) = delete;

    InputTensorListView& operator=(const InputTensorListView& other) = delete;

    size_t size() const { return edges_.size(); }

    TensorNode operator[](size_t index) const;

   private:
    std::vector<common::Shared<common::GraphEdge>> edges_;
    const hlir::framework::Graph* graph_;
  };

  class OutputTensorListView {
   public:
    OutputTensorListView(
        const std::vector<common::Shared<common::GraphEdge>>& edges,
        const hlir::framework::Graph* graph)
        : edges_(edges), graph_(graph) {}

    OutputTensorListView(const OutputTensorListView& other) = delete;
    OutputTensorListView(OutputTensorListView&& other) = delete;

    OutputTensorListView& operator=(const OutputTensorListView& other) = delete;

    size_t size() const { return edges_.size(); }

    TensorNode operator[](size_t index) const;

   private:
    std::vector<common::Shared<common::GraphEdge>> edges_;
    const hlir::framework::Graph* graph_;
  };

  bool operator==(const OpNode& other) const { return node_ == other.node_; }

  bool operator<(const OpNode& other) const { return node_ < other.node_; }

  const InputTensorListView& inputs() const { return input_tensors_; }

  const OutputTensorListView& outputs() const { return output_tensors_; }

  template <typename T>
  const T& GetAttr(const std::string& attr_name) const {
    return absl::get<T>(GetAttr(attr_name));
  }

 private:
  const Attribute& GetAttr(const std::string& attr_name) const {
    return node_->attrs.attr_store.at(attr_name);
  }

  friend struct std::hash<OpNode>;

  const hlir::framework::Node* node_;
  const hlir::framework::Graph* graph_;

  const InputTensorListView input_tensors_;
  const OutputTensorListView output_tensors_;
};

}  // namespace api
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::api::OpNode> {
  size_t operator()(const cinn::api::OpNode& obj) const {
    return std::hash<int64_t>()(reinterpret_cast<uint64_t>(obj.node_));
  }
};

}  // namespace std
