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

class OpNode {
 public:
  OpNode(const hlir::framework::Node* node, const hlir::framework::Graph* graph)
      : node_(node),
        graph_(graph),
        input_tensors_(node->inlinks_in_order(), graph_),
        output_tensors_(node->outlinks_in_order(), graph_) {}

  OpNode(const OpNode& other)
      : node_(other.node_),
        graph_(other.graph_),
        input_tensors_(node_->inlinks_in_order(), graph_),
        output_tensors_(node_->outlinks_in_order(), graph_) {}

  using OpPatternKind = cinn::hlir::framework::OpPatternKind;

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

  class TensorListIterator {
   public:
    TensorListIterator(
        std::vector<common::Shared<common::GraphEdge>>::const_iterator it,
        const hlir::framework::Graph* graph,
        std::function<hlir::framework::NodeData*(
            common::Shared<common::GraphEdge>)> get_tensor_from_edge)
        : iter_(it),
          graph_(graph),
          get_tensor_from_edge_(get_tensor_from_edge) {}

    TensorListIterator& operator++() {
      ++iter_;
      return *this;
    }

    TensorListIterator operator++(int) {
      TensorListIterator tmp = *this;
      ++iter_;
      return tmp;
    }

    bool operator==(const TensorListIterator& other) const {
      return iter_ == other.iter_;
    }

    bool operator!=(const TensorListIterator& other) const {
      return !(*this == other);
    }

    TensorNode operator*() const;

   private:
    std::vector<common::Shared<common::GraphEdge>>::const_iterator iter_;
    const hlir::framework::Graph* graph_;
    std::function<hlir::framework::NodeData*(common::Shared<common::GraphEdge>)>
        get_tensor_from_edge_;
  };

  using const_iterator = TensorListIterator;

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

    const_iterator begin() const {
      return const_iterator(
          edges_.begin(), graph_, [](common::Shared<common::GraphEdge> edge) {
            return edge->source()->safe_as<hlir::framework::NodeData>();
          });
    }

    const_iterator end() const {
      return const_iterator(
          edges_.end(), graph_, [](common::Shared<common::GraphEdge> edge) {
            return edge->source()->safe_as<hlir::framework::NodeData>();
          });
    }

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

    const_iterator begin() const {
      return const_iterator(
          edges_.begin(), graph_, [](common::Shared<common::GraphEdge> edge) {
            return edge->sink()->safe_as<hlir::framework::NodeData>();
          });
    }

    const_iterator end() const {
      return const_iterator(
          edges_.end(), graph_, [](common::Shared<common::GraphEdge> edge) {
            return edge->sink()->safe_as<hlir::framework::NodeData>();
          });
    }

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
  using Attribute = cinn::utils::Attribute;
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
    return std::hash<size_t>()(reinterpret_cast<size_t>(obj.node_));
  }
};

}  // namespace std
