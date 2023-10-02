// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <algorithm>
#include <unordered_set>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::Operator;
using framework::OpPatternKind;
using framework::shape_t;

class FusionHelperBase {
 public:
  explicit FusionHelperBase(const framework::Graph* graph)
      : shape_dict_(graph->GetAttrs<absl::flat_hash_map<std::string, shape_t>>(
            "infershape")),
        target_(graph->target_) {
    // get op pattern dict
    op_pattern_dict_ =
        &framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
    // output node set
    for (auto node_data : graph->outputs) {
      CHECK(node_data->source_node.get());
      output_nodes_set_.insert(node_data->source_node.get());
    }
  }

 public:
  OpPatternKind GetOpKind(const framework::Node* node) const {
    CHECK(op_pattern_dict_->Find(node->op()))
        << "Don't find the pattern of op : " << node->id();
    auto kind = op_pattern_dict_[0][node->op()];

    if (kind == framework::kBroadcast) {
      // As binary op was defined as broadcast, actually it should be
      // element-wise.
      if (node->op()->name != "broadcast_to") {
        return framework::kElementWise;
      }
    }

    return kind;
  }

  static bool IsConstOp(const framework::Node* node) {
    static std::unordered_set<std::string> const_op_type = {
        "const_scalar", "fill_constant", "arange"};
    if (const_op_type.count(node->op()->name)) {
      return true;
    } else {
      return false;
    }
  }

  static std::vector<NodeData*> GetNodeDatas(const Node* node) {
    std::vector<NodeData*> consumer_node_data;
    for (auto& edge : node->outlinks_in_order()) {
      auto output = edge->sink()->safe_as<NodeData>();
      CHECK(output) << "The op \"" << node->id()
                    << "\" output should not be empty!";
      consumer_node_data.push_back(output);
    }
    return consumer_node_data;
  }

  NodeData* GetNodeData(const Node* node) const {
    auto node_data = (*node->outlinks().begin())->sink()->safe_as<NodeData>();
    CHECK(node_data);
    return node_data;
  }

  shape_t GetNodeDataShape(const Node* node) const {
    auto* node_data = GetNodeData(node);
    CHECK(shape_dict_.count(node_data->id()))
        << "Can't find " << node_data->id() << " 's shape!";
    return shape_dict_.at(node_data->id());
  }

  shape_t GetNodeInputShape(const Node* node) const {
    auto node_datas = GetProducerNodeData(node);
    CHECK_GT(node_datas.size(), 0);
    CHECK(shape_dict_.count(node_datas[0]->id()))
        << "Can't find " << node_datas[0]->id() << " 's shape!";
    return shape_dict_.at(node_datas[0]->id());
  }

  static std::vector<NodeData*> GetProducerNodeData(const Node* node) {
    std::vector<NodeData*> producer_node_data;
    for (auto& edge : node->inlinks_in_order()) {
      auto graph_node = edge->source();
      auto producer_data = graph_node->safe_as<NodeData>();
      CHECK(producer_data);
      producer_node_data.push_back(producer_data);
    }
    return producer_node_data;
  }

  std::vector<Node*> GetProducerNode(const Node* node) const {
    std::vector<Node*> producer_node;
    for (auto& edge : node->inlinks_in_order()) {
      auto graph_node = edge->source();
      auto producer_data = graph_node->safe_as<NodeData>();
      CHECK(producer_data);
      auto producer = producer_data->source_node.get();
      if (producer) {
        producer_node.push_back(producer);
      }
    }
    return producer_node;
  }

  std::vector<Node*> GetConsumerNode(const Node* node) const {
    std::vector<Node*> consumer_nodes;
    auto node_data = GetNodeData(node);
    for (auto& link : node_data->outlinks()) {
      auto consumer = link->sink()->safe_as<Node>();
      CHECK(consumer);
      consumer_nodes.push_back(consumer);
    }
    return consumer_nodes;
  }

  bool WithoutLastDimInReduce(const std::vector<int>& inshape,
                              const std::vector<int>& axes) const {
    // if last axis is in reduce.
    if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
        std::find(axes.begin(), axes.end(), -1) != axes.end()) {
      return false;
    }

    int sum_last_axes = 1;
    for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      sum_last_axes *= inshape[idx];
    }

    if (sum_last_axes > 1) {
      return true;
    } else {
      return false;
    }
  }

  int GetSharedSize(const Node* node) const {
    auto producers = GetProducerNodeData(node);
    CHECK_GT(producers.size(), 0);
    auto inshape = shape_dict_.at(producers[0]->id());
    auto axes = absl::get<std::vector<int>>(node->attrs.attr_store.at("dim"));
    if (WithoutLastDimInReduce(inshape, axes)) {
      int lane = 1;
      for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
        lane = inshape[idx];
      }
      int max_num_threads = common::DefaultNVGPUTarget().max_num_threads();
      if (lane > max_num_threads / 2) {
        return 0;
      }
      int index = axes.size() - 1;
      for (; index >= 0; --index) {
        if (index + 1 < axes.size() && axes[index] != axes[index + 1] - 1) {
          break;
        }
        lane *= inshape[axes[index]];
        if (lane > max_num_threads / 2) {
          break;
        }
      }
      // if lane > (max_num_threads / 2),the loop break from lane >
      // max_num_threads / 2.
      int axis = lane > (max_num_threads / 2) ? axes[index] : axes[index + 1];
      if (lane <= max_num_threads) {
        return lane * sizeof(float);
      } else {
        int prefix = inshape[axis];
        int tail = lane / prefix;
        for (int idx = max_num_threads / tail;
             idx > ((max_num_threads / 2) / tail);
             --idx) {
          if (prefix % idx == 0) {
            return idx * tail * sizeof(float);
          }
        }
        int num = max_num_threads / tail;
        return num * tail * sizeof(float);
      }
    }
    return 0;
  }
  // target
  const common::Target& target_;
  // output node set
  std::unordered_set<const Node*> output_nodes_set_;
  // shape dict
  const absl::flat_hash_map<std::string, shape_t>& shape_dict_;
  // op pattern dict
  const framework::OpValueType<OpPatternKind>* op_pattern_dict_;
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
