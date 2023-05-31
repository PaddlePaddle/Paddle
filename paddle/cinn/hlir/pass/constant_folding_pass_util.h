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

#include <queue>

#include "cinn/hlir/pass/fusion_helper_base.h"
#include "cinn/utils/functional.h"

namespace cinn {
namespace hlir {
namespace pass {

inline void fold_broadcast_to_constant(const FusionHelperBase* helper,
                                       Graph* graph,
                                       Node* node) {
  auto constant_op = helper->GetProducerNode(node)[0];
  CHECK(node->attrs.attr_store.count("out_shape"));
  auto shape =
      absl::get<std::vector<int>>(node->attrs.attr_store.at("out_shape"));
  CHECK(constant_op->attrs.attr_store.count("value"));
  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"),
                            "fill_constant",
                            common::UniqName("fill_constant"));
  // set node attr
  node_tmp->attrs.attr_store["dtype"] =
      constant_op->attrs.attr_store.at("dtype");
  node_tmp->attrs.attr_store["shape"] = shape;
  node_tmp->attrs.attr_store["value"] =
      constant_op->attrs.attr_store.at("value");
  node_tmp->attrs.attr_store["force_cpu"] = false;
  graph->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.Reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto constant_node_data = helper->GetNodeData(constant_op);
  if (constant_node_data->outlinks().size() == 1) {
    graph->DropNode(node);
    graph->DropNode(constant_op);
    graph->DropNode(constant_node_data);
  } else {
    constant_node_data->UnLinkSingleTo(node);
    graph->DropNode(node);
  }
}

inline void fold_reshape_fill_constant(const FusionHelperBase* helper,
                                       Graph* graph,
                                       Node* node) {
  auto constant_op = helper->GetProducerNode(node)[0];
  CHECK(node->attrs.attr_store.count("shape"));
  auto shape = absl::get<std::vector<int>>(node->attrs.attr_store.at("shape"));
  CHECK(constant_op->attrs.attr_store.count("value"));

  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"),
                            "fill_constant",
                            common::UniqName("fill_constant"));
  // set node attr
  node_tmp->attrs.attr_store["dtype"] =
      constant_op->attrs.attr_store.at("dtype");
  node_tmp->attrs.attr_store["shape"] = shape;
  node_tmp->attrs.attr_store["value"] =
      constant_op->attrs.attr_store.at("value");
  node_tmp->attrs.attr_store["force_cpu"] = false;
  graph->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.Reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto constant_node_data = helper->GetNodeData(constant_op);
  if (constant_node_data->outlinks().size() == 1) {
    graph->DropNode(node);
    graph->DropNode(constant_op);
    graph->DropNode(constant_node_data);
  } else {
    constant_node_data->UnLinkSingleTo(node);
    graph->DropNode(node);
  }
}

inline void fold_squeeze_fill_constant(const FusionHelperBase* helper,
                                       Graph* graph,
                                       Node* node) {
  auto constant_op = helper->GetProducerNode(node)[0];
  CHECK(constant_op->attrs.attr_store.count("shape"));
  auto shape =
      absl::get<std::vector<int>>(constant_op->attrs.attr_store.at("shape"));
  CHECK(node->attrs.attr_store.count("axes"));
  auto axes = absl::get<std::vector<int>>(node->attrs.attr_store.at("axes"));

  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"),
                            "fill_constant",
                            common::UniqName("fill_constant"));
  // set node attr
  std::vector<int> n_shape;
  if (axes.size() == 0) {
    for (auto s : shape) {
      if (s > 1) {
        n_shape.push_back(s);
      }
    }
  } else {
    for (int idx = 0; idx < shape.size(); ++idx) {
      if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
        n_shape.push_back(shape[idx]);
      }
    }
  }

  node_tmp->attrs.attr_store["dtype"] =
      constant_op->attrs.attr_store.at("dtype");
  node_tmp->attrs.attr_store["shape"] = n_shape;
  node_tmp->attrs.attr_store["value"] =
      constant_op->attrs.attr_store.at("value");
  node_tmp->attrs.attr_store["force_cpu"] = false;
  graph->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.Reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto constant_node_data = helper->GetNodeData(constant_op);
  if (constant_node_data->outlinks().size() == 1) {
    graph->DropNode(node);
    graph->DropNode(constant_op);
    graph->DropNode(constant_node_data);
  } else {
    constant_node_data->UnLinkSingleTo(node);
    graph->DropNode(node);
  }
}

inline void fold_expand_dims_fill_constant(const FusionHelperBase* helper,
                                           Graph* graph,
                                           Node* node) {
  auto constant_op = helper->GetProducerNode(node)[0];
  CHECK(constant_op->attrs.attr_store.count("shape"));
  auto shape =
      absl::get<std::vector<int>>(constant_op->attrs.attr_store.at("shape"));
  CHECK(node->attrs.attr_store.count("axes"));
  auto axes = absl::get<std::vector<int>>(node->attrs.attr_store.at("axes"));

  // create constant op.
  Node* node_tmp = new Node(Operator::Get("fill_constant"),
                            "fill_constant",
                            common::UniqName("fill_constant"));
  int shape_size = shape.size();
  int axes_size = axes.size();
  int total_size = shape_size + axes_size;
  // check axes whether in range [-total_size, total_size-1] and convert all to
  // [0, total_size-1].
  axes = utils::GetPositiveAxes(axes, total_size);
  // check axes can't repeat.
  std::sort(axes.begin(), axes.end(), std::less<int>());
  for (int idx = 0; idx < axes_size - 1; ++idx) {
    CHECK_NE(axes[idx], axes[idx + 1]);
  }
  // insert 1 to new shape.
  std::vector<int> n_shape(total_size, 1);
  for (int idx = 0, index = 0; idx < n_shape.size(); ++idx) {
    if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
      n_shape[idx] = shape[index++];
    }
  }

  // set node attr
  node_tmp->attrs.attr_store["dtype"] =
      constant_op->attrs.attr_store.at("dtype");
  node_tmp->attrs.attr_store["shape"] = n_shape;
  node_tmp->attrs.attr_store["value"] =
      constant_op->attrs.attr_store.at("value");
  node_tmp->attrs.attr_store["force_cpu"] = false;
  graph->RegisterNode(node_tmp->id(), node_tmp);
  // create new link.
  NodeData* node_data = helper->GetNodeData(node);
  node_data->source_node.Reset(node_tmp);
  node->UnLinkSingleTo(node_data);
  node_tmp->LinkTo(node_data);

  // drop node.
  auto constant_node_data = helper->GetNodeData(constant_op);
  if (constant_node_data->outlinks().size() == 1) {
    graph->DropNode(node);
    graph->DropNode(constant_op);
    graph->DropNode(constant_node_data);
  } else {
    constant_node_data->UnLinkSingleTo(node);
    graph->DropNode(node);
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
