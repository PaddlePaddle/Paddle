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

#include <string>
#include <unordered_set>

#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using framework::Graph;
using framework::Node;
using framework::NodeData;

using common::GraphEdge;
using common::GraphNode;

using InputToNodeMap =
    std::unordered_map<std::string, std::unordered_set<Node*>>;
using shape_dict_t = absl::flat_hash_map<std::string, framework::shape_t>;

std::unordered_set<std::string> unordered_ops = {
    "elementwise_add",
    "elementwise_mul",
    "max",
    "min",
    "logical_and",
    "logical_or",
    "logical_xor",
    "equal",
    "not_equal",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_and",
};

// When all the inputs are the same, those ops just ensure that all the outputs
// shape is the same.
std::unordered_set<std::string> reshape_ops = {
    "reshape",
    "concat",
};

// Those special attrs maybe different but equivalent.
std::unordered_map<std::string, int> special_attrs = {
    //    {"axis", 1}, // due to the issue in some ops
    //    {"dim", 1}, // due to the issue in some ops
    {"axes", 2},
    {"perm", 2}};

bool IsSameSubexpression(Node* op1,
                         Node* op2,
                         shape_dict_t& shape_dict) {  // NOLINT
  // Get the input edges for op1 and op2 in order.
  auto op1_in_edges = op1->inlinks_in_order();
  auto op2_in_edges = op2->inlinks_in_order();
  // Get the number of input edges for op1 and op2
  auto op1_inputs_size = op1_in_edges.size();
  auto op2_inputs_size = op2_in_edges.size();
  // If the number of input edges is not the same, the subexpression is not the
  // same.
  if (op1_inputs_size != op2_inputs_size) {
    return false;
  }
  // Get the number of attributes for op1 and op2.
  auto op1_attrs_size = op1->attrs.attr_store.size();
  auto op2_attrs_size = op2->attrs.attr_store.size();
  // If the number of attributes is not the same, the subexpression is not the
  // same.
  if (op1_attrs_size != op2_attrs_size) {
    return false;
  }
  // Check if the input nodes match.
  if (unordered_ops.count(op1->op()->name)) {
    // For unordered ops, check if any input node of op2 matches any input node
    // of op1.
    for (auto& op1_edge : op1_in_edges) {
      auto* op1_source_node = op1_edge->source()->safe_as<NodeData>();
      CHECK(op1_source_node);
      bool op1_equal_op2 = std::any_of(
          op2_in_edges.begin(),
          op2_in_edges.end(),
          [&](common::Shared<GraphEdge>& edge) {
            auto* op2_source_node = edge->source()->safe_as<NodeData>();
            CHECK(op2_source_node);
            if (op1_source_node->id() == op2_source_node->id()) {
              return true;
            }
            return false;
          });
      if (!op1_equal_op2) {
        return false;
      }
    }
  } else {
    // For ordered ops, check if the input nodes match one-to-one.
    for (int i = 0; i < op1_inputs_size; ++i) {
      auto* op1_source_node = op1_in_edges[i]->source()->safe_as<NodeData>();
      auto* op2_source_node = op2_in_edges[i]->source()->safe_as<NodeData>();
      CHECK(op1_source_node);
      CHECK(op2_source_node);
      if (op1_source_node->id() != op2_source_node->id()) {
        return false;
      }
    }
  }

  // Check if the number of dimensions is the same.
  auto* op1_sink_node = GetNodeData(op1);
  auto* op2_sink_node = GetNodeData(op2);
  if (shape_dict[op1_sink_node->id()].size() !=
      shape_dict[op2_sink_node->id()].size()) {
    return false;
  }
  if (reshape_ops.count(op1->op()->name)) {
    // For reshape ops, check if the reshaped shape is the same.
    return shape_dict[op1_sink_node->id()] == shape_dict[op2_sink_node->id()];
  } else {
    // For non-reshape ops, check if the attributes is the same.
    return std::all_of(
        op1->attrs.attr_store.begin(),
        op1->attrs.attr_store.end(),
        [&](auto attr) {
          if (!op2->attrs.attr_store.count(attr.first)) {
            return false;
          }
          auto& attr1 = attr.second;
          auto& attr2 = op2->attrs.attr_store[attr.first];
          auto ndim = static_cast<int>(shape_dict[op1_sink_node->id()].size());
          if (special_attrs.count(attr.first)) {
            switch (special_attrs[attr.first]) {
              case 1: {
                auto op1_axis = absl::get<int>(attr1);
                auto op2_axis = absl::get<int>(attr2);
                if (op1_axis < 0) {
                  op1_axis += ndim;
                }
                if (op2_axis < 0) {
                  op2_axis += ndim;
                }
                return op2_axis == op1_axis;
              }
              case 2: {
                auto& op1_axes = absl::get<std::vector<int>>(attr1);
                auto& op2_axes = absl::get<std::vector<int>>(attr2);
                auto op1_size = op1_axes.size();
                auto op2_size = op2_axes.size();
                if (op1_size != op2_size) {
                  return false;
                }
                for (int i = 0; i < op1_axes.size(); ++i) {
                  int op1_axis = op1_axes[i];
                  int op2_axis = op2_axes[i];
                  if (op1_axis < 0) {
                    op1_axis += ndim;
                  }
                  if (op2_axis < 0) {
                    op2_axis += ndim;
                  }
                  if (op2_axis != op1_axis) {
                    return false;
                  }
                }
                return true;
              }
            }
          }
          return attr1 == attr2;
        });
  }
}

void RemoveNodes(framework::Graph* graph, GraphNode* node) {
  auto in_edges = node->inlinks();
  for (auto& edge : in_edges) {
    auto* in_node = edge->source();
    in_node->UnLinkSingleTo(node);
  }
  auto out_edges = node->outlinks();
  for (auto& edge : out_edges) {
    auto* out_node = edge->sink();
    node->UnLinkSingleTo(out_node);
  }
  graph->DropNode(node);
}

void RemoveNodes(framework::Graph* graph, const std::vector<Node*>& nodes) {
  for (auto* node : nodes) {
    RemoveNodes(graph, node);
  }
}

void RemoveNodes(framework::Graph* graph,
                 const std::vector<NodeData*>& nodes_data) {
  for (auto* data : nodes_data) {
    if (std::find(graph->outputs.begin(), graph->outputs.end(), data) !=
        graph->outputs.end()) {
      return;
    }
    RemoveNodes(graph, data);
  }
}

void ReplaceNode(NodeData* src_new, NodeData* src_old, Node* trt) {
  std::vector<NodeData*> in_nodes;
  for (auto& in_edge : trt->inlinks_in_order()) {
    auto* in_node = in_edge->source()->safe_as<NodeData>();
    if (in_node->id() == src_old->id()) {
      in_node->UnLinkSingleTo(trt);
      src_new->LinkTo(trt);
    }
  }
}

void CommonSubexpressionElimination(Graph* graph,
                                    std::vector<GraphNode*> store_nodes,
                                    InputToNodeMap in2node) {
  std::unordered_map<std::string, std::vector<Node*>> candidates_map;
  auto shape_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, framework::shape_t>>(
          "infershape");
  std::vector<Node*> remove_nodes;
  std::vector<NodeData*> remove_nodes_data;

  while (!store_nodes.empty()) {
    auto* graph_node = store_nodes[0];
    store_nodes.erase(store_nodes.begin());
    VLOG(4) << "size of store_nodes is " << store_nodes.size();
    auto node = graph_node->safe_as<Node>();
    if (node) {
      auto& node_type = node->op()->name;
      auto& candidates = candidates_map[node_type];
      bool found = false;
      for (auto* candidate_node : candidates) {
        // If node is same with candidate_node, continue the next.
        if (node->id() == candidate_node->id()) continue;
        // If node is different from candidate_node, continue the next.
        if (!IsSameSubexpression(node, candidate_node, shape_dict)) continue;
        found = true;
        for (int k = 0; k < node->outlinks_in_order().size(); ++k) {
          const auto& out_links = node->outlinks_in_order();
          const auto& candidate_out_links = candidate_node->outlinks_in_order();
          CHECK(out_links.size() == candidate_out_links.size());
          auto* sink_node = out_links[k]->sink()->safe_as<NodeData>();
          auto* candidate_sink_node =
              candidate_out_links[k]->sink()->safe_as<NodeData>();
          CHECK(sink_node);
          CHECK(candidate_sink_node);
          auto iter_sink_node = std::find(
              graph->outputs.begin(), graph->outputs.end(), sink_node);
          if (iter_sink_node != graph->outputs.end()) {
            // If sink node in outputs, the node cannot be removed.
            continue;
          }
          remove_nodes_data.push_back(sink_node);
          // Replace sink_node with candidate_sink_node in nodes linked by
          // sink_node.
          auto out_nodes = in2node[sink_node->id()];
          for (auto out_node : out_nodes) {
            ReplaceNode(candidate_sink_node, sink_node, out_node);
            // The changed out node will be detected again.
            if (std::find(store_nodes.begin(), store_nodes.end(), out_node) ==
                store_nodes.end()) {
              store_nodes.insert(store_nodes.begin(), out_node);
            }
          }
        }
        remove_nodes.push_back(node);
        VLOG(4) << "remove " << node->id() << " node.";
        break;
      }
      if (!found) {
        candidates_map[node_type].push_back(node);
      }
    }
  }
  // Node should be deleted before node data.
  RemoveNodes(graph, remove_nodes);
  RemoveNodes(graph, remove_nodes_data);
}

void CommonSubexpressionEliminationPass(Graph* graph) {
  VLOG(3) << "CommonSubexpressionEliminationPass...!";
  std::unordered_map<std::string, std::vector<Node*>> candidates_map;
  InputToNodeMap in2node;
  auto store_nodes = std::get<0>(graph->topological_order());

  for (auto& graph_node : store_nodes) {
    auto node = graph_node->safe_as<Node>();
    if (node) {
      for (auto& in_edge : node->inlinks_in_order()) {
        auto* source_node = in_edge->source()->safe_as<NodeData>();
        in2node[source_node->id()].insert(node);
      }
    }
  }

  CommonSubexpressionElimination(graph, store_nodes, in2node);
  VLOG(3) << "CommonSubexpressionEliminationPass Finish...!";
}
}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(CommonSubexpressionEliminationPass) {
  CINN_REGISTER_PASS(CommonSubexpressionEliminationPass)
      .describe("This pass  will remove these same sub-expression.")
      .set_change_structure(true)
      .set_body(cinn::hlir::pass::CommonSubexpressionEliminationPass);

  return true;
}
