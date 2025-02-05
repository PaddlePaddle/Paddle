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

#include "paddle/fluid/framework/ir/set_subgraph_edge_pass.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace paddle::framework::ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES GET_IR_NODE(ops);

// Delete dequantize_linear_op, then dequantize weight
void SetSubgraphEdge::ApplyImpl(Graph *graph) const {
  if (!(FLAGS_all_blocks_convert_trt && FLAGS_convert_all_blocks)) {
    VLOG(3) << "Running set_subgraph_edge_pass need set environment variables: "
               "export FLAGS_convert_all_blocks = true";
    return;
  }

  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));

  VLOG(3) << "Running set_subgraph_edge_pass.";
  if (graph->IsMainGraph()) {
    VLOG(3)
        << "The ID of block running set_subgraph_edge_pass is: 0(main_graph)";
  } else {
    VLOG(3) << "The ID of block running set_subgraph_edge_pass is: "
            << graph->GetBlockId();
  }

  const std::string pattern_name = "subgraph_edge_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto *scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::InvalidArgument("Scope in SetSubgraphEdge should not be "
                                      "null."));
  // Create pattern
  patterns::SubgraphEdgePattern pattern(gpd.mutable_pattern(), pattern_name);

  std::unordered_set<std::string> ops_type = {"conditional_block", "while"};
  pattern(ops_type);

  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    GET_NODES;
    std::unordered_set<const Node *> nodes2rm = {};

    // block ops
    auto *block_op = ops->Op();

    // block inputs
    std::vector<std::string> block_inputs = {};
    if (block_op->Type() == "while") {
      block_inputs = block_op->Input("X");
    } else {
      block_inputs = block_op->Input("Input");
    }

    // block outputs
    auto block_outputs = block_op->Output("Out");

    // block id
    size_t block_id = -1;
    for (const auto &attr : block_op->Proto()->attrs()) {
      if (attr.type() == proto::AttrType::BLOCK) {
        block_id = attr.block_idx();
      }
    }
    Graph *sub_graph = nullptr;
    if (graph->IsMainGraph()) {
      sub_graph = graph->GetSubGraph(block_id);
    } else {
      sub_graph = graph->GetMainGraph()->GetSubGraph(block_id);
    }

    for (auto input_name : block_inputs) {
      ir::Node *subgraph_node = nullptr;
      for (auto *node : sub_graph->Nodes()) {
        if (node->IsVar()) {
          if (node->Var()->Name() == input_name) {
            subgraph_node = node;
          }
        }
      }
      if (subgraph_node) {
        subgraph_node->SetSubgraphInput();
      }
    }

    for (auto output_name : block_outputs) {
      ir::Node *subgraph_node = nullptr;
      for (auto *node : sub_graph->Nodes()) {
        if (node->IsVar()) {
          if (node->Var()->Name() == output_name) {
            subgraph_node = node;
          }
        }
      }
      if (subgraph_node) {
        subgraph_node->SetSubgraphOutput();
      } else {
        PADDLE_THROW(common::errors::Fatal("Subgraph don't have block node."));
      }
    }
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(set_subgraph_edge_pass, paddle::framework::ir::SetSubgraphEdge);
