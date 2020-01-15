// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/transpose_flatten_concat_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

void RunTransposeFlattenConcatFuse(ir::Graph *graph, int times) {
  const std::string pattern_name =
      "transpose_flatten" + std::to_string(times) + "_concat_fuse";

  GraphPatternDetector gpd;
  std::vector<PDNode *> input_nodes;
  for (int i = 0; i < times; i++) {
    input_nodes.push_back(gpd.mutable_pattern()
                              ->NewNode("x" + std::to_string(i))
                              ->assert_is_op_input("transpose2", "X")
                              ->AsInput());
  }

  patterns::TransposeFlattenConcat pattern(gpd.mutable_pattern(), pattern_name);
  pattern(input_nodes, times);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    const int kNumFields = 5;
    const int kTransOffset = 1;
    const int kTransOutOffset = 2;
    const int kFlattenOffset = 3;
    const int kFlattenOutOffset = 4;
    std::vector<Node *> nodes;

    for (int i = 0; i < times; i++) {
      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("transpose" + std::to_string(i))));
      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("transpose_out" + std::to_string(i))));
      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("flatten" + std::to_string(i))));
      PADDLE_ENFORCE(
          subgraph.at(pattern.GetPDNode("flatten_out" + std::to_string(i))));
      PADDLE_ENFORCE(subgraph.at(input_nodes[i]));

      nodes.push_back(subgraph.at(input_nodes[i]));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("transpose" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("transpose_out" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("flatten" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("flatten_out" + std::to_string(i))));
    }

    Node *concat_op = subgraph.at(pattern.GetPDNode("concat"));
    Node *concat_out = subgraph.at(pattern.GetPDNode("concat_out"));
    std::vector<std::string> input_names;
    std::vector<int> trans_axis = boost::get<std::vector<int>>(
        nodes[kTransOffset]->Op()->GetAttr("axis"));
    int flatten_axis =
        boost::get<int>(nodes[kFlattenOffset]->Op()->GetAttr("axis"));
    int concat_axis = boost::get<int>(concat_op->Op()->GetAttr("axis"));
    std::string output_name = concat_out->Name();

    for (int i = 0; i < times; i++) {
      input_names.push_back(nodes[i * kNumFields]->Name());
    }

    framework::OpDesc new_op_desc;
    new_op_desc.SetType("fusion_transpose_flatten_concat");
    new_op_desc.SetInput("X", input_names);
    new_op_desc.SetAttr("trans_axis", trans_axis);
    new_op_desc.SetAttr("flatten_axis", flatten_axis);
    new_op_desc.SetAttr("concat_axis", concat_axis);
    new_op_desc.SetOutput("Out", {output_name});
    new_op_desc.Flush();

    // Create a new node for the fused op.
    auto *new_conv_op = graph->CreateOpNode(&new_op_desc);

    std::unordered_set<const Node *> delete_nodes;

    for (int i = 0; i < times; i++) {
      nodes[i * kNumFields]->outputs.push_back(new_conv_op);
      new_conv_op->inputs.push_back(nodes[i * kNumFields]);
      delete_nodes.insert(nodes[i * kNumFields + kTransOffset]);
      delete_nodes.insert(nodes[i * kNumFields + kTransOutOffset]);
      delete_nodes.insert(nodes[i * kNumFields + kFlattenOffset]);
      delete_nodes.insert(nodes[i * kNumFields + kFlattenOutOffset]);
    }
    delete_nodes.insert(concat_op);

    new_conv_op->outputs.push_back(concat_out);
    concat_out->inputs.push_back(new_conv_op);

    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph, delete_nodes);
  };

  gpd(graph, handler);
}

void TransposeFlattenConcatFusePass::ApplyImpl(ir::Graph *graph) const {
  const int pattern_nums = 6;
  const std::string pattern_name = "transpose_flatten_concat_fuse";
  FusePassBase::Init(pattern_name, graph);
  for (int i = 1; i <= pattern_nums; i++) {
    RunTransposeFlattenConcatFuse(graph, i);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(transpose_flatten_concat_fuse_pass,
              paddle::framework::ir::TransposeFlattenConcatFusePass);
