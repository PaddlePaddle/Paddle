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

#include "paddle/fluid/framework/ir/transpose_flatten_concat_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

TransposeFlattenConcatFusePass::TransposeFlattenConcatFusePass() {
  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int>>()
      .End();
  AddOpCompat(OpCompat("flatten2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumGE(0)
      .End();
  AddOpCompat(OpCompat("concat"))
      .AddInput("X")  // Input("X"): vector<tensors>
      .End()
      .AddInput("AxisTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({0, 1})
      .End();
}

void TransposeFlattenConcatFusePass::RunTransposeFlattenConcatFuse(
    ir::Graph *graph, int times) const {
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
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }

    const int kNumFields = 5;
    const int kTransOffset = 1;
    const int kTransOutOffset = 2;
    const int kFlattenOffset = 3;
    const int kFlattenOutOffset = 4;

    std::vector<Node *> nodes;
    std::vector<int> trans_axis0;
    int flatten_axis0;
    for (int i = 0; i < times; i++) {
      PADDLE_ENFORCE_NOT_NULL(
          subgraph.at(pattern.GetPDNode("transpose" + std::to_string(i))),
          platform::errors::NotFound("Can not find transpose%d in subgraph.",
                                     i));
      PADDLE_ENFORCE_NOT_NULL(
          subgraph.at(pattern.GetPDNode("transpose_out" + std::to_string(i))),
          platform::errors::NotFound(
              "Can not find transpose_out%d in subgraph.", i));
      PADDLE_ENFORCE_NOT_NULL(
          subgraph.at(pattern.GetPDNode("flatten" + std::to_string(i))),
          platform::errors::NotFound("Can not find flatten%d in subgraph.", i));
      PADDLE_ENFORCE_NOT_NULL(
          subgraph.at(pattern.GetPDNode("flatten_out" + std::to_string(i))),
          platform::errors::NotFound("Can not find flatten_out%d in subgraph.",
                                     i));
      PADDLE_ENFORCE_NOT_NULL(
          subgraph.at(input_nodes[i]),
          platform::errors::NotFound("Can not find %s in subgraph.",
                                     input_nodes[i]->name()));

      if (i == 0) {
        trans_axis0 = BOOST_GET_CONST(
            std::vector<int>,
            subgraph.at(pattern.GetPDNode("transpose" + std::to_string(0)))
                ->Op()
                ->GetAttr("axis"));
        flatten_axis0 = BOOST_GET_CONST(
            int, subgraph.at(pattern.GetPDNode("flatten" + std::to_string(0)))
                     ->Op()
                     ->GetAttr("axis"));
      } else {
        std::vector<int> trans_axis = BOOST_GET_CONST(
            std::vector<int>,
            subgraph.at(pattern.GetPDNode("transpose" + std::to_string(i)))
                ->Op()
                ->GetAttr("axis"));
        // All axis of transpose should be the same
        if (trans_axis0 != trans_axis) return;

        int flatten_axis = BOOST_GET_CONST(
            int, subgraph.at(pattern.GetPDNode("flatten" + std::to_string(0)))
                     ->Op()
                     ->GetAttr("axis"));
        // All axis of flatten should be the same
        if (flatten_axis0 != flatten_axis) return;
      }

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
    std::vector<int> trans_axis = BOOST_GET_CONST(
        std::vector<int>, nodes[kTransOffset]->Op()->GetAttr("axis"));
    int flatten_axis =
        BOOST_GET_CONST(int, nodes[kFlattenOffset]->Op()->GetAttr("axis"));
    int concat_axis = BOOST_GET_CONST(int, concat_op->Op()->GetAttr("axis"));
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
REGISTER_PASS_CAPABILITY(transpose_flatten_concat_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("transpose", 0)
            .EQ("transpose2", 0)
            .EQ("flatten", 0)
            .EQ("concat", 0)
            .EQ("fusion_transpose_flatten_concat", 0));
