// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/set_transformer_input_convert_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
SetTransformerInputConvertPass::SetTransformerInputConvertPass() {
  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .End();
}
namespace patterns {

void SetTransformerInputConvert::operator()() {
  std::unordered_set<std::string> lookup_table_ops{"lookup_table",
                                                   "lookup_table_v2"};
  // Create nodes for lookup_table1 op.
  auto *lookup_table1_x = pattern->NewNode(lookup_table1_x_repr())
                              ->assert_is_ops_input(lookup_table_ops, "Ids");
  auto *lookup_table1_w = pattern->NewNode(lookup_table1_w_repr())
                              ->assert_is_ops_input(lookup_table_ops, "W");
  auto *lookup_table1_op =
      pattern->NewNode(lookup_table1_repr())->assert_is_ops(lookup_table_ops);
  auto *lookup_table1_out = pattern->NewNode(lookup_table1_out_repr())
                                ->assert_is_ops_output(lookup_table_ops)
                                ->AsIntermediate()
                                ->assert_is_op_input("elementwise_add", "X");

  // Create nodes for lookup_table2 op.
  auto *lookup_table2_x = pattern->NewNode(lookup_table2_x_repr())
                              ->assert_is_ops_input(lookup_table_ops, "Ids");
  auto *lookup_table2_w = pattern->NewNode(lookup_table2_w_repr())
                              ->assert_is_ops_input(lookup_table_ops, "W");
  auto *lookup_table2_op =
      pattern->NewNode(lookup_table2_repr())->assert_is_ops(lookup_table_ops);
  auto *lookup_table2_out = pattern->NewNode(lookup_table2_out_repr())
                                ->assert_is_ops_output(lookup_table_ops)
                                ->AsIntermediate()
                                ->assert_is_op_input("elementwise_add", "Y");

  // Create nodes for elementwise_add op.
  auto *elementwise_op =
      pattern->NewNode(elementwise_repr())->assert_is_op("elementwise_add");
  auto *elementwise_out = pattern->NewNode(elementwise_out_repr())
                              ->AsOutput()
                              ->assert_is_only_output_of_op("elementwise_add");

  // links nodes.
  lookup_table1_op->LinksFrom({lookup_table1_x, lookup_table1_w})
      .LinksTo({lookup_table1_out});
  lookup_table2_op->LinksFrom({lookup_table2_x, lookup_table2_w})
      .LinksTo({lookup_table2_out});
  elementwise_op->LinksFrom({lookup_table1_out, lookup_table2_out})
      .LinksTo({elementwise_out});
}

}  // namespace patterns

void SetTransformerInputConvertPass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init(name_scope_, graph);
  int found_subgraph_count = 0;

  GraphPatternDetector gpd;
  patterns::SetTransformerInputConvert fused_pattern(
      gpd.mutable_pattern(), "transformer_input_convert_pass");
  fused_pattern();

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "transformer_input_convert_pass in op compat failed.";
      return;
    }

    VLOG(3) << "transformer_input_convert_pass for pos_id, max_seqlen";

    GET_IR_NODE_FROM_SUBGRAPH(lookup_table2_x, lookup_table2_x, fused_pattern);

    // create op, var in graph
    OpDesc new_desc;
    new_desc.SetType("transformer_input_convert");

    // inputs
    new_desc.SetInput("X", {lookup_table2_x->Name()});

    // outputs
    std::vector<std::string> output_0 = {"pos_id_tensor"};
    std::vector<std::string> output_1 = {"max_seqlen_tensor"};
    new_desc.SetOutput("PosId", output_0);
    new_desc.SetOutput("MaxSeqlen", output_1);

    std::string transformer_input_convert_out0_name = "pos_id_tensor";
    std::string transformer_input_convert_out1_name = "max_seqlen_tensor";
    VarDesc transformer_input_convert_out0(transformer_input_convert_out0_name);
    VarDesc transformer_input_convert_out1(transformer_input_convert_out1_name);
    transformer_input_convert_out0.SetDataType(proto::VarType::INT32);
    transformer_input_convert_out1.SetDataType(proto::VarType::INT32);
    transformer_input_convert_out0.SetShape({-1});
    transformer_input_convert_out1.SetShape({-1});
    transformer_input_convert_out0.SetPersistable(false);
    transformer_input_convert_out1.SetPersistable(false);

    auto new_op_node = graph->CreateOpNode(&new_desc);
    auto transformer_input_convert_out0_node =
        graph->CreateVarNode(&transformer_input_convert_out0);
    auto transformer_input_convert_out1_node =
        graph->CreateVarNode(&transformer_input_convert_out1);

    // needn't create variable in scope

    IR_NODE_LINK_TO(lookup_table2_x, new_op_node);
    IR_NODE_LINK_TO(new_op_node, transformer_input_convert_out0_node);
    IR_NODE_LINK_TO(new_op_node, transformer_input_convert_out1_node);

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(set_transformer_input_convert_pass,
              paddle::framework::ir::SetTransformerInputConvertPass);
REGISTER_PASS_CAPABILITY(set_transformer_input_convert_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("lookup_table", 1)
            .LE("lookup_table_v2", 1)
            .LE("elementweise_add", 1));
