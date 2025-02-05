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

namespace paddle::framework::ir::patterns {

void SetTransformerInputConvert::operator()(const std::string &pos_id) {
  std::unordered_set<std::string> lookup_table_ops{"lookup_table",
                                                   "lookup_table_v2"};
  // Create nodes for lookup_table.
  auto *lookup_table_id =
      pattern->NewNode(lookup_table_id_repr())
          ->assert_is_ops_input(lookup_table_ops, "Ids")
          ->assert_more([&](Node *node) { return node->Name() == pos_id; });
  auto *lookup_table_op =
      pattern->NewNode(lookup_table_repr())->assert_is_ops(lookup_table_ops);

  // links nodes.
  lookup_table_op->LinksFrom({lookup_table_id});
}

void MultiheadMatmulOP::operator()() {
  // Create nodes for multihead_matmul op.
  auto *multihead_matmul = pattern->NewNode(multihead_matmul_repr())
                               ->assert_is_op("multihead_matmul");
  auto *multihead_matmul_out =
      pattern->NewNode(multihead_matmul_out_repr())
          ->assert_is_op_output("multihead_matmul", "Out");

  // links nodes.
  multihead_matmul_out->LinksFrom({multihead_matmul});
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

void SetTransformerInputConvertPass::ApplyImpl(ir::Graph *graph) const {
  bool with_dynamic_shape = Get<bool>("with_dynamic_shape");
  std::string pos_id = Get<std::string>("tensorrt_transformer_posid");

  if (!(graph->Has(framework::ir::kMultiheadMatmulPass) && with_dynamic_shape &&
        (!pos_id.empty()))) {
    VLOG(3) << "Transformer model need MultiheadMatmul, and "
               "with_dynamic_shape. Stop this pass, "
               "please reconfig.";
    return;
  }
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init(name_scope_, graph);
  int found_subgraph_count = 0;
  Node *transformer_input_convert_out0_node;
  Node *transformer_input_convert_out1_node;
  GraphPatternDetector gpd0;
  patterns::SetTransformerInputConvert fused_pattern(
      gpd0.mutable_pattern(), "transformer_input_convert_pass");
  fused_pattern(pos_id);
  auto handler0 = [&](const GraphPatternDetector::subgraph_t &subgraph,
                      Graph *graph) {
    VLOG(3)
        << "transformer_input_convert_pass for pos_id, max_seqlen, mask_tensor";
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table, lookup_table, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table_id, lookup_table_id, fused_pattern);

    // create op, var in graph
    OpDesc new_desc(lookup_table->Op()->Block());

    new_desc.SetType("transformer_input_convert");

    // inputs
    new_desc.SetInput("Input", {lookup_table_id->Name()});

    // outputs
    std::string transformer_input_convert_out0_name = "pos_id_tensor";
    std::string transformer_input_convert_out1_name = "max_seqlen_tensor";
    std::string transformer_input_convert_out2_name = "mask_tensor";
    std::vector<std::string> output_0 = {transformer_input_convert_out0_name};
    std::vector<std::string> output_1 = {transformer_input_convert_out1_name};
    std::vector<std::string> output_2 = {transformer_input_convert_out2_name};
    new_desc.SetOutput("PosId", output_0);
    new_desc.SetOutput("MaxSeqlen", output_1);
    new_desc.SetOutput("MaskTensor", output_2);

    auto *transformer_input_convert_out0 =
        lookup_table->Op()->Block()->Var(transformer_input_convert_out0_name);
    auto *transformer_input_convert_out1 =
        lookup_table->Op()->Block()->Var(transformer_input_convert_out1_name);
    auto *transformer_input_convert_out2 =
        lookup_table->Op()->Block()->Var(transformer_input_convert_out2_name);
    transformer_input_convert_out0->SetDataType(proto::VarType::INT32);
    transformer_input_convert_out1->SetDataType(proto::VarType::INT32);
    transformer_input_convert_out2->SetDataType(proto::VarType::INT32);
    transformer_input_convert_out0->SetShape({-1});
    transformer_input_convert_out1->SetShape({-1});

    transformer_input_convert_out2->SetShape({-1});

    transformer_input_convert_out0->SetPersistable(false);
    transformer_input_convert_out1->SetPersistable(false);
    transformer_input_convert_out2->SetPersistable(false);

    auto new_op_node = graph->CreateOpNode(&new_desc);
    auto transformer_input_convert_out0_node =
        graph->CreateVarNode(transformer_input_convert_out0);
    auto transformer_input_convert_out1_node =
        graph->CreateVarNode(transformer_input_convert_out1);
    auto transformer_input_convert_out2_node =
        graph->CreateVarNode(transformer_input_convert_out2);

    // needn't create variable in scope

    IR_NODE_LINK_TO(lookup_table_id, new_op_node);
    IR_NODE_LINK_TO(new_op_node, transformer_input_convert_out0_node);
    IR_NODE_LINK_TO(new_op_node, transformer_input_convert_out1_node);
    IR_NODE_LINK_TO(new_op_node, transformer_input_convert_out2_node);
  };
  gpd0(graph, handler0);

  GraphPatternDetector gpd1;
  patterns::MultiheadMatmulOP multihead_matmul_pattern(
      gpd1.mutable_pattern(), "transformer_input_convert_pass");
  multihead_matmul_pattern();
  auto handler1 = [&](const GraphPatternDetector::subgraph_t &subgraph,
                      Graph *graph) {
    VLOG(3) << "link pos_id, max_seqlen to multihead_matmul.";
    GET_IR_NODE_FROM_SUBGRAPH(
        multihead_matmul, multihead_matmul, multihead_matmul_pattern);

    IR_NODE_LINK_TO(transformer_input_convert_out0_node, multihead_matmul);
    IR_NODE_LINK_TO(transformer_input_convert_out1_node, multihead_matmul);
  };
  gpd1(graph, handler1);

  found_subgraph_count++;
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(set_transformer_input_convert_pass,
              paddle::framework::ir::SetTransformerInputConvertPass);
