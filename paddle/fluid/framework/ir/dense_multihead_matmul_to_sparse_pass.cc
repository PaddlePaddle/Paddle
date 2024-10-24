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

#include "paddle/fluid/framework/ir/dense_multihead_matmul_to_sparse_pass.h"

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir::patterns {
PDNode *patterns::DenseMultiheadMatmul::operator()() {
  auto *multihead_matmul = pattern->NewNode(multihead_matmul_repr())
                               ->assert_is_op("multihead_matmul");
  // Input
  auto *multihead_matmul_input =
      pattern->NewNode(multihead_matmul_input_repr())
          ->AsInput()
          ->assert_is_op_input("multihead_matmul", "Input");
  // Filter
  auto *multihead_matmul_weights =
      pattern->NewNode(multihead_matmul_weights_repr())
          ->AsInput()
          ->assert_is_op_input("multihead_matmul", "W");
  // Bias
  auto *multihead_matmul_bias =
      pattern->NewNode(multihead_matmul_bias_repr())
          ->AsInput()
          ->assert_is_op_input("multihead_matmul", "Bias");
  // BiasQK
  auto *multihead_matmul_biasqk =
      pattern->NewNode(multihead_matmul_biasqk_repr())
          ->AsInput()
          ->assert_is_op_input("multihead_matmul", "BiasQK");
  // Output
  auto *multihead_matmul_out =
      pattern->NewNode(multihead_matmul_out_repr())
          ->AsOutput()
          ->assert_is_op_output("multihead_matmul", "Out")
          ->assert_is_only_output_of_op("multihead_matmul");

  multihead_matmul
      ->LinksFrom({multihead_matmul_input,
                   multihead_matmul_weights,
                   multihead_matmul_bias,
                   multihead_matmul_biasqk})
      .LinksTo({multihead_matmul_out});

  return multihead_matmul_out;
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {
DenseMultiheadMatmulToSparsePass::DenseMultiheadMatmulToSparsePass() {
  AddOpCompat(OpCompat("multihead_matmul"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddInput("BiasQK")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}

void DenseMultiheadMatmulToSparsePass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));

  std::string name_scope = "dense_multihead_matmul_to_sparse_pass";
  FusePassBase::Init(name_scope, graph);
  GraphPatternDetector gpd;

  patterns::DenseMultiheadMatmul multihead_matmul_pattern(
      gpd.mutable_pattern(), "dense_multihead_matmul_replace_pass");
  multihead_matmul_pattern();
  int found_multihead_matmul_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Replace dense multihead matmul with sparse multihead matmul.";

    /*   if (!IsCompat(subgraph, g)) {
         LOG(WARNING) << "Pass in op compat failed.";
         return;
       }*/

    GET_IR_NODE_FROM_SUBGRAPH(
        multihead_matmul_out, multihead_matmul_out, multihead_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        multihead_matmul, multihead_matmul, multihead_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(multihead_matmul_input,
                              multihead_matmul_input,
                              multihead_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(multihead_matmul_weights,
                              multihead_matmul_weights,
                              multihead_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        multihead_matmul_bias, multihead_matmul_bias, multihead_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(multihead_matmul_biasqk,
                              multihead_matmul_biasqk,
                              multihead_matmul_pattern);

    auto *multihead_matmul_op = multihead_matmul->Op();
    auto w_name = multihead_matmul_op->Input("W")[0];
    // recognize sparse op by name
    if (w_name.find("sparse_2_4") != w_name.npos) {
      // fake op
      OpDesc desc(multihead_matmul_op->Block());
      desc.SetType("sparse_multihead_matmul");
      desc.SetInput("Input", {multihead_matmul_input->Name()});
      desc.SetInput("W", {multihead_matmul_weights->Name()});
      desc.SetInput("Bias", {multihead_matmul_bias->Name()});
      desc.SetInput("BiasQK", {multihead_matmul_biasqk->Name()});
      desc.SetOutput("Out", {multihead_matmul_out->Name()});

      // copy all attr
      desc.SetAttr("alpha", multihead_matmul_op->GetAttr("alpha"));
      desc.SetAttr("head_number", multihead_matmul_op->GetAttr("head_number"));
      if (multihead_matmul_op->HasAttr("Input_scale")) {
        desc.SetAttr("Input_scale",
                     multihead_matmul_op->GetAttr("Input_scale"));
      }
      if (multihead_matmul_op->HasAttr("fc_out_threshold")) {
        desc.SetAttr("fc_out_threshold",
                     multihead_matmul_op->GetAttr("fc_out_threshold"));
      }
      if (multihead_matmul_op->HasAttr("qkv2context_plugin_int8")) {
        desc.SetAttr("qkv2context_plugin_int8",
                     multihead_matmul_op->GetAttr("qkv2context_plugin_int8"));
      }
      if (multihead_matmul_op->HasAttr("dp_probs")) {
        desc.SetAttr("dp_probs", multihead_matmul_op->GetAttr("dp_probs"));
      }
      if (multihead_matmul_op->HasAttr("out_threshold")) {
        desc.SetAttr("out_threshold",
                     multihead_matmul_op->GetAttr("out_threshold"));
      }
      desc.Flush();
      GraphSafeRemoveNodes(g, {multihead_matmul});
      auto sparse_multihead_matmul_node = g->CreateOpNode(&desc);

      IR_NODE_LINK_TO(multihead_matmul_input, sparse_multihead_matmul_node);
      IR_NODE_LINK_TO(multihead_matmul_weights, sparse_multihead_matmul_node);
      IR_NODE_LINK_TO(multihead_matmul_bias, sparse_multihead_matmul_node);
      IR_NODE_LINK_TO(multihead_matmul_biasqk, sparse_multihead_matmul_node);
      IR_NODE_LINK_TO(sparse_multihead_matmul_node, multihead_matmul_out);
      found_multihead_matmul_count++;
    }
  };

  gpd(graph, handler);
  AddStatis(found_multihead_matmul_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(dense_multihead_matmul_to_sparse_pass,
              paddle::framework::ir::DenseMultiheadMatmulToSparsePass);
