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

#include "paddle/fluid/framework/ir/dense_fc_to_sparse_pass.h"

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir::patterns {

PDNode *patterns::DenseFC::operator()() {
  auto *fc = pattern->NewNode(fc_repr())->assert_is_op("fc");
  // Input
  auto *fc_input = pattern->NewNode(fc_input_repr())
                       ->AsInput()
                       ->assert_is_op_input("fc", "Input");
  // Filter
  auto *fc_weights = pattern->NewNode(fc_weights_repr())
                         ->AsInput()
                         ->assert_is_op_input("fc", "W");
  // Bias
  auto *fc_bias = pattern->NewNode(fc_bias_repr())
                      ->AsInput()
                      ->assert_is_op_input("fc", "Bias");
  // Output
  auto *fc_out = pattern->NewNode(fc_out_repr())
                     ->AsOutput()
                     ->assert_is_op_output("fc", "Out")
                     ->assert_is_only_output_of_op("fc");

  fc->LinksFrom({fc_input, fc_weights, fc_bias}).LinksTo({fc_out});

  return fc_out;
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

DenseFCToSparsePass::DenseFCToSparsePass() {
  AddOpCompat(OpCompat("fc"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}

void DenseFCToSparsePass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));

  std::string name_scope = "dense_fc_to_sparse_pass";
  FusePassBase::Init(name_scope, graph);
  GraphPatternDetector gpd;

  patterns::DenseFC dense_fc_pattern(gpd.mutable_pattern(),
                                     "dense_fc_replace_pass");
  dense_fc_pattern();
  int found_dense_fc_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Replace dense fc with sparse_fc.";

    /*   if (!IsCompat(subgraph, g)) {
         LOG(WARNING) << "Pass in op compat failed.";
         return;
       }*/

    GET_IR_NODE_FROM_SUBGRAPH(fc_out, fc_out, dense_fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc, fc, dense_fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_input, fc_input, dense_fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_weights, fc_weights, dense_fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_bias, fc_bias, dense_fc_pattern);

    auto *fc_op = fc->Op();
    auto w_name = fc_op->Input("W")[0];
    // recognize sparse op by name
    if (w_name.find("sparse_2_4") != w_name.npos) {
      // fake op
      OpDesc desc(fc_op->Block());
      desc.SetType("sparse_fc");
      desc.SetInput("Input", {fc_input->Name()});
      desc.SetInput("W", {fc_weights->Name()});
      desc.SetInput("Bias", {fc_bias->Name()});
      desc.SetOutput("Out", {fc_out->Name()});

      // copy all attr
      if (fc_op->HasAttr("x_num_col_dims")) {
        desc.SetAttr("x_num_col_dims", fc_op->GetAttr("x_num_col_dims"));
      }
      if (fc_op->HasAttr("in_num_col_dims")) {
        desc.SetAttr("in_num_col_dims", fc_op->GetAttr("in_num_col_dims"));
      }
      desc.SetAttr("activation_type", fc_op->GetAttr("activation_type"));
      if (fc_op->HasAttr("enable_int8")) {
        desc.SetAttr("enable_int8", fc_op->GetAttr("enable_int8"));
      }
      if (fc_op->HasAttr("Input_scale")) {
        desc.SetAttr("Input_scale", fc_op->GetAttr("Input_scale"));
      }
      if (fc_op->HasAttr("support_int8")) {
        desc.SetAttr("support_int8", fc_op->GetAttr("support_int8"));
      }
      if (fc_op->HasAttr("out_threshold")) {
        desc.SetAttr("out_threshold", fc_op->GetAttr("out_threshold"));
      }
      desc.Flush();
      GraphSafeRemoveNodes(g, {fc});
      auto sparse_fc_node = g->CreateOpNode(&desc);

      IR_NODE_LINK_TO(fc_input, sparse_fc_node);
      IR_NODE_LINK_TO(fc_weights, sparse_fc_node);
      IR_NODE_LINK_TO(fc_bias, sparse_fc_node);
      IR_NODE_LINK_TO(sparse_fc_node, fc_out);
      found_dense_fc_count++;
    }
  };

  gpd(graph, handler);
  AddStatis(found_dense_fc_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(dense_fc_to_sparse_pass,
              paddle::framework::ir::DenseFCToSparsePass);
