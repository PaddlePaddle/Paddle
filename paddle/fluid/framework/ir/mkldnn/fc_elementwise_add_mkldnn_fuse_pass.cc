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

#include "paddle/fluid/framework/ir/mkldnn/fc_elementwise_add_mkldnn_fuse_pass.h"

#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/mkldnn/mkldnn_pass_util.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

FCResidualConnectionMKLDNNFusePass::FCResidualConnectionMKLDNNFusePass() {
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
      .End()
      .AddAttr("in_num_col_dims")
      .IsNumGE(1)
      .End();

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
      .IsIntIn({-1, 0, 1})
      .End();
}

GraphWithStats FCResidualConnectionMKLDNNFusePass::FuseFC(
    const std::string& name_scope,
    const GraphWithStats& graph_with_stats,
    bool fc_as_x) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::FCMKLDNN fc_pattern{pattern, name_scope};
  auto fc_output = fc_pattern(false /* with residual */);

  patterns::ResidualElementwise elementwise_pattern{
      pattern, name_scope, fc_as_x};
  elementwise_pattern(
      fc_output,
      pattern->NewNode(elementwise_pattern.residual_data_repr()),
      "elementwise_add",
      fc_as_x);
  fc_output->AsIntermediate();

  int found_fc_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    LOG(INFO) << "Fuse fc + elementwise_add as residual";
    GetInfoFromTheTmpOp(
        g, "has_quant_info", "var_quant_scales", var_quant_scales_);

    GET_IR_NODE_FROM_SUBGRAPH(fc_op, fc, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_input, input, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_weights, weights, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_output, output, fc_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_op, elementwise_op, elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        residual_data, residual_data, elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_out, elementwise_out, elementwise_pattern);

    if (FindFuseOption(*fc_op, *elementwise_op) != FUSE_MKLDNN) {
      VLOG(4) << "Skipping fusion for " << fc_op->Name() << "(" << fc_op->id()
              << ") with " << elementwise_op->Name() << "("
              << elementwise_op->id()
              << ") because not both ops have use_mkldnn";
      return;
    }
    if (!IsReachable(g, residual_data, fc_output)) {
      VLOG(4) << "Skipping fusion for " << fc_op->Name() << "(" << fc_op->id()
              << ") with " << elementwise_op->Name() << "("
              << elementwise_op->id() << ") because residual input "
              << residual_data->Name() << "(" << residual_data->id()
              << ") is not "
                 "reachable";
      return;
    }
    if (HasFusedActivation(fc_op)) {
      VLOG(4) << "Skipping fusion for " << fc_op->Name() << "(" << fc_op->id()
              << ") with " << elementwise_op->Name() << "("
              << elementwise_op->id() << ") because fc has activation fused";
      return;
    }

    // Binary_add may have some error when scale in int8, thus skip
    // proto::VarType::Type data_type = fc_input->Var()->GetDataType();
    // LOG(INFO) << fc_input->Var()->GetDataType();
    // if (data_type == proto::VarType::INT8 ||
    //     data_type == proto::VarType::UINT8) {
    //   LOG(INFO) << "Skip fusion fc + elementwise_add with int8 data type";
    //   return;
    // }
    // auto& quant_var_scales =
    // Get<std::unordered_map<std::string, std::pair<bool,
    // phi::DenseTensor>>>("quant_var_scales");

    // if (!quant_var_scales.empty()) {
    //   LOG(INFO) << "Skip fusion fc + elementwise_add with quantize data
    //   type"; return;
    // }else{
    //   LOG(INFO) << "It's empty";
    // }
    for (auto node : {fc_input, fc_weights}) {
      if (!var_quant_scales_->empty()) {
        LOG(INFO) << "11";
        if (var_quant_scales_->count(node->Name()) != 0) {
          LOG(INFO) << "It's int8";
          return;
        }
      }
    }

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "op compat for fc_elementwise_add_mkldnn_fuse_pass failed.";
      return;
    }

    fc_op->Op()->SetInput("ResidualData", {residual_data->Name()});
    fc_op->Op()->SetOutput("Out", {elementwise_out->Name()});
    fc_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {fc_output, elementwise_op});

    IR_NODE_LINK_TO(residual_data, fc_op);
    IR_NODE_LINK_TO(fc_op, elementwise_out);

    found_fc_count++;
  };

  gpd(graph_with_stats.first, handler);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      (found_fc_count > 0)) {
    std::stringstream msg_ss;
    std::string fusionMode = fc_as_x ? "x" : "y";
    msg_ss << "---    Fused " << found_fc_count << " fc (as " << fusionMode
           << ") + elementwise_add patterns";
    paddle::string::PrettyLogDetail(msg_ss.str().c_str());
  }

  return std::make_pair(graph_with_stats.first,
                        found_fc_count + graph_with_stats.second);
}

void FCResidualConnectionMKLDNNFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto graph_with_stats = FuseFC(name_scope_, std::make_pair(graph, 0), true);
  graph_with_stats = FuseFC(name_scope_, graph_with_stats, false);

  AddStatis(graph_with_stats.second);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_elementwise_add_mkldnn_fuse_pass,
              paddle::framework::ir::FCResidualConnectionMKLDNNFusePass);
REGISTER_PASS_CAPABILITY(fc_elementwise_add_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("fc", 0)
            .LE("elementwise_add", 1));
