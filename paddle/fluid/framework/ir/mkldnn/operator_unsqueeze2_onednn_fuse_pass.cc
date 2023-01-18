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

#include "paddle/fluid/framework/ir/mkldnn/operator_unsqueeze2_onednn_fuse_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void FuseOperatorUnsqueeze2OneDNNPass::ApplyImpl(Graph *graph) const {
  std::vector<std::pair<std::string, int>> ops_and_outputs = {
      {"transpose2", 2}, {"elementwise_mul", 1}};

  for (const auto &op_and_outputs : ops_and_outputs)
    FuseUnsqueeze2(graph, op_and_outputs.first, op_and_outputs.second);
}

void FuseOperatorUnsqueeze2OneDNNPass::FuseUnsqueeze2(
    Graph *graph, const std::string &op_type, int num_of_outputs) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(op_type + "_unsqueeze2_onednn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::OperatorUnsqueeze2 op_unsqueeze2_pattern(
      gpd.mutable_pattern(), op_type + "_unsqueeze2_onednn_fuse_pass");
  op_unsqueeze2_pattern(op_type, num_of_outputs);

  int found_operator_unsqueeze2_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    GET_IR_NODE_FROM_SUBGRAPH(operator_op, preceding_op, op_unsqueeze2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        operator_out, preceding_op_out, op_unsqueeze2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        unsqueeze2_op, unsqueeze2_op, op_unsqueeze2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        unsqueeze2_out, unsqueeze2_out, op_unsqueeze2_pattern);

    if (!operator_op->Op()->HasAttr("use_mkldnn") ||
        (operator_op->Op()->HasAttr("use_mkldnn") &&
         !(PADDLE_GET_CONST(bool, operator_op->Op()->GetAttr("use_mkldnn"))))) {
      VLOG(4) << "Only oneDNN version of " << op_type
              << "can be fused with unsqueeze2.";
      return;
    }

    std::vector<int> unsqueeze2_axes = PADDLE_GET_CONST(
        std::vector<int>, unsqueeze2_op->Op()->GetAttr("axes"));

    auto const &names = unsqueeze2_op->Op()->InputNames();

    bool has_axes_tensor =
        std::find(names.begin(), names.end(), "AxesTensor") != names.end();
    bool has_axes_tensor_list =
        std::find(names.begin(), names.end(), "AxesTensorList") != names.end();

    if (has_axes_tensor &&
        unsqueeze2_op->Op()->Input("AxesTensor").size() > 0) {
      VLOG(4) << "Cannot fuse " << op_type
              << " and unsqueeze2 because unsqueeze2 dims are specified by "
                 "AxesTensor!";
      return;
    }

    if (has_axes_tensor_list &&
        unsqueeze2_op->Op()->Input("AxesTensorList").size() > 0) {
      VLOG(4) << "Cannot fuse " << op_type
              << " and unsqueeze2 because unsqueeze2 dims are specified by "
                 "AxesTensorList!";
      return;
    }

    operator_op->Op()->SetAttr("fused_unsqueeze2_axes", unsqueeze2_axes);
    operator_op->Op()->SetOutput("Out", {unsqueeze2_out->Name()});

    IR_OP_VAR_LINK(operator_op, unsqueeze2_out);
    GraphSafeRemoveNodes(g, {unsqueeze2_op, operator_out});
    found_operator_unsqueeze2_count++;
  };

  gpd(graph, handler);
  AddStatis(found_operator_unsqueeze2_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      found_operator_unsqueeze2_count > 0)
    PrettyLogDetail("---    fused %d %s with unsqueeze2",
                    found_operator_unsqueeze2_count,
                    op_type);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(operator_unsqueeze2_onednn_fuse_pass,
              paddle::framework::ir::FuseOperatorUnsqueeze2OneDNNPass);
REGISTER_PASS_CAPABILITY(operator_unsqueeze2_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("unsqueeze2", 0)
            .GE("transpose2", 0));
