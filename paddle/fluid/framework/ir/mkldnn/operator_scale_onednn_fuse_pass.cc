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

#include "paddle/fluid/framework/ir/mkldnn/operator_scale_onednn_fuse_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void FuseOperatorScaleOneDNNPass::ApplyImpl(Graph *graph) const {
  const std::vector<std::string> fusable_ops{
      "fc",
      "matmul",
      "matmul_v2",
      "elementwise_add",
      "elementwise_sub",
      "elementwise_mul",
      "elementwise_div",
  };
  for (const auto &op : fusable_ops) FuseScale(graph, op);
}

void FuseOperatorScaleOneDNNPass::FuseScale(Graph *graph,
                                            const std::string &op_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(op_type + "_scale_onednn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::OperatorActivation op_scale_pattern(
      gpd.mutable_pattern(), op_type + "_scale_onednn_fuse_pass");
  op_scale_pattern(op_type, "scale");

  int found_operator_scale_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    GET_IR_NODE_FROM_SUBGRAPH(operator_op, preceding_op, op_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(operator_out, preceding_op_out, op_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_op, activation, op_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, activation_out, op_scale_pattern);

    if (operator_op->Op()->HasAttr("use_mkldnn") &&
        !(PADDLE_GET_CONST(bool, operator_op->Op()->GetAttr("use_mkldnn")))) {
      VLOG(4) << "Only oneDNN version of " << op_type
              << "can be fused with scale.";
      return;
    }

    if (scale_op->Op()->GetAttrIfExists<float>("bias") != 0.0) {
      VLOG(4) << op_type << " can be fused only with unbiased scale.";
      return;
    }

    float scale = PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("scale"));

    auto *scope = param_scope();
    auto const &names = scale_op->Op()->InputNames();
    bool has_scale_tensor =
        std::find(names.begin(), names.end(), "ScaleTensor") != names.end();

    if (has_scale_tensor && scale_op->Op()->Input("ScaleTensor").size() > 0) {
      std::string scale_var_name = scale_op->Op()->Input("ScaleTensor").front();
      auto *scale_var = scope->FindVar(scale_var_name);
      // ScaleTensor must be weight
      if (scale_var == nullptr) return;
      auto *scale_tensor = scale_var->GetMutable<phi::DenseTensor>();
      scale = *(scale_tensor->data<float>());
    }

    operator_op->Op()->SetAttr("fused_output_scale", scale);
    operator_op->Op()->SetOutput("Out", {scale_out->Name()});

    IR_OP_VAR_LINK(operator_op, scale_out);
    GraphSafeRemoveNodes(g, {scale_op, operator_out});
    found_operator_scale_count++;
  };

  gpd(graph, handler);
  AddStatis(found_operator_scale_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      found_operator_scale_count > 0)
    PrettyLogDetail(
        "---    fused %d %s with scale", found_operator_scale_count, op_type);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(operator_scale_onednn_fuse_pass,
              paddle::framework::ir::FuseOperatorScaleOneDNNPass);
REGISTER_PASS_CAPABILITY(operator_scale_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("fc", 0)
            .LE("matmul", 1)
            .EQ("matmul_v2", 0)
            .LE("elementwise_add", 1)
            .LE("elementwise_sub", 1)
            .LE("elementwise_mul", 1)
            .LE("elementwise_div", 1)
            .EQ("scale", 0));
