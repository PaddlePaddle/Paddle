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

#include "paddle/fluid/framework/ir/mkldnn/layer_norm_onednn_optimization_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void LayerNormOneDNNOptimizationPass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("layer_norm_onednn_optimization_pass", graph);

  GraphPatternDetector gpd;
  patterns::LayerNormShiftScale layer_norm_shift_scale_pattern(
      gpd.mutable_pattern(), "layer_norm_onednn_optimization_pass");
  layer_norm_shift_scale_pattern();

  int found_layer_norm = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_op, layer_norm_op, layer_norm_shift_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_bias, layer_norm_bias, layer_norm_shift_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, layer_norm_shift_scale_pattern);

    if (layer_norm_op->Op()->HasAttr("use_mkldnn") &&
        !(PADDLE_GET_CONST(bool, layer_norm_op->Op()->GetAttr("use_mkldnn")))) {
      VLOG(4) << "Only oneDNN version of layer_norm can be optimized to "
                 "include Bias and Shift in a single tensor.";
      return;
    }

    auto *scope = param_scope();

    auto ln_bias_name = layer_norm_op->Op()->Input("Bias");
    auto ln_scale_name = layer_norm_op->Op()->Input("Scale");

    auto *ln_bias_tensor =
        scope->FindVar(ln_bias_name[0])->GetMutable<phi::DenseTensor>();
    auto *ln_scale_tensor =
        scope->FindVar(ln_scale_name[0])->GetMutable<phi::DenseTensor>();

    const int channels = ln_bias_tensor->dims()[0];

    VarDesc scale_shift_desc(patterns::PDNodeName(
        "layer_norm_onednn_optimization_pass", "ScaleShift"));
    scale_shift_desc.SetShape({channels * 2});
    scale_shift_desc.SetDataType(
        framework::TransToProtoVarType(ln_bias_tensor->dtype()));
    scale_shift_desc.SetPersistable(true);

    auto scale_shift_node = g->CreateVarNode(&scale_shift_desc);
    auto *scale_shift_tensor =
        scope->Var(scale_shift_node->Name())->GetMutable<phi::DenseTensor>();

    scale_shift_tensor->Resize(phi::make_ddim({channels * 2}));

    memcpy(scale_shift_tensor->mutable_data<float>(platform::CPUPlace()),
           ln_scale_tensor->data<float>(),
           channels * sizeof(float));

    memcpy(scale_shift_tensor->data<float>() + channels,
           ln_bias_tensor->data<float>(),
           channels * sizeof(float));

    layer_norm_op->Op()->SetInput("ScaleShift", {scale_shift_node->Name()});

    IR_NODE_LINK_TO(scale_shift_node, layer_norm_op);
    found_layer_norm++;
  };

  gpd(graph, handler);
  AddStatis(found_layer_norm);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      found_layer_norm > 0)
    PrettyLogDetail("---    optimized %d layer_norms by merging Scale and Bias",
                    found_layer_norm);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(layer_norm_onednn_optimization_pass,
              paddle::framework::ir::LayerNormOneDNNOptimizationPass);
REGISTER_PASS_CAPABILITY(layer_norm_onednn_optimization_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().GE(
            "layer_norm", 0));
