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
#include "paddle/fluid/framework/ir/mkldnn/fc_scale_mkldnn_fuse_pass.h"
#include <cmath>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

FcScaleMKLDNNFusePass::FcScaleMKLDNNFusePass() {
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

  AddOpCompat(OpCompat("scale"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("scale")
      .IsNumGT(0.0f)
      .End()
      .AddAttr("bias")
      .IsNumEQ(0.0f)
      .End()
      .AddAttr("bias_after_scale")
      .IsOptional()
      .IsType<bool>()
      .End();
}

void FcScaleMKLDNNFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));

  FusePassBase::Init("fc_scale_fuse_pass", graph);
  GraphPatternDetector gpd;
  patterns::FcScale fc_scale_pattern{gpd.mutable_pattern(), "fc_scale"};
  fc_scale_pattern();

  int found_fc_scale_fuse_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(input, input, fc_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(weights, weights, fc_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bias, bias, fc_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc, fc, fc_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_in, scale_in, fc_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_op, scale_op, fc_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, fc_scale_pattern);

    auto* scope = param_scope();
    float bias_val = PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("bias"));
    if (std::abs(bias_val) > 1e-5) return;

    float scale = PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("scale"));

    auto const& names = scale_op->Op()->InputNames();
    bool has_scale_tensor =
        std::find(names.begin(), names.end(), "ScaleTensor") != names.end();

    if (has_scale_tensor && scale_op->Op()->Input("ScaleTensor").size() > 0) {
      std::string scale_var_name = scale_op->Op()->Input("ScaleTensor").front();
      auto* scale_var = scope->FindVar(scale_var_name);
      // ScaleTensor must be weight
      if (scale_var == nullptr) return;
      auto* scale_tensor = scale_var->GetMutable<LoDTensor>();
      scale = *(scale_tensor->data<float>());
    }

    if (scope->FindVar(weights->Name()) == nullptr ||
        scope->FindVar(bias->Name()) == nullptr)
      return;
    auto* weight_tensor =
        scope->FindVar(weights->Name())->GetMutable<LoDTensor>();
    auto* bias_tensor = scope->FindVar(bias->Name())->GetMutable<LoDTensor>();

    auto weights_data =
        weight_tensor->mutable_data<float>(platform::CPUPlace());
    for (auto i = 0; i < weight_tensor->numel(); i++) {
      weights_data[i] *= scale;
    }

    auto bias_data = bias_tensor->mutable_data<float>(platform::CPUPlace());
    for (auto i = 0; i < bias_tensor->numel(); i++) {
      bias_data[i] *= scale;
    }

    // VLOG(0) << "FC scale fuse pass. "<< scale<<" "<<weight_tensor->numel()<<"
    // "<<bias_tensor->numel();

    OpDesc* fc_desc = fc->Op();
    fc_desc->SetOutput("Out", {scale_out->Name()});
    if (!IsCompat(*fc_desc)) {
      LOG(WARNING) << "fc_scale_mkldnn_fuse_pass in out fc op compat failed.";
      return;
    }
    IR_NODE_LINK_TO(fc, scale_out);
    GraphSafeRemoveNodes(graph, {scale_in, scale_op});
    ++found_fc_scale_fuse_count;
  };
  gpd(graph, handler);
  AddStatis(found_fc_scale_fuse_count);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    std::stringstream msg_ss;
    msg_ss << "---    Fused " << found_fc_scale_fuse_count
           << " fc + scale patterns";
    paddle::string::PrettyLogDetail(msg_ss.str().c_str());
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_scale_mkldnn_fuse_pass,
              paddle::framework::ir::FcScaleMKLDNNFusePass);
REGISTER_PASS_CAPABILITY(fc_scale_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("fc", 0)
            .LE("scale", 1));
