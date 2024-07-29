// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/matmul_scale_fuse_pass.h"

#include <cmath>
#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

class Node;

MatmulScaleFusePass::MatmulScaleFusePass() {
  AddOpCompat(OpCompat("matmul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("transpose_X")
      .IsType<bool>()
      .End()
      .AddAttr("transpose_Y")
      .IsType<bool>()
      .End()
      .AddAttr("alpha")
      .IsType<float>()
      .End();

  AddOpCompat(OpCompat("scale"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("ScaleTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("bias_after_scale")
      .IsType<bool>()
      .End()
      .AddAttr("scale")
      .End()
      .AddAttr("bias")
      .IsNumEQ(0.0f)
      .End();
}

MatmulV2ScaleFusePass::MatmulV2ScaleFusePass() {
  AddOpCompat(OpCompat("matmul_v2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("trans_x")
      .IsType<bool>()
      .End()
      .AddAttr("trans_y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("scale"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("ScaleTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("bias_after_scale")
      .IsType<bool>()
      .End()
      .AddAttr("scale")
      .End()
      .AddAttr("bias")
      .IsNumEQ(0.0f)
      .End();
}

void MatmulScaleFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "matmul_scale_fuse";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::MatmulScale matmul_scale_pattern(gpd.mutable_pattern(), name_scope);
  matmul_scale_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "matmul_scale_fuse pass";
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_x, matmul_in_x, matmul_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_y, matmul_in_y, matmul_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, matmul_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_in_x, scale_in_x, matmul_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_op, scale_op, matmul_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, matmul_scale_pattern);

    auto* scope = param_scope();
    float bias = PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("bias"));
    if (std::abs(bias) > 1e-5) return;
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "matmul_scale_fuse_pass in op compat failed.";
      return;
    }

    float scale = PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("scale"));
    float matmul_alpha =
        PADDLE_GET_CONST(float, matmul_op->Op()->GetAttr("alpha"));
    auto const& names = scale_op->Op()->InputNames();
    bool has_scale_tensor =
        std::find(names.begin(), names.end(), "ScaleTensor") != names.end();
    if (has_scale_tensor && !scale_op->Op()->Input("ScaleTensor").empty()) {
      std::string scale_var_name = scale_op->Op()->Input("ScaleTensor").front();
      auto* scale_var = scope->FindVar(scale_var_name);
      // ScaleTensor must be weight
      if (scale_var == nullptr) return;
      auto* scale_tensor = scale_var->GetMutable<phi::DenseTensor>();
      scale = *(scale_tensor->data<float>());
    }

    OpDesc* matmul_desc = matmul_op->Op();
    matmul_desc->SetAttr("alpha", scale * matmul_alpha);
    matmul_desc->SetOutput("Out", {scale_out->Name()});
    if (!IsCompat(*matmul_desc)) {
      LOG(WARNING) << "matmul_scale_fuse_pass in out mul op compat failed.";
      return;
    }
    IR_NODE_LINK_TO(matmul_op, scale_out);
    GraphSafeRemoveNodes(graph, {scale_in_x, scale_op});
    ++found_count;
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

void MatmulV2ScaleFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "matmul_v2_scale_fuse";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::MatmulV2Scale matmul_v2_scale_pattern(gpd.mutable_pattern(),
                                                  name_scope);
  matmul_v2_scale_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "matmul_v2_scale_fuse pass";
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_v2_in_x, matmul_v2_in_x, matmul_v2_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_v2_in_y, matmul_v2_in_y, matmul_v2_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_v2_op, matmul_v2_op, matmul_v2_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_in_x, scale_in_x, matmul_v2_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_op, scale_op, matmul_v2_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, matmul_v2_scale_pattern);

    auto* scope = param_scope();
    float bias = PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("bias"));
    if (std::abs(bias) > 1e-5) return;
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "matmul_v2_scale_fuse_pass in op compat failed.";
      return;
    }

    float scale = PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("scale"));
    auto const& names = scale_op->Op()->InputNames();
    bool has_scale_tensor =
        std::find(names.begin(), names.end(), "ScaleTensor") != names.end();
    if (has_scale_tensor && !scale_op->Op()->Input("ScaleTensor").empty()) {
      std::string scale_var_name = scale_op->Op()->Input("ScaleTensor").front();
      auto* scale_var = scope->FindVar(scale_var_name);
      // ScaleTensor must be weight
      if (scale_var == nullptr) return;
      auto* scale_tensor = scale_var->GetMutable<phi::DenseTensor>();
      scale = *(scale_tensor->data<float>());
    }

    auto* matmul_y =
        scope->FindVar(matmul_v2_in_y->Name())->GetMutable<phi::DenseTensor>();
    auto y_data = matmul_y->mutable_data<float>(phi::CPUPlace());
    for (int i = 0; i < matmul_y->numel(); ++i) {
      y_data[i] *= scale;
    }

    OpDesc* matmul_v2_desc = matmul_v2_op->Op();
    matmul_v2_desc->SetOutput("Out", {scale_out->Name()});
    if (!IsCompat(*matmul_v2_desc)) {
      LOG(WARNING) << "matmul_v2_scale_fuse_pass in out mul op compat failed.";
      return;
    }
    IR_NODE_LINK_TO(matmul_v2_op, scale_out);
    GraphSafeRemoveNodes(graph, {scale_in_x, scale_op});
    ++found_count;
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(matmul_scale_fuse_pass,
              paddle::framework::ir::MatmulScaleFusePass);
REGISTER_PASS_CAPABILITY(matmul_scale_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("matmul", 1)
            .EQ("scale", 0));

REGISTER_PASS(matmul_v2_scale_fuse_pass,
              paddle::framework::ir::MatmulV2ScaleFusePass);
REGISTER_PASS_CAPABILITY(matmul_v2_scale_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("matmul_v2", 0)
            .EQ("scale", 0));
