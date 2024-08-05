/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/onednn/scale_matmul_fuse_pass.h"

#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/onednn_helper.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle::framework::ir {

class Graph;

using string::PrettyLogDetail;
ScaleMatmulFusePass::ScaleMatmulFusePass() {
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
      .AddAttr("alpha")
      .IsNumGT(0.0f)
      .End()
      .AddAttr("transpose_X")
      .IsType<bool>()
      .End()
      .AddAttr("transpose_Y")
      .IsType<bool>()
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

void ScaleMatmulFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          common::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));

  FusePassBase::Init("scale_matmul_fuse_pass", graph);
  GraphPatternDetector gpd;
  patterns::ScaleMatmul scale_matmul_pattern{gpd.mutable_pattern(),
                                             "scale_matmul"};
  scale_matmul_pattern();

  int found_scale_matmul_fuse_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(scale_in, scale_in, scale_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_op, scale_op, scale_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, scale_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, scale_matmul_pattern);

    if ((scale_out->outputs).size() != 1) {
      return;
    }

    if (scale_op->Op()->GetAttrIfExists<float>("bias") == 0.0) {
      auto matmul_alpha = matmul_op->Op()->GetAttrIfExists<float>("alpha");
      auto scale_scale = scale_op->Op()->GetAttrIfExists<float>("scale");
      PADDLE_ENFORCE_GT(
          matmul_alpha,
          0.0f,
          common::errors::InvalidArgument(
              "Alpha(%f) of matmul op should have positive value.",
              matmul_alpha));
      PADDLE_ENFORCE_GT(scale_scale,
                        0.0f,
                        common::errors::InvalidArgument(
                            "Scale(%f) of scale op should have positive value.",
                            scale_scale));

      std::string matmul_op_input_name =
          FindInputNameByVarName(matmul_op->Op(), scale_out->Name());

      PADDLE_ENFORCE_NE(
          matmul_op_input_name.empty(),
          true,
          common::errors::NotFound("Operator after scale operator(%s) "
                                   "should have scale output as input.",
                                   scale_out->Name()));
      matmul_op->Op()->SetAttr("alpha", matmul_alpha * scale_scale);
      matmul_op->Op()->SetInput(matmul_op_input_name,
                                std::vector<std::string>({scale_in->Name()}));
      IR_NODE_LINK_TO(scale_in, matmul_op);

      if (!IsCompat(*matmul_op->Op())) {
        LOG(WARNING) << "scale_matmul_fuse_pass in out fc op compat failed.";
        return;
      }
      GraphSafeRemoveNodes(graph, {scale_op, scale_out});
      found_scale_matmul_fuse_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_scale_matmul_fuse_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      (found_scale_matmul_fuse_count > 0))
    PrettyLogDetail("---    fused %d scale with matmul",
                    found_scale_matmul_fuse_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(scale_matmul_fuse_pass,
              paddle::framework::ir::ScaleMatmulFusePass);

REGISTER_PASS_CAPABILITY(scale_matmul_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("scale", 0)
            .LE("matmul", 1));
