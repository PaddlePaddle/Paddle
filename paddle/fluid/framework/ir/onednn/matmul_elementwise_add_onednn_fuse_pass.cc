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

#include "paddle/fluid/framework/ir/onednn/matmul_elementwise_add_onednn_fuse_pass.h"

#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/onednn/onednn_pass_util.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle::framework::ir {

using string::PrettyLogDetail;

void MatmulElementwiseAddMKLDNNFusePass::ApplyImpl(Graph* graph) const {
  auto matmul_types = {"fused_matmul", "matmul", "matmul_v2"};
  auto matmul_as_x = {true, false};

  for (const auto& matmul_type : matmul_types)
    for (const auto& as_x : matmul_as_x) {
      FuseMatmulElementwiseAdd(graph, matmul_type, as_x);
    }
}

void MatmulElementwiseAddMKLDNNFusePass::FuseMatmulElementwiseAdd(
    Graph* graph, const std::string& matmul_type, bool matmul_as_x) const {
  const std::string fusion_mode = matmul_as_x ? "x" : "y";
  const auto name_scope = matmul_type + "_elementwise_add_as_" + fusion_mode;
  FusePassBase::Init(name_scope, graph);
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::MatmulElementwiseAdd matmul_pattern(
      pattern, name_scope, matmul_type, matmul_as_x);
  matmul_pattern(matmul_type, matmul_as_x);

  int found_matmul_elementwise_add_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(matmul, matmul_op, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_add, elementwise_add_op, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_addend, elementwise_addend, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_add_out, elementwise_add_out, matmul_pattern);

    if (FindFuseOption(*matmul, *elementwise_add) != FUSE_MKLDNN) return;
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "op compat for matmul_elementwise_add_onednn_fuse_pass failed.";
      return;
    }

    ConvertToFusedOp(matmul->Op());
    matmul->Op()->SetInput("ResidualData", {elementwise_addend->Name()});
    matmul->Op()->SetOutput("Out", {elementwise_add_out->Name()});

    GraphSafeRemoveNodes(g, {matmul_out, elementwise_add});

    IR_NODE_LINK_TO(elementwise_addend, matmul);
    IR_NODE_LINK_TO(matmul, elementwise_add_out);

    found_matmul_elementwise_add_count++;
  };

  gpd(graph, handler);
  AddStatis(found_matmul_elementwise_add_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      (found_matmul_elementwise_add_count > 0)) {
    PrettyLogDetail("---    fused %d %s (as %s) with elementwise_add",
                    found_matmul_elementwise_add_count,
                    matmul_type,
                    fusion_mode);
  }
}

MatmulElementwiseAddMKLDNNFusePass::MatmulElementwiseAddMKLDNNFusePass() {
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
      .IsType<float>()
      .End()
      .AddAttr("transpose_X")
      .IsType<bool>()
      .End()
      .AddAttr("transpose_Y")
      .IsType<bool>()
      .End();

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

  AddOpCompat(OpCompat("fused_matmul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddInput("ResidualData")
      .IsTensor()
      .IsOptional()
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

}  // namespace paddle::framework::ir

REGISTER_PASS(matmul_elementwise_add_onednn_fuse_pass,
              paddle::framework::ir::MatmulElementwiseAddMKLDNNFusePass);
REGISTER_PASS_CAPABILITY(matmul_elementwise_add_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("fused_matmul", 0)
            .LE("matmul", 1)
            .EQ("matmul_v2", 0)
            .LE("elementwise_add", 1));
