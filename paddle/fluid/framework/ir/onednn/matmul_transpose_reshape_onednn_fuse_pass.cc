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

#include "paddle/fluid/framework/ir/onednn/matmul_transpose_reshape_onednn_fuse_pass.h"
#include "paddle/fluid/framework/ir/onednn/onednn_pass_util.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle::framework::ir {

using string::PrettyLogDetail;

void MatmulTransposeReshapeMKLDNNPass::ApplyImpl(Graph *graph) const {
  auto matmul_types = {"fused_matmul", "matmul", "matmul_v2"};

  for (const auto &matmul_type : matmul_types) {
    Fuse(graph, matmul_type);
  }
}

void MatmulTransposeReshapeMKLDNNPass::Fuse(
    Graph *graph, const std::string &matmul_type) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          common::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init(matmul_type + "_transpose_reshape_onednn_fuse_pass",
                     graph);
  GraphPatternDetector gpd;
  patterns::MatmulTransposeReshapePattern mtrp(
      gpd.mutable_pattern(),
      matmul_type + "_transpose_reshape_onednn_fuse_pass");
  mtrp(matmul_type);

  int found_matmul_transpose_reshape_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_op, transpose_op, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose_out, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out_xshape, transpose_out_xshape, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_op, reshape_op, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape_out, mtrp);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out_xshape, reshape_out_xshape, mtrp);

    auto reshape_shape =
        PADDLE_GET_CONST(std::vector<int>, reshape_op->Op()->GetAttr("shape"));
    auto transpose_axis =
        PADDLE_GET_CONST(std::vector<int>, transpose_op->Op()->GetAttr("axis"));

    const std::vector<int> supported_axis{0, 2, 1, 3};
    if (transpose_axis != supported_axis) {
      VLOG(3) << "do not perform " + matmul_type + "_transpose_reshape fuse: "
              << "supported transpose axis for the fuse are {0, 2, 1, 3}";
      return;
    }
    if (reshape_shape.size() != 3) {
      VLOG(3) << "do not perform " + matmul_type + "_transpose_reshape fuse: "
              << "reshape_out supported rank is 3, received "
              << reshape_shape.size();
      return;
    }
    if (std::count(reshape_shape.begin(), reshape_shape.end(), -1) > 1) {
      VLOG(3) << "Only one dim can be undefined / marked as '-1'";
      return;
    }

    OpDesc *matmul_desc = matmul_op->Op();
    ConvertToFusedOp(matmul_desc);
    matmul_desc->SetOutput("Out", {reshape_out->Name()});
    matmul_desc->SetAttr("fused_reshape_Out", reshape_shape);
    matmul_desc->SetAttr("fused_transpose_Out", transpose_axis);

    GraphSafeRemoveNodes(graph,
                         {matmul_out,
                          transpose_op,
                          transpose_out,
                          reshape_op,
                          transpose_out_xshape,
                          reshape_out_xshape});

    IR_OP_VAR_LINK(matmul_op, reshape_out);

    found_matmul_transpose_reshape_count++;
  };

  gpd(graph, handler);
  AddStatis(found_matmul_transpose_reshape_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      found_matmul_transpose_reshape_count > 0) {
    PrettyLogDetail("---    fused %d %s + transpose + reshape patterns",
                    found_matmul_transpose_reshape_count,
                    matmul_type);
  }
}

MatmulTransposeReshapeMKLDNNPass::MatmulTransposeReshapeMKLDNNPass() {
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

  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      // The reshape2 op for this pass should not have "Shape" and "ShapeTensor"
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsTensor()
      .End()
      .AddAttr("shape")
      .IsType<std::vector<int>>()
      .End();
}

}  // namespace paddle::framework::ir

REGISTER_PASS(matmul_transpose_reshape_onednn_fuse_pass,
              paddle::framework::ir::MatmulTransposeReshapeMKLDNNPass);

REGISTER_PASS_CAPABILITY(matmul_transpose_reshape_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("fused_matmul", 0)
            .LE("matmul", 1)
            .EQ("matmul_v2", 0)
            .EQ("transpose2", 0)
            .EQ("reshape2", 0));
