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

#include "paddle/fluid/framework/ir/onednn/reshape_transpose_matmul_onednn_fuse_pass.h"
#include "paddle/fluid/framework/ir/onednn/onednn_pass_util.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle::framework::ir {

void ReshapeTransposeMatmulMkldnnFusePass::ApplyImpl(Graph *graph) const {
  auto matmul_types = {"matmul", "matmul_v2", "fused_matmul"};

  for (const auto &matmul_type : matmul_types) {
    Fuse(graph,
         matmul_type,
         false /*with_reshape_xshape*/,
         false /*with_transpose_xshape*/);
    Fuse(graph,
         matmul_type,
         false /*with_reshape_xshape*/,
         true /*with_transpose_xshape*/);
    Fuse(graph,
         matmul_type,
         true /*with_reshape_xshape*/,
         false /*with_transpose_xshape*/);
    Fuse(graph,
         matmul_type,
         true /*with_reshape_xshape*/,
         true /*with_transpose_xshape*/);
  }
}

void ReshapeTransposeMatmulMkldnnFusePass::Fuse(
    Graph *graph,
    const std::string &matmul_type,
    bool with_reshape_xshape,
    bool with_transpose_xshape) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          common::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init("reshape_transpose_" + matmul_type + "_onednn_fuse_pass",
                     graph);

  GraphPatternDetector gpd;
  patterns::ReshapeTransposeMatmulPattern rtm_pattern(
      gpd.mutable_pattern(),
      "reshape_transpose_" + matmul_type + "_onednn_fuse_pass");

  rtm_pattern(matmul_type, with_reshape_xshape, with_transpose_xshape);

  int found_reshape_transpose_matmul_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Op compatible check in reshape_transpose_" << matmul_type
                   << "_onednn_fuse_pass failed.";
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(reshape_in, reshape_in, rtm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_op, reshape_op, rtm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape_out, rtm_pattern);
    ir::Node *reshape_xshape{nullptr};
    if (with_reshape_xshape) {
      GET_IR_NODE_FROM_SUBGRAPH(reshape_xshape1, reshape_xshape, rtm_pattern);
      reshape_xshape = reshape_xshape1;
    }
    GET_IR_NODE_FROM_SUBGRAPH(transpose_op, transpose_op, rtm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose_out, rtm_pattern);
    ir::Node *transpose_xshape{nullptr};
    if (with_transpose_xshape) {
      GET_IR_NODE_FROM_SUBGRAPH(
          transpose_xshape1, transpose_xshape, rtm_pattern);
      transpose_xshape = transpose_xshape1;
    }
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, rtm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, rtm_pattern);

    OpDesc *matmul_desc = matmul_op->Op();
    std::string input_var_name = transpose_out->Name();
    std::string matmul_input_name;
    if (matmul_desc->Inputs().at("X").at(0) == input_var_name) {
      matmul_input_name = "X";
    } else if (matmul_desc->Inputs().at("Y").at(0) == input_var_name) {
      matmul_input_name = "Y";
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unexpected input to %s encountered.", matmul_type));
    }

    // Return if input of fused_matmul is already fused
    if (matmul_type == "fused_matmul") {
      auto is_already_fused_X =
          matmul_desc->HasAttr("fused_reshape_X")
              ? !(PADDLE_GET_CONST(std::vector<int>,
                                   matmul_desc->GetAttr("fused_reshape_X"))
                      .empty())
              : false;
      if (is_already_fused_X && matmul_input_name == "X") return;

      auto is_already_fused_Y =
          matmul_desc->HasAttr("fused_reshape_Y")
              ? !(PADDLE_GET_CONST(std::vector<int>,
                                   matmul_desc->GetAttr("fused_reshape_Y"))
                      .empty())
              : false;
      if (is_already_fused_Y && matmul_input_name == "Y") return;
    }

    auto reshape_shape =
        paddle::get<std::vector<int>>(reshape_op->Op()->GetAttr("shape"));
    auto transpose_axis =
        paddle::get<std::vector<int>>(transpose_op->Op()->GetAttr("axis"));

    if (reshape_shape.size() < 2 || reshape_shape.size() > 4) {
      VLOG(3) << "shape_" + matmul_input_name + " attribute of " + matmul_type +
                     " was implemented for 2, 3 or 4 dimensions.";
      return;
    }
    if (reshape_shape.size() != transpose_axis.size()) {
      VLOG(3) << "Ranks of shape_" + matmul_input_name + " and axis_" +
                     matmul_input_name + "attributes of " + matmul_type +
                     " must be equal.";
      return;
    }
    if (std::count(reshape_shape.begin(), reshape_shape.end(), -1) > 1) {
      VLOG(3) << "Only one dim can be undefined / marked as '-1'";
      return;
    }

    ConvertToFusedOp(matmul_desc);
    matmul_desc->SetInput(matmul_input_name, {(reshape_in)->Name()});
    matmul_desc->SetAttr("fused_reshape_" + matmul_input_name, reshape_shape);
    matmul_desc->SetAttr("fused_transpose_" + matmul_input_name,
                         transpose_axis);

    std::unordered_set<const ir::Node *> nodes_to_remove{
        reshape_op, reshape_out, transpose_op, transpose_out};
    if (with_reshape_xshape) nodes_to_remove.insert(reshape_xshape);
    if (with_transpose_xshape) nodes_to_remove.insert(transpose_xshape);
    GraphSafeRemoveNodes(graph, nodes_to_remove);

    IR_NODE_LINK_TO(reshape_in, matmul_op);

    ++found_reshape_transpose_matmul_count;
  };

  gpd(graph, handler);
  AddStatis(found_reshape_transpose_matmul_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      found_reshape_transpose_matmul_count > 0) {
    std::stringstream msg_ss;
    msg_ss << "---    fused " << found_reshape_transpose_matmul_count
           << " reshape + transpose + " << matmul_type;
    if (with_reshape_xshape) msg_ss << " with reshape's xshape";
    if (with_transpose_xshape) msg_ss << " with transpose's xshape";
    string::PrettyLogDetail(msg_ss.str().c_str());
  }
}

ReshapeTransposeMatmulMkldnnFusePass::ReshapeTransposeMatmulMkldnnFusePass() {
  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      // The reshape2 op for this pass should not have "Shape" and "ShapeTensor"
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("shape")
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int>>()
      .End();

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
}

}  // namespace paddle::framework::ir

REGISTER_PASS(reshape_transpose_matmul_onednn_fuse_pass,
              paddle::framework::ir::ReshapeTransposeMatmulMkldnnFusePass);

REGISTER_PASS_CAPABILITY(reshape_transpose_matmul_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("fused_matmul", 0)
            .EQ("matmul", 1)
            .EQ("matmul_v2", 0));
