// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/reshape_transpose_matmul_mkldnn_fuse_pass.h"
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

ReshapeTransposeMatmulMkldnnFusePass::ReshapeTransposeMatmulMkldnnFusePass() {
  op_name_ = "matmul";

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

  AddOpCompat(OpCompat(op_name_))
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
}

void ReshapeTransposeMatmulMkldnnFusePass::Fuse(
    Graph *graph, bool with_reshape_xshape, bool with_transpose_xshape) const {
  GraphPatternDetector gpd;
  patterns::ReshapeTransposeMatmulPattern rtm_pattern(gpd.mutable_pattern(),
                                                      name_scope_);

  rtm_pattern(op_name_, with_reshape_xshape, with_transpose_xshape);

  int found_reshape_transpose_matmul_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Op compatible check in reshape_transpose_" << op_name_
                   << "_mkldnn_fuse_pass failed.";
      return;
    }
    VLOG(4) << "handle reshape_transpose_" << op_name_ << " fuse";
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
      GET_IR_NODE_FROM_SUBGRAPH(transpose_xshape1, transpose_xshape,
                                rtm_pattern);
      transpose_xshape = transpose_xshape1;
    }
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, rtm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, rtm_pattern);

    auto reshape_shape =
        boost::get<std::vector<int>>(reshape_op->Op()->GetAttr("shape"));
    auto transpose_axis =
        boost::get<std::vector<int>>(transpose_op->Op()->GetAttr("axis"));

    OpDesc *matmul_desc = matmul_op->Op();
    std::string input_var_name = transpose_out->Name();

    auto UpdateMatmul = [&](std::string matmul_input_name) {
      matmul_desc->SetInput(matmul_input_name, {(reshape_in)->Name()});
      matmul_desc->SetAttr("fused_reshape_" + matmul_input_name, reshape_shape);
      matmul_desc->SetAttr("fused_transpose_" + matmul_input_name,
                           transpose_axis);
    };
    if (matmul_desc->Inputs().at("X").at(0) == input_var_name) {
      UpdateMatmul("X");
    } else if (matmul_desc->Inputs().at("Y").at(0) == input_var_name) {
      UpdateMatmul("Y");
    } else {
      throw platform::errors::InvalidArgument("Unexpected input to " +
                                              op_name_ + " encountered.");
    }

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
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    std::stringstream msg_ss;
    msg_ss << "---    Fused " << found_reshape_transpose_matmul_count
           << " ReshapeTransposeMatmul patterns for " << op_name_ << " Op";
    if (with_reshape_xshape) msg_ss << " with reshape's xshape";
    if (with_transpose_xshape) msg_ss << " with transpose's xshape";
    string::PrettyLogDetail(msg_ss.str().c_str());
  }
}

void ReshapeTransposeMatmulMkldnnFusePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init(name_scope_, graph);

  Fuse(graph, false, false);
  Fuse(graph, false, true);
  Fuse(graph, true, false);
  Fuse(graph, true, true);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(reshape_transpose_matmul_mkldnn_fuse_pass,
              paddle::framework::ir::ReshapeTransposeMatmulMkldnnFusePass);

REGISTER_PASS_CAPABILITY(reshape_transpose_matmul_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "matmul", 1));
