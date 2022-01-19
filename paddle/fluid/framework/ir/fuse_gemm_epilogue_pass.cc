// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_gemm_epilogue_pass.h"
#include <string>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void FuseGemmEpiloguePass::ApplyImpl(ir::Graph *graph) const {
  std::unordered_set<std::string> act_types = {"relu", "gelu"};
  graph = FuseLinearActFwd(graph, act_types, false);
  graph = FuseLinearActFwd(graph, act_types, true);
  graph = FuseLinearFwd(graph, false);
  graph = FuseLinearFwd(graph, true);
}

ir::Graph *FuseGemmEpiloguePass::FuseLinearFwd(ir::Graph *graph,
                                               bool is_training) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("gemm_epilogue", graph);

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("gemm_epilogue/x")
                ->AsInput()
                ->assert_is_op_input("matmul_v2", "X");
  patterns::LinearAct linear_act_pattern(gpd.mutable_pattern(), "linear_act");

  linear_act_pattern(x, {}, is_training);

  int found_linear_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle LinearAct fuse";

    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_w, matmul_w, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add_op, ele_add, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_bias, ele_bias, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_out, elewise_add_out, linear_act_pattern);

    std::vector<int64_t> matmul_x_shape = subgraph.at(x)->Var()->GetShape();
    std::vector<int64_t> matmul_w_shape = matmul_w->Var()->GetShape();

    // Note (Ming Huang): We only support matmul_v2 from paddle.nn.Linear
    // currently. The conditions below are used to verify wether matmul_v2
    // is created by paddle.nn.Linear
    auto matmul_op_desc = matmul_op->Op();
    if (!IsGemmFromLinear_(matmul_x_shape, matmul_w_shape, matmul_op_desc))
      return;

    OpDesc gemm_epilogue_op_desc(matmul_op->Op()->Block());
    std::string activation = "none";
    gemm_epilogue_op_desc.SetType("fused_gemm_epilogue");
    gemm_epilogue_op_desc.SetInput("X", {subgraph.at(x)->Name()});
    gemm_epilogue_op_desc.SetInput("Y", {matmul_w->Name()});
    gemm_epilogue_op_desc.SetInput("bias", {ele_bias->Name()});
    gemm_epilogue_op_desc.SetOutput("out", {ele_out->Name()});
    gemm_epilogue_op_desc.SetAttr("activation", activation);
    gemm_epilogue_op_desc.SetAttr("op_role",
                                  matmul_op_desc->GetAttr("op_role"));
    auto gemm_epilogue_node = g->CreateOpNode(&gemm_epilogue_op_desc);

    IR_NODE_LINK_TO(subgraph.at(x), gemm_epilogue_node);
    IR_NODE_LINK_TO(matmul_w, gemm_epilogue_node);
    IR_NODE_LINK_TO(ele_bias, gemm_epilogue_node);
    IR_NODE_LINK_TO(gemm_epilogue_node, ele_out);

    GraphSafeRemoveNodes(g, {matmul_op, matmul_out, ele_add_op});

    VLOG(4) << "\n\t " << subgraph.at(x)->Name() << " and " << matmul_w->Name()
            << " -> " << matmul_op->Name() << " -> " << matmul_out->Name()
            << "\n\t " << matmul_out->Name() << " and " << ele_bias->Name()
            << " -> " << ele_add_op->Name() << " -> " << ele_out->Name()
            << "\n\t " << ele_out->Name();
    found_linear_count++;
  };

  gpd(graph, handler);

  AddStatis(found_linear_count);
  return graph;
}

ir::Graph *FuseGemmEpiloguePass::FuseLinearActFwd(
    ir::Graph *graph, const std::unordered_set<std::string> &act_types,
    bool is_training) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("gemm_epilogue", graph);

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("gemm_epilogue/x")
                ->AsInput()
                ->assert_is_op_input("matmul_v2", "X");
  patterns::LinearAct linear_act_pattern(gpd.mutable_pattern(), "linear_act");

  linear_act_pattern(x, act_types, is_training);

  int found_linear_act_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle LinearAct fuse";

    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_w, matmul_w, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add_op, ele_add, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_bias, ele_bias, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_out, elewise_add_out, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_op, act, linear_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, linear_act_pattern);

    std::vector<int64_t> matmul_x_shape = subgraph.at(x)->Var()->GetShape();
    std::vector<int64_t> matmul_w_shape = matmul_w->Var()->GetShape();

    // Note (Ming Huang): We only support matmul_v2 from paddle.nn.Linear
    // currently. The conditions below are used to verify wether matmul_v2
    // is created by paddle.nn.Linear
    auto matmul_op_desc = matmul_op->Op();
    if (!IsGemmFromLinear_(matmul_x_shape, matmul_w_shape, matmul_op_desc))
      return;

    // Only need to check weight.shape[1] for auxiliary pointer
    // and mark it the act op is fused for backward epilogue fusion.
    // That because cuBlasLt epilogue's restriction.
    auto activation = act_op->Op()->Type();
    if (is_training) {
      int divisor_of_n = activation == "relu" ? 128 : 8;
      if (matmul_w_shape[1] % divisor_of_n) return;
      EpiloguePassActivationCache::Instance().InsertFusedActivation(
          act_out->Var()->Name());
    }

    OpDesc gemm_epilogue_op_desc(matmul_op->Op()->Block());
    std::string act_name = "none";
    gemm_epilogue_op_desc.SetType("fused_gemm_epilogue");
    gemm_epilogue_op_desc.SetInput("X", {subgraph.at(x)->Name()});
    gemm_epilogue_op_desc.SetInput("Y", {matmul_w->Name()});
    gemm_epilogue_op_desc.SetInput("bias", {ele_bias->Name()});
    gemm_epilogue_op_desc.SetOutput("out", {act_out->Name()});
    gemm_epilogue_op_desc.SetAttr("activation", activation);
    gemm_epilogue_op_desc.SetAttr("op_role",
                                  matmul_op_desc->GetAttr("op_role"));
    auto gemm_epilogue_node = g->CreateOpNode(&gemm_epilogue_op_desc);

    IR_NODE_LINK_TO(subgraph.at(x), gemm_epilogue_node);
    IR_NODE_LINK_TO(matmul_w, gemm_epilogue_node);
    IR_NODE_LINK_TO(ele_bias, gemm_epilogue_node);
    IR_NODE_LINK_TO(gemm_epilogue_node, act_out);

    GraphSafeRemoveNodes(g,
                         {matmul_op, matmul_out, ele_add_op, ele_out, act_op});

    VLOG(4) << "\n\t " << subgraph.at(x)->Name() << " and " << matmul_w->Name()
            << " -> " << matmul_op->Name() << " -> " << matmul_out->Name()
            << "\n\t " << matmul_out->Name() << " and " << ele_bias->Name()
            << " -> " << ele_add_op->Name() << " -> " << ele_out->Name()
            << "\n\t " << ele_out->Name() << " -> " << act_op->Name() << " -> "
            << act_out->Name();
    found_linear_act_count++;
  };

  gpd(graph, handler);

  AddStatis(found_linear_act_count);
  return graph;
}

bool FuseGemmEpiloguePass::IsGemmFromLinear_(std::vector<int64_t> x_shape,
                                             std::vector<int64_t> w_shape,
                                             OpDesc *matmul_v2_op) const {
  if (w_shape.size() != 2 || x_shape.size() < 2) return false;
  for (auto attr_name :
       {"fused_reshape_Out", "fused_reshape_X", "fused_reshape_Y",
        "fused_transpose_Out", "fused_transpose_X", "fused_transpose_Y"}) {
    if (matmul_v2_op->HasAttr(attr_name)) {
      std::vector<int> tmp_vec =
          BOOST_GET_CONST(std::vector<int>, matmul_v2_op->GetAttr(attr_name));
      if (tmp_vec.size() > 0) return false;
    }
  }
  if (BOOST_GET_CONST(bool, matmul_v2_op->GetAttr("trans_x")) ||
      BOOST_GET_CONST(bool, matmul_v2_op->GetAttr("trans_y")))
    return false;

  return true;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_gemm_epilogue_pass,
              paddle::framework::ir::FuseGemmEpiloguePass);
