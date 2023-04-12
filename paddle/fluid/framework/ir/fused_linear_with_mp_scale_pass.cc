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

#include "paddle/fluid/framework/ir/fused_linear_with_mp_scale_pass.h"

#include <string>
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

static void GetTransposeAttrsFromOp(const OpDesc &op,
                                    bool *trans_x,
                                    bool *trans_y) {
  *trans_x = PADDLE_GET_CONST(bool, op.GetAttr("trans_x"));
  *trans_y = PADDLE_GET_CONST(bool, op.GetAttr("trans_y"));
}

void FuseLinearWithMPScalePass::ApplyImpl(ir::Graph *graph) const {
  // Replace Linear in ColumnParallelLinear with FusedLinear
  graph = FusedLinearFwd(graph);
  graph = FusedLinearBwd(graph);

  // Replace Linear in RowParallelLinear with FusedLinear + MPScale
  graph = FusedLinearWithMpScaleFwd(graph);
  graph = FusedLinearWithMpScaleBwd(graph);
}

ir::Graph *FuseLinearWithMPScalePass::FusedLinearFwd(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("mp_fused_linear");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(scope_name, "x"))
                ->AsInput()
                ->assert_is_op_input("c_identity", "X");
  patterns::FusedLinear fused_linear_pattern(gpd.mutable_pattern(),
                                             "fused_linear");

  fused_linear_pattern(x);

  int found_fused_linear_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle linear in ColumnParallelLinear fuse";

    GET_IR_NODE_FROM_SUBGRAPH(c_identity_op, c_identity, fused_linear_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        c_identity_out, c_identity_out, fused_linear_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul, fused_linear_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_w, matmul_w, fused_linear_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, fused_linear_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add_op, ele_add, fused_linear_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add_bias, ele_add_bias, fused_linear_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add_out, ele_add_out, fused_linear_pattern);

    std::vector<int64_t> matmul_x_shape = subgraph.at(x)->Var()->GetShape();
    std::vector<int64_t> matmul_w_shape = matmul_w->Var()->GetShape();

    // Note (GhostScreaming): We only support matmul_v2 from paddle.nn.Linear
    // currently. The conditions below are used to verify wether matmul_v2
    // is created by paddle.nn.Linear
    auto matmul_op_desc = matmul_op->Op();
    if (!IsGemmFromLinear_(matmul_x_shape, matmul_w_shape, matmul_op_desc))
      return;

    bool trans_x, trans_y;
    GetTransposeAttrsFromOp(*matmul_op_desc, &trans_x, &trans_y);

    OpDesc fused_gemm_epilogue_op_desc(matmul_op->Op()->Block());
    std::string activation = "none";
    fused_gemm_epilogue_op_desc.SetType("fused_gemm_epilogue");
    fused_gemm_epilogue_op_desc.SetInput("X", {c_identity_out->Name()});
    fused_gemm_epilogue_op_desc.SetInput("Y", {matmul_w->Name()});
    fused_gemm_epilogue_op_desc.SetInput("Bias", {ele_add_bias->Name()});
    fused_gemm_epilogue_op_desc.SetOutput("Out", {ele_add_out->Name()});
    fused_gemm_epilogue_op_desc.SetAttr("activation", activation);
    fused_gemm_epilogue_op_desc.SetAttr("op_role",
                                        matmul_op_desc->GetAttr("op_role"));
    fused_gemm_epilogue_op_desc.SetAttr("trans_x", trans_x);
    fused_gemm_epilogue_op_desc.SetAttr("trans_y", trans_y);

    auto gemm_epilogue_node = g->CreateOpNode(&fused_gemm_epilogue_op_desc);
    IR_NODE_LINK_TO(c_identity_out, gemm_epilogue_node);
    IR_NODE_LINK_TO(matmul_w, gemm_epilogue_node);
    IR_NODE_LINK_TO(ele_add_bias, gemm_epilogue_node);
    IR_NODE_LINK_TO(gemm_epilogue_node, ele_add_out);

    GraphSafeRemoveNodes(g, {matmul_op, ele_add_op});
    found_fused_linear_count++;
  };

  gpd(graph, handler);

  AddStatis(found_fused_linear_count);
  return graph;
}

ir::Graph *FuseLinearWithMPScalePass::FusedLinearBwd(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("mp_fused_linear_grad");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  auto *dout =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(scope_name, "dout"))
          ->AsInput()
          ->assert_is_op_input("elementwise_add_grad", GradVarName("Out"));

  patterns::FusedLinearGrad fused_linear_grad_pattern(gpd.mutable_pattern(),
                                                      "fused_linear_grad");
  fused_linear_grad_pattern(dout);

  int found_mp_fused_linear_grad_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle linear in ColumnParallelLinear backward fuse";

    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_grad_op, ele_add_grad, fused_linear_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add_x, ele_add_x, fused_linear_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_bias, ele_add_bias, fused_linear_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_x_grad, ele_add_x_grad, fused_linear_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_bias_grad, ele_add_bias_grad, fused_linear_grad_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_op, matmul_grad, fused_linear_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_x, matmul_x, fused_linear_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_w, matmul_w, fused_linear_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_x_grad, matmul_x_grad, fused_linear_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_w_grad, matmul_w_grad, fused_linear_grad_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        allreduce_sum_op, allreduce_sum, fused_linear_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        allreduce_sum_out, allreduce_sum_out, fused_linear_grad_pattern);

    std::vector<int64_t> matmul_grad_x_shape = matmul_x->Var()->GetShape();
    std::vector<int64_t> matmul_grad_w_shape = matmul_w->Var()->GetShape();

    // Note (GhostScreaming): We only support matmul_v2_grad from
    // paddle.nn.Linear currently. The conditions below are used to verify
    // wether matmul_v2 is created by paddle.nn.Linear
    auto matmul_grad_op_desc = matmul_grad_op->Op();
    if (!IsGemmFromLinear_(
            matmul_grad_x_shape, matmul_grad_w_shape, matmul_grad_op_desc))
      return;

    bool trans_x, trans_y;
    GetTransposeAttrsFromOp(*matmul_grad_op_desc, &trans_x, &trans_y);

    OpDesc fused_gemm_epilogue_grad_op_desc(ele_add_grad_op->Op()->Block());
    std::string activation_grad = "none";
    fused_gemm_epilogue_grad_op_desc.SetType("fused_gemm_epilogue_grad");
    fused_gemm_epilogue_grad_op_desc.SetInput("DOut",
                                              {subgraph.at(dout)->Name()});
    fused_gemm_epilogue_grad_op_desc.SetInput("X", {matmul_x->Name()});
    fused_gemm_epilogue_grad_op_desc.SetInput("Y", {matmul_w->Name()});
    fused_gemm_epilogue_grad_op_desc.SetOutput("DX", {matmul_x_grad->Name()});
    fused_gemm_epilogue_grad_op_desc.SetOutput("DY", {matmul_w_grad->Name()});
    fused_gemm_epilogue_grad_op_desc.SetOutput("DBias",
                                               {ele_add_bias_grad->Name()});
    fused_gemm_epilogue_grad_op_desc.SetAttr("activation_grad",
                                             activation_grad);
    fused_gemm_epilogue_grad_op_desc.SetAttr(
        "op_role", matmul_grad_op_desc->GetAttr("op_role"));
    fused_gemm_epilogue_grad_op_desc.SetAttr("trans_x", trans_x);
    fused_gemm_epilogue_grad_op_desc.SetAttr("trans_y", trans_y);
    auto matmul_grad_op_role_val =
        details::GetOpRoleVarsOrEmpty(*(matmul_grad_op->Op()));
    auto ele_add_grad_op_role_val =
        details::GetOpRoleVarsOrEmpty(*(ele_add_grad_op->Op()));
    std::vector<std::string> fused_gemm_epilogue_grad_op_role_var;
    for (auto i : matmul_grad_op_role_val) {
      fused_gemm_epilogue_grad_op_role_var.push_back(i);
    }
    for (auto i : ele_add_grad_op_role_val) {
      fused_gemm_epilogue_grad_op_role_var.push_back(i);
    }
    fused_gemm_epilogue_grad_op_desc.SetAttr(
        "op_role_var", fused_gemm_epilogue_grad_op_role_var);

    auto gemm_epilogue_grad_node =
        g->CreateOpNode(&fused_gemm_epilogue_grad_op_desc);

    IR_NODE_LINK_TO(subgraph.at(dout), gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(matmul_x, gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(matmul_w, gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, matmul_x_grad);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, matmul_w_grad);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, ele_add_bias_grad);

    GraphSafeRemoveNodes(g, {ele_add_grad_op, matmul_grad_op});
    found_mp_fused_linear_grad_count++;
  };

  gpd(graph, handler);

  AddStatis(found_mp_fused_linear_grad_count);
  return graph;
}

ir::Graph *FuseLinearWithMPScalePass::FusedLinearWithMpScaleFwd(
    ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("fused_linear_with_mp_scale");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(scope_name, "x"))
                ->AsInput()
                ->assert_is_op_input("matmul_v2", "X");
  patterns::FusedLinearMpScale linear_with_mp_scale_pattern(
      gpd.mutable_pattern(), "linear_with_mp_scale");

  linear_with_mp_scale_pattern(x);

  int found_linear_with_mp_scale_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle FusedLinear with MPScale (RowParallelLinear) fuse";

    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul, linear_with_mp_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_w, matmul_w, linear_with_mp_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_out, matmul_out, linear_with_mp_scale_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        allreduce_sum_op, allreduce_sum, linear_with_mp_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        allreduce_sum_out, allreduce_sum_out, linear_with_mp_scale_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_op, ele_add, linear_with_mp_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_bias, ele_add_bias, linear_with_mp_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_out, ele_add_out, linear_with_mp_scale_pattern);

    std::vector<int64_t> matmul_x_shape = subgraph.at(x)->Var()->GetShape();
    std::vector<int64_t> matmul_w_shape = matmul_w->Var()->GetShape();

    // Note (FuseLinearWithMPScalePass): We only support matmul_v2 from
    // paddle.nn.Linear currently. The conditions below are used to verify
    // wether matmul_v2 is created by paddle.nn.Linear
    auto matmul_op_desc = matmul_op->Op();
    if (!IsGemmFromLinear_(matmul_x_shape, matmul_w_shape, matmul_op_desc))
      return;

    bool trans_x, trans_y;
    GetTransposeAttrsFromOp(*matmul_op_desc, &trans_x, &trans_y);

    OpDesc scale_op_desc(matmul_op->Op()->Block());
    VarDesc scaled_bias_desc(
        patterns::PDNodeName(scope_name, "fused_linear_scaled_bias"));
    scaled_bias_desc.SetShape(ele_add_bias->Var()->GetShape());
    scaled_bias_desc.SetDataType(ele_add_bias->Var()->GetDataType());
    scaled_bias_desc.SetLoDLevel(ele_add_bias->Var()->GetLoDLevel());
    auto *scaled_bias = g->CreateVarNode(&scaled_bias_desc);

    scale_op_desc.SetType("scale");
    scale_op_desc.SetInput("X", {ele_add_bias->Name()});
    scale_op_desc.SetOutput("Out", {scaled_bias->Name()});
    // TODO(GhostScreaming): use ring_id -> nranks map to get mp_degree value.
    // ring_id to nranks map.
    auto nranks_map = this->Get<std::map<int, int>>("nranks_map");
    int ring_id =
        PADDLE_GET_CONST(int, allreduce_sum_op->Op()->GetAttr("ring_id"));
    float mp_degree = 1.0f / static_cast<float>(nranks_map[ring_id]);
    std::cout << "ring_id: " << ring_id << ", mp_degree: " << mp_degree
              << std::endl;
    scale_op_desc.SetAttr("scale", mp_degree);
    auto *mp_scale_op_node = g->CreateOpNode(&scale_op_desc);

    OpDesc fused_gemm_epilogue_op_desc(matmul_op->Op()->Block());
    std::string activation = "none";
    fused_gemm_epilogue_op_desc.SetType("fused_gemm_epilogue");
    fused_gemm_epilogue_op_desc.SetInput("X", {subgraph.at(x)->Name()});
    fused_gemm_epilogue_op_desc.SetInput("Y", {matmul_w->Name()});
    fused_gemm_epilogue_op_desc.SetInput("Bias", {scaled_bias->Name()});
    fused_gemm_epilogue_op_desc.SetOutput("Out", {allreduce_sum_out->Name()});
    fused_gemm_epilogue_op_desc.SetAttr("activation", activation);
    fused_gemm_epilogue_op_desc.SetAttr("op_role",
                                        matmul_op_desc->GetAttr("op_role"));
    fused_gemm_epilogue_op_desc.SetAttr("trans_x", trans_x);
    fused_gemm_epilogue_op_desc.SetAttr("trans_y", trans_y);
    auto *gemm_epilogue_node = g->CreateOpNode(&fused_gemm_epilogue_op_desc);

    OpDesc new_c_allreduce_sum_op_desc(allreduce_sum_op->Op()->Block());
    new_c_allreduce_sum_op_desc.SetType("c_allreduce_sum");
    new_c_allreduce_sum_op_desc.SetInput("X", {allreduce_sum_out->Name()});
    new_c_allreduce_sum_op_desc.SetOutput("Out", {ele_add_out->Name()});
    new_c_allreduce_sum_op_desc.SetAttr("use_model_parallel", true);
    auto *new_c_allreduce_sum_op =
        g->CreateOpNode(&new_c_allreduce_sum_op_desc);

    IR_NODE_LINK_TO(ele_add_bias, mp_scale_op_node);
    IR_NODE_LINK_TO(mp_scale_op_node, scaled_bias);

    IR_NODE_LINK_TO(subgraph.at(x), gemm_epilogue_node);
    IR_NODE_LINK_TO(matmul_w, gemm_epilogue_node);
    IR_NODE_LINK_TO(scaled_bias, gemm_epilogue_node);
    IR_NODE_LINK_TO(gemm_epilogue_node, allreduce_sum_out);

    IR_NODE_LINK_TO(allreduce_sum_out, new_c_allreduce_sum_op);
    IR_NODE_LINK_TO(new_c_allreduce_sum_op, ele_add_out);

    GraphSafeRemoveNodes(g, {matmul_op, ele_add_op, allreduce_sum_op});
    found_linear_with_mp_scale_count++;
  };

  gpd(graph, handler);

  AddStatis(found_linear_with_mp_scale_count);
  return graph;
}

ir::Graph *FuseLinearWithMPScalePass::FusedLinearWithMpScaleBwd(
    ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("fused_linear_scaled_bias_grad");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  auto *dout =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(scope_name, "dout"))
          ->AsInput()
          ->assert_is_op_input("elementwise_add_grad", GradVarName("Out"));

  patterns::FusedLinearMpScaleGrad linear_with_mp_scale_grad_pattern(
      gpd.mutable_pattern(), "fused_linear_scaled_bias_grad");
  linear_with_mp_scale_grad_pattern(dout);

  int found_linear_with_mp_scale_grad_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4)
        << "handle FusedLinear with MPScale (RowParallelLinear) backward fuse";

    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_grad_op, ele_add_grad, linear_with_mp_scale_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_x, ele_add_x, linear_with_mp_scale_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_bias, ele_add_bias, linear_with_mp_scale_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_x_grad, ele_add_x_grad, linear_with_mp_scale_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ele_add_bias_grad,
                              ele_add_bias_grad,
                              linear_with_mp_scale_grad_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        c_identity_op, c_identity, linear_with_mp_scale_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        c_identity_out, c_identity_out, linear_with_mp_scale_grad_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_grad_op, matmul_grad, linear_with_mp_scale_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_x, matmul_x, linear_with_mp_scale_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_w, matmul_w, linear_with_mp_scale_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_x_grad, matmul_x_grad, linear_with_mp_scale_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_w_grad, matmul_w_grad, linear_with_mp_scale_grad_pattern);

    std::vector<int64_t> matmul_grad_x_shape = matmul_x->Var()->GetShape();
    std::vector<int64_t> matmul_grad_w_shape = matmul_w->Var()->GetShape();

    // Note (GhostScreaming): We only support matmul_v2_grad from
    // paddle.nn.Linear currently. The conditions below are used to verify
    // wether matmul_v2 is created by paddle.nn.Linear
    auto matmul_grad_op_desc = matmul_grad_op->Op();
    if (!IsGemmFromLinear_(
            matmul_grad_x_shape, matmul_grad_w_shape, matmul_grad_op_desc))
      return;

    bool trans_x, trans_y;
    GetTransposeAttrsFromOp(*matmul_grad_op_desc, &trans_x, &trans_y);

    OpDesc fused_gemm_epilogue_grad_op_desc(ele_add_grad_op->Op()->Block());
    std::string activation_grad = "none";
    fused_gemm_epilogue_grad_op_desc.SetType("fused_gemm_epilogue_grad");
    fused_gemm_epilogue_grad_op_desc.SetInput("DOut",
                                              {subgraph.at(dout)->Name()});
    fused_gemm_epilogue_grad_op_desc.SetInput("X", {matmul_x->Name()});
    fused_gemm_epilogue_grad_op_desc.SetInput("Y", {matmul_w->Name()});
    fused_gemm_epilogue_grad_op_desc.SetOutput("DX", {matmul_x_grad->Name()});
    fused_gemm_epilogue_grad_op_desc.SetOutput("DY", {matmul_w_grad->Name()});
    fused_gemm_epilogue_grad_op_desc.SetOutput("DBias",
                                               {ele_add_bias_grad->Name()});
    fused_gemm_epilogue_grad_op_desc.SetAttr("activation_grad",
                                             activation_grad);
    fused_gemm_epilogue_grad_op_desc.SetAttr(
        "op_role", matmul_grad_op_desc->GetAttr("op_role"));
    fused_gemm_epilogue_grad_op_desc.SetAttr("trans_x", trans_x);
    fused_gemm_epilogue_grad_op_desc.SetAttr("trans_y", trans_y);

    auto matmul_grad_op_role_val =
        details::GetOpRoleVarsOrEmpty(*(matmul_grad_op->Op()));
    auto ele_add_grad_op_role_val =
        details::GetOpRoleVarsOrEmpty(*(ele_add_grad_op->Op()));
    std::vector<std::string> fused_gemm_epilogue_grad_op_role_var;
    for (auto i : matmul_grad_op_role_val) {
      fused_gemm_epilogue_grad_op_role_var.push_back(i);
    }
    for (auto i : ele_add_grad_op_role_val) {
      fused_gemm_epilogue_grad_op_role_var.push_back(i);
    }
    fused_gemm_epilogue_grad_op_desc.SetAttr(
        "op_role_var", fused_gemm_epilogue_grad_op_role_var);

    auto gemm_epilogue_grad_node =
        g->CreateOpNode(&fused_gemm_epilogue_grad_op_desc);

    IR_NODE_LINK_TO(subgraph.at(dout), gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(matmul_x, gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(matmul_w, gemm_epilogue_grad_node);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, matmul_x_grad);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, matmul_w_grad);
    IR_NODE_LINK_TO(gemm_epilogue_grad_node, ele_add_bias_grad);

    GraphSafeRemoveNodes(g, {ele_add_grad_op, c_identity_op, matmul_grad_op});
    found_linear_with_mp_scale_grad_count++;
  };

  gpd(graph, handler);

  AddStatis(found_linear_with_mp_scale_grad_count);
  return graph;
}

bool FuseLinearWithMPScalePass::IsGemmFromLinear_(
    const std::vector<int64_t> &x_shape,
    const std::vector<int64_t> &w_shape,
    OpDesc *matmul_v2_op) const {
  if (w_shape.size() != 2 || x_shape.size() < 2) return false;
  for (auto attr_name : {"fused_reshape_Out",
                         "fused_reshape_X",
                         "fused_reshape_Y",
                         "fused_transpose_Out",
                         "fused_transpose_X",
                         "fused_transpose_Y"}) {
    if (matmul_v2_op->HasAttr(attr_name)) {
      std::vector<int> tmp_vec =
          PADDLE_GET_CONST(std::vector<int>, matmul_v2_op->GetAttr(attr_name));
      if (tmp_vec.size() > 0) return false;
    }
  }
  return true;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fused_linear_with_mp_scale_pass,
              paddle::framework::ir::FuseLinearWithMPScalePass);
