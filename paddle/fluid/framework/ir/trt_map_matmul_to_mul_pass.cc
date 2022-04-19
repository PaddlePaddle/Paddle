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

#include "paddle/fluid/framework/ir/trt_map_matmul_to_mul_pass.h"

#include <cmath>
#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_proto_maker.h"

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

class Node;

TrtMapMatmul2MulPass::TrtMapMatmul2MulPass() {
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
      .IsNumGE(0.99f)
      .IsNumLE(1.01f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumGE(1)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();
}

TrtMapMatmulV2ToMulPass::TrtMapMatmulV2ToMulPass() {
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
      .IsBoolEQ(false)
      .End()
      .AddAttr("trans_y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumGE(1)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();
}

TrtMapMatmulV2ToMatmulPass::TrtMapMatmulV2ToMatmulPass() {
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

  AddOpCompat(OpCompat("matmul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsNumEQ(1.0f)
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("transpose_X")
      .IsType<bool>()
      .End()
      .AddAttr("transpose_Y")
      .IsType<bool>()
      .End();
}

TrtFlatten2MatmulFusePass::TrtFlatten2MatmulFusePass() {
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
      .IsNumGE(0.99f)
      .IsNumLE(1.01f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")
      .IsBoolEQ(false)
      .End();

  AddOpCompat(OpCompat("flatten2"))
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
      .IsNumEQ(1)
      .End();

  AddOpCompat(OpCompat("mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumGE(1)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();
}

TrtSqueeze2MatmulFusePass::TrtSqueeze2MatmulFusePass() {
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
      .IsNumGE(0.99f)
      .IsNumLE(1.01f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")
      .IsBoolEQ(false)
      .End();

  AddOpCompat(OpCompat("squeeze2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsTensor()
      .End()
      .AddAttr("axes")
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumEQ(1)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();
}

void TrtMapMatmul2MulPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "trt_map_matmul_to_mul_pass";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::Matmul matmul_pattern(gpd.mutable_pattern(), name_scope);
  matmul_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "trt map matmul to mul";
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_x, matmul_in_x, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_y, matmul_in_y, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, matmul_pattern);
    bool flag = true;

    bool transpose_X =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_X"));
    float alpha = BOOST_GET_CONST(float, matmul_op->Op()->GetAttr("alpha"));
    flag = flag && !transpose_X && std::abs(alpha - 1.0) < 1e-5;

    std::vector<int64_t> x_shape = matmul_in_x->Var()->GetShape();
    std::vector<int64_t> y_shape = matmul_in_y->Var()->GetShape();
    size_t x_rank = x_shape.size();
    size_t y_rank = y_shape.size();
    flag = flag && x_rank >= 2 && y_rank == 2;

    if (flag) {
      if (!IsCompat(subgraph, g)) {
        LOG(WARNING) << "TrtMapMatmul2MulPass in op compat failed.";
        return;
      }
      OpDesc desc(matmul_op->Op()->Block());
      desc.SetType("mul");
      desc.SetInput("X", {matmul_in_x->Name()});
      desc.SetInput("Y", {matmul_in_y->Name()});
      desc.SetOutput("Out", {matmul_out->Name()});
      desc.SetAttr("x_num_col_dims", static_cast<int>(x_rank - 1));
      desc.SetAttr("y_num_col_dims", 1);
      desc.SetAttr("transpose_Y", matmul_op->Op()->GetAttr("transpose_Y"));
      if (matmul_op->Op()->HasAttr("enable_int8")) {
        desc.SetAttr("enable_int8", matmul_op->Op()->GetAttr("enable_int8"));
        desc.SetAttr("Input_scale", matmul_op->Op()->GetAttr("Input_scale"));
        desc.SetAttr("out_threshold",
                     matmul_op->Op()->GetAttr("out_threshold"));
      }

      bool inscale_flag = false;
      bool outscale_flag = false;

      if (matmul_op->Op()->HasAttr("X")) {
        desc.SetAttr("X", matmul_op->Op()->GetAttr("X"));
        inscale_flag = true;
      }
      if (matmul_op->Op()->HasAttr("Out")) {
        desc.SetAttr("Out", matmul_op->Op()->GetAttr("Out"));
        outscale_flag = true;
      }
      desc.SetAttr("support_int8", inscale_flag && outscale_flag);

      auto mul_node = g->CreateOpNode(&desc);
      IR_NODE_LINK_TO(matmul_in_x, mul_node);
      IR_NODE_LINK_TO(matmul_in_y, mul_node);
      IR_NODE_LINK_TO(mul_node, matmul_out);
      GraphSafeRemoveNodes(graph, {matmul_op});
      ++found_count;

      if (!IsCompat(desc)) {
        LOG(WARNING) << "TrtMapMatmul2MulPass in out mul op compat failed.";
        return;
      }
    }
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

void TrtMapMatmulV2ToMulPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "trt_map_matmul_v2_to_mul_pass";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::MatmulV2Weight matmul_v2_weight_pattern(gpd.mutable_pattern(),
                                                    name_scope);
  matmul_v2_weight_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(3) << "trt map matmul_v2 to mul";
    GET_IR_NODE_FROM_SUBGRAPH(matmul_v2_in_x, matmul_v2_in_x,
                              matmul_v2_weight_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_v2_in_y, matmul_v2_in_y,
                              matmul_v2_weight_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_v2_op, matmul_v2_op,
                              matmul_v2_weight_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_v2_out, matmul_v2_out,
                              matmul_v2_weight_pattern);

    bool flag = true;
    bool trans_x =
        BOOST_GET_CONST(bool, matmul_v2_op->Op()->GetAttr("trans_x"));
    flag = flag && !trans_x;

    std::vector<int64_t> x_shape = matmul_v2_in_x->Var()->GetShape();
    std::vector<int64_t> y_shape = matmul_v2_in_y->Var()->GetShape();
    size_t x_rank = x_shape.size();
    size_t y_rank = y_shape.size();
    flag = flag && x_rank >= 2 && y_rank == 2;

    if (flag) {
      if (!IsCompat(subgraph, g)) {
        LOG(WARNING) << "TrtMapMatmulV2ToMulPass in op compat failed.";
        return;
      }
      OpDesc desc(matmul_v2_op->Op()->Block());
      desc.SetType("mul");
      desc.SetInput("X", {matmul_v2_in_x->Name()});
      desc.SetInput("Y", {matmul_v2_in_y->Name()});
      desc.SetOutput("Out", {matmul_v2_out->Name()});
      desc.SetAttr("x_num_col_dims", static_cast<int>(x_rank - 1));
      desc.SetAttr("y_num_col_dims", 1);
      desc.SetAttr("transpose_Y", matmul_v2_op->Op()->GetAttr("trans_y"));
      if (matmul_v2_op->Op()->HasAttr("enable_int8")) {
        desc.SetAttr("enable_int8", matmul_v2_op->Op()->GetAttr("enable_int8"));
        desc.SetAttr("Input_scale", matmul_v2_op->Op()->GetAttr("Input_scale"));
        desc.SetAttr("out_threshold",
                     matmul_v2_op->Op()->GetAttr("out_threshold"));
      }

      bool inscale_flag = false;
      bool outscale_flag = false;
      if (matmul_v2_op->Op()->HasAttr("X")) {
        desc.SetAttr("X", matmul_v2_op->Op()->GetAttr("X"));
        inscale_flag = true;
      }
      if (matmul_v2_op->Op()->HasAttr("Out")) {
        desc.SetAttr("Out", matmul_v2_op->Op()->GetAttr("Out"));
        outscale_flag = true;
      }
      desc.SetAttr("support_int8", inscale_flag && outscale_flag);

      auto mul_node = g->CreateOpNode(&desc);
      IR_NODE_LINK_TO(matmul_v2_in_x, mul_node);
      IR_NODE_LINK_TO(matmul_v2_in_y, mul_node);
      IR_NODE_LINK_TO(mul_node, matmul_v2_out);
      GraphSafeRemoveNodes(graph, {matmul_v2_op});
      ++found_count;

      if (!IsCompat(desc)) {
        LOG(WARNING) << "TrtMapMatmulV2ToMulPass in out mul op compat failed.";
        return;
      }
    }
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

void TrtMapMatmulV2ToMatmulPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "trt_map_matmul_v2_to_matmul_pass";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::MatmulV2 matmul_v2_pattern(gpd.mutable_pattern(), name_scope);
  matmul_v2_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "trt map matmul_v2 to matmul";
    GET_IR_NODE_FROM_SUBGRAPH(matmul_v2_in_x, matmul_v2_in_x,
                              matmul_v2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_v2_in_y, matmul_v2_in_y,
                              matmul_v2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_v2_op, matmul_v2_op, matmul_v2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_v2_out, matmul_v2_out, matmul_v2_pattern);
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "TrtMapMatmulV2ToMatmulPass in op compat failed.";
      return;
    }

    std::vector<int64_t> x_shape = matmul_v2_in_x->Var()->GetShape();
    std::vector<int64_t> y_shape = matmul_v2_in_y->Var()->GetShape();
    if (x_shape.size() != y_shape.size()) {
      LOG(WARNING)
          << "matmul op not support broadcast, please check inputs'shape. ";
      return;
    }
    uint64_t dims = 2;
    for (size_t i = 0; i < x_shape.size() - dims; ++i) {
      if (x_shape[i] != y_shape[i] && (x_shape[i] == 1 || y_shape[i] == 1)) {
        LOG(WARNING) << "matmul op not support broadcast, please check "
                        "inputs'shape[i]. ";
        return;
      }
    }

    OpDesc desc(matmul_v2_op->Op()->Block());
    desc.SetType("matmul");
    desc.SetInput("X", {matmul_v2_in_x->Name()});
    desc.SetInput("Y", {matmul_v2_in_y->Name()});
    desc.SetOutput("Out", {matmul_v2_out->Name()});
    desc.SetAttr("transpose_X", matmul_v2_op->Op()->GetAttr("trans_x"));
    desc.SetAttr("transpose_Y", matmul_v2_op->Op()->GetAttr("trans_y"));
    desc.SetAttr("alpha", 1.0f);
    if (matmul_v2_op->Op()->HasAttr("use_mkldnn")) {
      desc.SetAttr("use_mkldnn", matmul_v2_op->Op()->GetAttr("use_mkldnn"));
    }
    if (matmul_v2_op->Op()->HasAttr("enable_int8")) {
      desc.SetAttr("enable_int8", matmul_v2_op->Op()->GetAttr("enable_int8"));
      desc.SetAttr("Input_scale", matmul_v2_op->Op()->GetAttr("Input_scale"));
      desc.SetAttr("out_threshold",
                   matmul_v2_op->Op()->GetAttr("out_threshold"));
    }

    bool inscale_flag = false;
    bool outscale_flag = false;
    if (matmul_v2_op->Op()->HasAttr("X")) {
      desc.SetAttr("X", matmul_v2_op->Op()->GetAttr("X"));
      inscale_flag = true;
    }
    if (matmul_v2_op->Op()->HasAttr("Out")) {
      desc.SetAttr("Out", matmul_v2_op->Op()->GetAttr("Out"));
      outscale_flag = true;
    }
    desc.SetAttr("support_int8", inscale_flag && outscale_flag);

    auto matmul_node = g->CreateOpNode(&desc);
    IR_NODE_LINK_TO(matmul_v2_in_x, matmul_node);
    IR_NODE_LINK_TO(matmul_v2_in_y, matmul_node);
    IR_NODE_LINK_TO(matmul_node, matmul_v2_out);
    GraphSafeRemoveNodes(graph, {matmul_v2_op});
    ++found_count;

    if (!IsCompat(desc)) {
      LOG(WARNING)
          << "TrtMapMatmulV2ToMatmulPass in out matmul op compat failed.";
      return;
    }
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

void TrtSqueeze2MatmulFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "trt_squeeze2_matmul_fuse_pass";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::Squeeze2Matmul fuse_pattern(gpd.mutable_pattern(), name_scope);
  fuse_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "trt fuse squeeze2+matmul to mul";
    GET_IR_NODE_FROM_SUBGRAPH(squeeze2_in_x, squeeze2_in_x, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(squeeze2_op, squeeze2_op, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_x, matmul_in_x, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_y, matmul_in_y, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, fuse_pattern);
    bool flag = true;

    size_t squeeze2_in_x_rank = (squeeze2_in_x->Var()->GetShape()).size();
    std::vector<int> squeeze2_op_axes =
        BOOST_GET_CONST(std::vector<int>, squeeze2_op->Op()->GetAttr("axes"));
    flag = flag && squeeze2_in_x_rank == 4 &&
           squeeze2_op_axes == std::vector<int>{2, 3} &&
           (matmul_in_x->outputs).size() == 1 &&
           matmul_in_y->Var()->Persistable();

    bool transpose_X =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_X"));
    bool transpose_Y =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_Y"));
    float alpha = BOOST_GET_CONST(float, matmul_op->Op()->GetAttr("alpha"));
    size_t matmul_in_x_rank = (matmul_in_x->Var()->GetShape()).size();
    size_t matmul_in_y_rank = (matmul_in_y->Var()->GetShape()).size();
    flag = flag && !transpose_X && !transpose_Y &&
           std::abs(alpha - 1.0) < 1e-5 && matmul_in_x_rank == 2 &&
           matmul_in_y_rank == 2;

    std::vector<Node*>& next_ops = matmul_out->outputs;
    flag = flag && next_ops.size() == 1 &&
           next_ops[0]->Name() == "elementwise_add";

    if (flag) {
      if (!IsCompat(subgraph, g)) {
        LOG(WARNING) << "TrtSqueeze2MatmulFusePass in op compat failed.";
        return;
      }
      OpDesc desc(matmul_op->Op()->Block());
      desc.SetType("mul");
      desc.SetInput("X", {squeeze2_in_x->Name()});
      desc.SetInput("Y", {matmul_in_y->Name()});
      desc.SetOutput("Out", {matmul_out->Name()});
      desc.SetAttr("x_num_col_dims", 1);
      desc.SetAttr("y_num_col_dims", 1);
      if (matmul_op->Op()->HasAttr("enable_int8")) {
        desc.SetAttr("enable_int8", matmul_op->Op()->GetAttr("enable_int8"));
        desc.SetAttr("Input_scale", matmul_op->Op()->GetAttr("Input_scale"));
        desc.SetAttr("out_threshold",
                     matmul_op->Op()->GetAttr("out_threshold"));
      }

      bool inscale_flag_x = false;
      bool outscale_flag = false;

      if (squeeze2_op->Op()->HasAttr("X")) {
        desc.SetAttr("X", squeeze2_op->Op()->GetAttr("X"));
        inscale_flag_x = true;
      }
      if (matmul_op->Op()->HasAttr("Out")) {
        desc.SetAttr("Out", matmul_op->Op()->GetAttr("Out"));
        outscale_flag = true;
      }
      desc.SetAttr("support_int8", inscale_flag_x && outscale_flag);

      auto mul_node = g->CreateOpNode(&desc);
      IR_NODE_LINK_TO(squeeze2_in_x, mul_node);
      IR_NODE_LINK_TO(matmul_in_y, mul_node);
      IR_NODE_LINK_TO(mul_node, matmul_out);
      GraphSafeRemoveNodes(graph, {squeeze2_op, matmul_in_x, matmul_op});
      ++found_count;
      if (!IsCompat(desc)) {
        LOG(WARNING)
            << "TrtSqueeze2MatmulFusePass in out mul op compat failed.";
        return;
      }
    }
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

TrtReshape2MatmulFusePass::TrtReshape2MatmulFusePass() {
  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Shape")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ShapeTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsTensor()
      .End()
      .AddAttr("shape")  // ints
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
      .IsNumGT(0.99999f)
      .IsNumLT(1.00001f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")
      .IsBoolEQ(false)
      .End();

  AddOpCompat(OpCompat("mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumEQ(1)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();
}

void TrtReshape2MatmulFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "trt_reshape2_matmul_fuse_pass";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::Reshape2Matmul fuse_pattern(gpd.mutable_pattern(), name_scope);
  fuse_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "trt fuse reshape2+matmul to mul";
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_in_x, reshape2_in_x, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_op, reshape2_op, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_x, matmul_in_x, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_y, matmul_in_y, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, fuse_pattern);
    bool flag = true;

    size_t reshape2_in_nums = reshape2_op->inputs.size();
    auto reshape2_in_x_shape = reshape2_in_x->Var()->GetShape();
    size_t reshape2_in_x_rank = reshape2_in_x_shape.size();
    std::vector<int> reshape2_op_shape =
        BOOST_GET_CONST(std::vector<int>, reshape2_op->Op()->GetAttr("shape"));
    flag = flag && reshape2_in_nums == 1 && reshape2_in_x_rank == 4 &&
           reshape2_in_x_shape[2] == 1 && reshape2_in_x_shape[3] == 1 &&
           reshape2_op_shape.size() == 2 && (matmul_in_x->outputs).size() == 1;

    bool transpose_X =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_X"));
    bool transpose_Y =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_Y"));
    float alpha = BOOST_GET_CONST(float, matmul_op->Op()->GetAttr("alpha"));
    size_t matmul_in_x_rank = (matmul_in_x->Var()->GetShape()).size();
    size_t matmul_in_y_rank = (matmul_in_y->Var()->GetShape()).size();
    flag = flag && !transpose_X && !transpose_Y &&
           std::abs(alpha - 1.0) < 1e-5 && matmul_in_x_rank == 2 &&
           matmul_in_y_rank == 2 && matmul_in_y->Var()->Persistable();

    std::vector<Node*>& next_ops = matmul_out->outputs;
    flag = flag && next_ops.size() == 1 &&
           next_ops[0]->Name() == "elementwise_add";

    if (flag) {
      if (!IsCompat(subgraph, g)) {
        LOG(WARNING) << "TrtReshape2MatmulFusePass in op compat failed.";
        return;
      }
      OpDesc desc(matmul_op->Op()->Block());
      desc.SetType("mul");
      desc.SetInput("X", {reshape2_in_x->Name()});
      desc.SetInput("Y", {matmul_in_y->Name()});
      desc.SetOutput("Out", {matmul_out->Name()});
      desc.SetAttr("x_num_col_dims", 1);
      desc.SetAttr("y_num_col_dims", 1);
      if (matmul_op->Op()->HasAttr("enable_int8")) {
        desc.SetAttr("enable_int8", matmul_op->Op()->GetAttr("enable_int8"));
        desc.SetAttr("Input_scale", matmul_op->Op()->GetAttr("Input_scale"));
        desc.SetAttr("out_threshold",
                     matmul_op->Op()->GetAttr("out_threshold"));
      }

      bool inscale_flag_x = false;
      bool outscale_flag = false;

      if (reshape2_op->Op()->HasAttr("X")) {
        desc.SetAttr("X", reshape2_op->Op()->GetAttr("X"));
        inscale_flag_x = true;
      }
      if (matmul_op->Op()->HasAttr("Out")) {
        desc.SetAttr("Out", matmul_op->Op()->GetAttr("Out"));
        outscale_flag = true;
      }
      desc.SetAttr("support_int8", inscale_flag_x && outscale_flag);

      if (!IsCompat(desc)) {
        LOG(WARNING)
            << "TrtReshape2MatmulFusePass in out mul op compat failed.";
        return;
      }
      auto mul_node = g->CreateOpNode(&desc);
      IR_NODE_LINK_TO(reshape2_in_x, mul_node);
      IR_NODE_LINK_TO(matmul_in_y, mul_node);
      IR_NODE_LINK_TO(mul_node, matmul_out);
      GraphSafeRemoveNodes(graph, {reshape2_op, matmul_in_x, matmul_op});
      ++found_count;
    }
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

void TrtFlatten2MatmulFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "trt_flatten2_matmul_fuse_pass";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::Flatten2Matmul fuse_pattern(gpd.mutable_pattern(), name_scope);
  fuse_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "trt fuse flatten2+matmul to mul";
    GET_IR_NODE_FROM_SUBGRAPH(flatten2_in_x, flatten2_in_x, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(flatten2_op, flatten2_op, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_x, matmul_in_x, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_y, matmul_in_y, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, fuse_pattern);
    bool pattern_found = true;

    size_t flatten2_in_nums = flatten2_op->inputs.size();
    auto flatten2_in_x_shape = flatten2_in_x->Var()->GetShape();
    size_t flatten2_in_x_rank = flatten2_in_x_shape.size();
    int flatten2_axis =
        BOOST_GET_CONST(int, flatten2_op->Op()->GetAttr("axis"));
    // only convert matmul to mul when the flatten2 has a single input
    // and the rank of input is 4 and the size of the output of matmul
    // is 1.
    pattern_found = pattern_found && flatten2_in_nums == 1 &&
                    flatten2_in_x_rank == 4 &&
                    (matmul_in_x->outputs).size() == 1;

    bool transpose_X =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_X"));
    bool transpose_Y =
        BOOST_GET_CONST(bool, matmul_op->Op()->GetAttr("transpose_Y"));
    float alpha = BOOST_GET_CONST(float, matmul_op->Op()->GetAttr("alpha"));
    size_t matmul_in_x_rank = (matmul_in_x->Var()->GetShape()).size();
    size_t matmul_in_y_rank = (matmul_in_y->Var()->GetShape()).size();
    pattern_found = pattern_found && !transpose_X && !transpose_Y &&
                    std::abs(alpha - 1.0) < 1e-5 && matmul_in_x_rank == 2 &&
                    matmul_in_y_rank == 2 && matmul_in_y->Var()->Persistable();

    std::vector<Node*>& next_ops = matmul_out->outputs;
    // we further require the matmul op is followed by one elementwise
    // add op.
    pattern_found = pattern_found && next_ops.size() == 1 &&
                    next_ops[0]->Name() == "elementwise_add";

    if (pattern_found) {
      if (!IsCompat(subgraph, g)) {
        LOG(WARNING) << "TrtFlatten2MatmulFusePass in op compat failed.";
        return;
      }
      OpDesc desc(matmul_op->Op()->Block());
      desc.SetType("mul");
      desc.SetInput("X", {flatten2_in_x->Name()});
      desc.SetInput("Y", {matmul_in_y->Name()});
      desc.SetOutput("Out", {matmul_out->Name()});
      desc.SetAttr("x_num_col_dims", flatten2_axis);
      desc.SetAttr("y_num_col_dims", 1);
      if (matmul_op->Op()->HasAttr("enable_int8")) {
        desc.SetAttr("enable_int8", matmul_op->Op()->GetAttr("enable_int8"));
        desc.SetAttr("Input_scale", matmul_op->Op()->GetAttr("Input_scale"));
        desc.SetAttr("out_threshold",
                     matmul_op->Op()->GetAttr("out_threshold"));
      }

      bool inscale_flag_x = false;
      bool outscale_flag = false;

      if (flatten2_op->Op()->HasAttr("X")) {
        desc.SetAttr("X", flatten2_op->Op()->GetAttr("X"));
        inscale_flag_x = true;
      }
      if (matmul_op->Op()->HasAttr("Out")) {
        desc.SetAttr("Out", matmul_op->Op()->GetAttr("Out"));
        outscale_flag = true;
      }
      desc.SetAttr("support_int8", inscale_flag_x && outscale_flag);

      auto mul_node = g->CreateOpNode(&desc);
      IR_NODE_LINK_TO(flatten2_in_x, mul_node);
      IR_NODE_LINK_TO(matmul_in_y, mul_node);
      IR_NODE_LINK_TO(mul_node, matmul_out);
      GraphSafeRemoveNodes(graph, {flatten2_op, matmul_in_x, matmul_op});
      ++found_count;

      if (!IsCompat(desc)) {
        LOG(WARNING)
            << "TrtFlatten2MatmulFusePass in out mul op compat failed.";
        return;
      }
    }
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(trt_map_matmul_to_mul_pass,
              paddle::framework::ir::TrtMapMatmul2MulPass);
REGISTER_PASS_CAPABILITY(trt_map_matmul_to_mul_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("matmul", 1)
            .EQ("mul", 0));

REGISTER_PASS(trt_map_matmul_v2_to_mul_pass,
              paddle::framework::ir::TrtMapMatmulV2ToMulPass);
REGISTER_PASS_CAPABILITY(trt_map_matmul_v2_to_mul_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("matmul_v2", 0)
            .EQ("mul", 0));

REGISTER_PASS(trt_map_matmul_v2_to_matmul_pass,
              paddle::framework::ir::TrtMapMatmulV2ToMatmulPass);
REGISTER_PASS_CAPABILITY(trt_map_matmul_v2_to_matmul_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("matmul_v2", 0)
            .LE("matmul", 1));

REGISTER_PASS(trt_squeeze2_matmul_fuse_pass,
              paddle::framework::ir::TrtSqueeze2MatmulFusePass);
REGISTER_PASS_CAPABILITY(trt_squeeze2_matmul_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("matmul", 1)
            .EQ("squeeze2", 0)
            .EQ("mul", 0));

REGISTER_PASS(trt_reshape2_matmul_fuse_pass,
              paddle::framework::ir::TrtReshape2MatmulFusePass);
REGISTER_PASS_CAPABILITY(trt_reshape2_matmul_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("matmul", 1)
            .EQ("reshape2", 0)
            .EQ("mul", 0));

REGISTER_PASS(trt_flatten2_matmul_fuse_pass,
              paddle::framework::ir::TrtFlatten2MatmulFusePass);
REGISTER_PASS_CAPABILITY(trt_flatten2_matmul_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("matmul", 1)
            .EQ("flatten2", 0)
            .EQ("mul", 0));
