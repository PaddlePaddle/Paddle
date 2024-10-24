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

#include "paddle/fluid/framework/ir/xpu/reshape2_matmul_xpu_fuse_pass.h"

#include <cmath>
#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {
struct MatmulV2Pattern : public PatternBase {
  MatmulV2Pattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(matmul_v2);
  // declare variable node's name
  PATTERN_DECL_NODE(matmul_x);
  PATTERN_DECL_NODE(matmul_y);
  PATTERN_DECL_NODE(matmul_out);
};

MatmulV2Pattern::MatmulV2Pattern(PDPattern* pattern,
                                 const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto matmul_x = pattern->NewNode(matmul_x_repr())
                      ->assert_is_op_input("matmul_v2", "X")
                      ->AsInput();
  auto* matmul_y = pattern->NewNode(matmul_y_repr())
                       ->assert_is_op_input("matmul_v2", "Y")
                       ->AsInput();
  auto* matmul_v2 = pattern->NewNode(matmul_v2_repr())
                        ->assert_is_op("matmul_v2")
                        ->assert_more([](Node* node) {
                          if (node->inputs.size() != 2) {
                            return false;
                          }
                          return node->inputs[0]->Var()->GetShape().size() ==
                                 node->inputs[1]->Var()->GetShape().size();
                        });
  auto* matmul_out = pattern->NewNode(matmul_out_repr())
                         ->assert_is_op_output("matmul_v2", "Out")
                         ->AsOutput();
  matmul_v2->LinksFrom({matmul_x, matmul_y}).LinksTo({matmul_out});
}

struct Reshape2MatmulPattern : public PatternBase {
  Reshape2MatmulPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(reshape2);
  PATTERN_DECL_NODE(matmul);
  // declare variable node's name
  PATTERN_DECL_NODE(reshape2_in);
  PATTERN_DECL_NODE(matmul_x);
  PATTERN_DECL_NODE(matmul_y);
  PATTERN_DECL_NODE(matmul_out);
};

Reshape2MatmulPattern::Reshape2MatmulPattern(PDPattern* pattern,
                                             const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* reshape2_in =
      pattern->NewNode(reshape2_in_repr())
          ->assert_is_op_input("reshape2", "X")
          ->AsInput()
          ->assert_more([](Node* node) {
            auto reshape2_in_x_shape = node->Var()->GetShape();
            size_t reshape2_in_rank = reshape2_in_x_shape.size();
            return reshape2_in_rank == 4 && ((reshape2_in_x_shape[2] == 1 &&
                                              reshape2_in_x_shape[3] == 1) ||
                                             (reshape2_in_x_shape[1] == 1 &&
                                              reshape2_in_x_shape[3] == 1));
          });
  auto* reshape2 =
      pattern->NewNode(reshape2_repr())
          ->assert_is_op("reshape2")
          ->assert_has_n_inputs(1)
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto reshape2_shape_attr =
                op_desc->GetAttrIfExists<std::vector<int>>("shape");
            return reshape2_shape_attr.size() == 2;
          });
  auto matmul_x = pattern->NewNode(matmul_x_repr())
                      ->assert_is_op_output("reshape2", "Out")
                      ->assert_has_n_outputs(1)
                      ->assert_is_op_input("matmul", "X")
                      ->assert_more([](Node* node) {
                        auto matmul_x_shape = node->Var()->GetShape();
                        size_t matmul_x_rank = matmul_x_shape.size();
                        return matmul_x_rank == 2;
                      });
  auto* matmul_y = pattern->NewNode(matmul_y_repr())
                       ->assert_is_op_input("matmul", "Y")
                       ->assert_is_persistable_var()
                       ->assert_more([](Node* node) {
                         auto matmul_y_shape = node->Var()->GetShape();
                         size_t matmul_y_rank = matmul_y_shape.size();
                         return matmul_y_rank == 2;
                       });
  auto* matmul = pattern->NewNode(matmul_repr())
                     ->assert_is_op("matmul")
                     ->assert_op_attr<bool>("transpose_X", false)
                     ->assert_op_attr<bool>("transpose_Y", false);
  auto* matmul_out = pattern->NewNode(matmul_out_repr())
                         ->assert_is_op_output("matmul", "Out")
                         ->AsOutput();
  reshape2->LinksFrom({reshape2_in}).LinksTo({matmul_x});
  matmul->LinksFrom({matmul_x, matmul_y}).LinksTo({matmul_out});
}

struct Squeeze2MatmulPattern : public PatternBase {
  Squeeze2MatmulPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(squeeze2);
  PATTERN_DECL_NODE(matmul);
  // declare variable node's name
  PATTERN_DECL_NODE(squeeze2_in);
  PATTERN_DECL_NODE(matmul_x);
  PATTERN_DECL_NODE(matmul_y);
  PATTERN_DECL_NODE(matmul_out);
};

Squeeze2MatmulPattern::Squeeze2MatmulPattern(PDPattern* pattern,
                                             const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* squeeze2_in =
      pattern->NewNode(squeeze2_in_repr())
          ->assert_is_op_input("squeeze2", "X")
          ->AsInput()
          ->assert_more([](Node* node) {
            auto squeeze2_in_x_shape = node->Var()->GetShape();
            size_t squeeze2_in_rank = squeeze2_in_x_shape.size();
            return squeeze2_in_rank == 4 &&
                   (squeeze2_in_x_shape[2] == 1 && squeeze2_in_x_shape[3] == 1);
          });
  auto* squeeze2 = pattern->NewNode(squeeze2_repr())
                       ->assert_is_op("squeeze2")
                       ->assert_has_n_inputs(1)
                       ->assert_more([](Node* node) {
                         auto* op_desc = node->Op();
                         auto squeeze2_op_axes =
                             op_desc->GetAttrIfExists<std::vector<int>>("axes");
                         return squeeze2_op_axes == std::vector<int>{2, 3};
                       });
  auto matmul_x = pattern->NewNode(matmul_x_repr())
                      ->assert_is_op_output("squeeze2", "Out")
                      ->assert_has_n_outputs(1)
                      ->assert_is_op_input("matmul", "X")
                      ->assert_more([](Node* node) {
                        auto matmul_x_shape = node->Var()->GetShape();
                        size_t matmul_x_rank = matmul_x_shape.size();
                        return matmul_x_rank == 2;
                      });
  auto* matmul_y = pattern->NewNode(matmul_y_repr())
                       ->assert_is_op_input("matmul", "Y")
                       ->assert_is_persistable_var()
                       ->assert_more([](Node* node) {
                         auto matmul_y_shape = node->Var()->GetShape();
                         size_t matmul_y_rank = matmul_y_shape.size();
                         return matmul_y_rank == 2;
                       });
  auto* matmul = pattern->NewNode(matmul_repr())
                     ->assert_is_op("matmul")
                     ->assert_op_attr<bool>("transpose_X", false)
                     ->assert_op_attr<bool>("transpose_Y", false)
                     ->assert_more([](Node* node) {
                       auto* op_desc = node->Op();
                       auto matmul_alpha_attr =
                           op_desc->GetAttrIfExists<float>("alpha");
                       return std::abs(matmul_alpha_attr - 1.f) < 1e-5;
                     });
  auto* matmul_out = pattern->NewNode(matmul_out_repr())
                         ->assert_is_op_output("matmul", "Out")
                         ->AsOutput();
  squeeze2->LinksFrom({squeeze2_in}).LinksTo({matmul_x});
  matmul->LinksFrom({matmul_x, matmul_y}).LinksTo({matmul_out});
}
}  // namespace patterns

void Reshape2MatmulXPUFusePass::FuseReshape2Matmul(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::Reshape2MatmulPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ReShape2MatmulXPUFusePass";
    /* declare operator node's name */
    GET_IR_NODE(reshape2);
    GET_IR_NODE(matmul);
    /* declare variable node's name*/
    GET_IR_NODE(reshape2_in);
    GET_IR_NODE(matmul_x);
    GET_IR_NODE(matmul_y);
    GET_IR_NODE(matmul_out);

    bool flag = true;
    std::vector<Node*>& next_ops = matmul_out->outputs;
    flag = flag && next_ops.size() == 1 &&
           (next_ops[0]->Name() == "elementwise_add" ||
            next_ops[0]->Name() == "batch_norm");

    if (flag) {
      OpDesc desc(matmul->Op()->Block());
      desc.SetType("mul");
      desc.SetInput("X", {reshape2_in->Name()});
      desc.SetInput("Y", {matmul_y->Name()});
      desc.SetOutput("Out", {matmul_out->Name()});
      desc.SetAttr("x_num_col_dims", 1);
      desc.SetAttr("y_num_col_dims", 1);

      auto mul_node = graph->CreateOpNode(&desc);
      IR_NODE_LINK_TO(reshape2_in, mul_node);
      IR_NODE_LINK_TO(matmul_y, mul_node);
      IR_NODE_LINK_TO(mul_node, matmul_out);
      GraphSafeRemoveNodes(graph, {reshape2, matmul_x, matmul});
      found_subgraph_count++;
    }
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void Reshape2MatmulXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FuseReshape2Matmul(graph);
}

void MapMatmulV2ToMatmulXPUPass::MapMatmulV2ToMatmul(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::MatmulV2Pattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle MapMatmulV2ToMatmulXPUPass";
    /* declare operator node's name */
    GET_IR_NODE(matmul_v2);
    /* declare variable node's name*/
    GET_IR_NODE(matmul_x);
    GET_IR_NODE(matmul_y);
    GET_IR_NODE(matmul_out);

    std::vector<int64_t> x_shape = matmul_x->Var()->GetShape();
    std::vector<int64_t> y_shape = matmul_y->Var()->GetShape();
    uint64_t dims = 2;
    for (size_t i = 0; i < x_shape.size() - dims; ++i) {
      if (x_shape[i] != y_shape[i] && (x_shape[i] == 1 || y_shape[i] == 1)) {
        LOG(WARNING) << "matmul op not support broadcast, please check "
                        "inputs'shape[i]. ";
        return;
      }
    }
    OpDesc desc(matmul_v2->Op()->Block());
    desc.SetType("matmul");
    desc.SetInput("X", {matmul_x->Name()});
    desc.SetInput("Y", {matmul_y->Name()});
    desc.SetOutput("Out", {matmul_out->Name()});
    desc.SetAttr("transpose_X", matmul_v2->Op()->GetAttr("trans_x"));
    desc.SetAttr("transpose_Y", matmul_v2->Op()->GetAttr("trans_y"));
    desc.SetAttr("alpha", 1.0f);
    auto matmul_node = graph->CreateOpNode(&desc);
    IR_NODE_LINK_TO(matmul_x, matmul_node);
    IR_NODE_LINK_TO(matmul_y, matmul_node);
    IR_NODE_LINK_TO(matmul_node, matmul_out);
    GraphSafeRemoveNodes(graph, {matmul_v2});
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void MapMatmulV2ToMatmulXPUPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  MapMatmulV2ToMatmul(graph);
}

void Squeeze2MatmulXPUFusePass::FuseSqueeze2Matmul(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::Squeeze2MatmulPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle Squeeze2MatmulXPUFusePass";
    /* declare operator node's name */
    GET_IR_NODE(squeeze2);
    GET_IR_NODE(matmul);
    /* declare variable node's name*/
    GET_IR_NODE(squeeze2_in);
    GET_IR_NODE(matmul_x);
    GET_IR_NODE(matmul_y);
    GET_IR_NODE(matmul_out);

    bool flag = true;
    std::vector<Node*>& next_ops = matmul_out->outputs;
    flag = flag && next_ops.size() == 1 &&
           (next_ops[0]->Name() == "elementwise_add" ||
            next_ops[0]->Name() == "batch_norm");

    if (flag) {
      OpDesc desc(matmul->Op()->Block());
      desc.SetType("mul");
      desc.SetInput("X", {squeeze2_in->Name()});
      desc.SetInput("Y", {matmul_y->Name()});
      desc.SetOutput("Out", {matmul_out->Name()});
      desc.SetAttr("x_num_col_dims", 1);
      desc.SetAttr("y_num_col_dims", 1);

      auto mul_node = graph->CreateOpNode(&desc);
      IR_NODE_LINK_TO(squeeze2_in, mul_node);
      IR_NODE_LINK_TO(matmul_y, mul_node);
      IR_NODE_LINK_TO(mul_node, matmul_out);
      GraphSafeRemoveNodes(graph, {squeeze2, matmul_x, matmul});
      found_subgraph_count++;
    }
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void Squeeze2MatmulXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FuseSqueeze2Matmul(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(reshape2_matmul_xpu_fuse_pass,
              paddle::framework::ir::Reshape2MatmulXPUFusePass);

REGISTER_PASS_CAPABILITY(reshape2_matmul_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .LE("matmul", 1)
            .EQ("mul", 0));

REGISTER_PASS(map_matmulv2_to_matmul_xpu_pass,
              paddle::framework::ir::MapMatmulV2ToMatmulXPUPass);

REGISTER_PASS_CAPABILITY(map_matmulv2_to_matmul_xpu_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("matmul_v2", 0)
            .LE("matmul", 1));

REGISTER_PASS(squeeze2_matmul_xpu_fuse_pass,
              paddle::framework::ir::Squeeze2MatmulXPUFusePass);

REGISTER_PASS_CAPABILITY(squeeze2_matmul_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("squeeze2", 0)
            .LE("matmul", 1)
            .EQ("mul", 0));
