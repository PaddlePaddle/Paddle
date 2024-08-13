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

#include "paddle/fluid/framework/ir/xpu/reduce_ops_fuse_pass.h"
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

struct ReduceMaxFusePattern : public PatternBase {
  ReduceMaxFusePattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(transpose2_1);
  PATTERN_DECL_NODE(unsqueeze2);
  PATTERN_DECL_NODE(pool2d);
  PATTERN_DECL_NODE(squeeze2);
  PATTERN_DECL_NODE(transpose2_2);
  // declare variable node's name
  PATTERN_DECL_NODE(transpose2_1_out);
  PATTERN_DECL_NODE(unsqueeze2_out);
  PATTERN_DECL_NODE(pool2d_out);
  PATTERN_DECL_NODE(squeeze2_out);
  PATTERN_DECL_NODE(transpose2_2_out);
};

ReduceMaxFusePattern::ReduceMaxFusePattern(PDPattern* pattern,
                                           const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* x = pattern->NewNode(x_repr())
                ->assert_is_op_input("transpose2", "X")
                ->assert_more([](Node* node) {
                  auto x_shape = node->Var()->GetShape();
                  size_t x_rank = x_shape.size();
                  return x_rank == 3;
                });
  auto* transpose2_1 =
      pattern->NewNode(transpose2_1_repr())
          ->assert_is_op("transpose2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axis_array =
                op_desc->GetAttrIfExists<std::vector<int>>("axis");
            return axis_array == std::vector<int>{0, 2, 1};
          });
  auto* transpose2_1_out = pattern->NewNode(transpose2_1_out_repr())
                               ->assert_is_op_output("transpose2", "Out")
                               ->assert_is_op_input("unsqueeze2", "X");

  auto* unsqueeze2 =
      pattern->NewNode(unsqueeze2_repr())
          ->assert_is_op("unsqueeze2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axes_array =
                op_desc->GetAttrIfExists<std::vector<int>>("axes");
            return axes_array == std::vector<int>{2};
          });
  auto* unsqueeze2_out = pattern->NewNode(unsqueeze2_out_repr())
                             ->assert_is_op_output("unsqueeze2", "Out")
                             ->assert_is_op_input("pool2d", "X");

  auto* pool2d =
      pattern->NewNode(pool2d_repr())
          ->assert_is_op("pool2d")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto input_var = node->inputs[0]->Var();
            auto pool2d_x_shape = input_var->GetShape();
            std::vector<int> hw = {static_cast<int>(pool2d_x_shape[2]),
                                   static_cast<int>(pool2d_x_shape[3])};
            auto pool_type =
                op_desc->GetAttrIfExists<std::string>("pooling_type");
            auto ksize_array =
                op_desc->GetAttrIfExists<std::vector<int>>("ksize");
            auto strides_array =
                op_desc->GetAttrIfExists<std::vector<int>>("strides");
            auto paddings_array =
                op_desc->GetAttrIfExists<std::vector<int>>("paddings");
            return pool_type == "max" && ksize_array == hw &&
                   strides_array == hw &&
                   paddings_array == std::vector<int>{0, 0};
          });
  auto* pool2d_out = pattern->NewNode(pool2d_out_repr())
                         ->assert_is_op_output("pool2d", "Out")
                         ->assert_is_op_input("squeeze2", "X");

  auto* squeeze2 = pattern->NewNode(squeeze2_repr())
                       ->assert_is_op("squeeze2")
                       ->assert_more([](Node* node) {
                         auto* op_desc = node->Op();
                         auto axes_array =
                             op_desc->GetAttrIfExists<std::vector<int>>("axes");
                         return axes_array == std::vector<int>{2};
                       });
  auto* squeeze2_out = pattern->NewNode(squeeze2_out_repr())
                           ->assert_is_op_output("squeeze2", "Out")
                           ->assert_is_op_input("transpose2", "X");

  auto* transpose2_2 =
      pattern->NewNode(transpose2_2_repr())
          ->assert_is_op("transpose2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axis_array =
                op_desc->GetAttrIfExists<std::vector<int>>("axis");
            return axis_array == std::vector<int>{0, 2, 1};
          });
  auto* transpose2_2_out = pattern->NewNode(transpose2_2_out_repr())
                               ->assert_is_op_output("transpose2", "Out");

  transpose2_1->LinksFrom({x}).LinksTo({transpose2_1_out});
  unsqueeze2->LinksFrom({transpose2_1_out}).LinksTo({unsqueeze2_out});
  pool2d->LinksFrom({unsqueeze2_out}).LinksTo({pool2d_out});
  squeeze2->LinksFrom({pool2d_out}).LinksTo({squeeze2_out});
  transpose2_2->LinksFrom({squeeze2_out}).LinksTo({transpose2_2_out});
}

struct ReduceMeanFusePattern : public PatternBase {
  ReduceMeanFusePattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(unsqueeze2);
  PATTERN_DECL_NODE(pool2d);
  PATTERN_DECL_NODE(squeeze2);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(unsqueeze2_out);
  PATTERN_DECL_NODE(pool2d_out);
  PATTERN_DECL_NODE(squeeze2_out);
};

ReduceMeanFusePattern::ReduceMeanFusePattern(PDPattern* pattern,
                                             const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* x = pattern->NewNode(x_repr())
                ->assert_is_op_input("unsqueeze2", "X")
                ->assert_more([](Node* node) {
                  auto x_shape = node->Var()->GetShape();
                  size_t x_rank = x_shape.size();
                  return x_rank == 3;
                });
  auto* unsqueeze2 =
      pattern->NewNode(unsqueeze2_repr())
          ->assert_is_op("unsqueeze2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axes_array =
                op_desc->GetAttrIfExists<std::vector<int>>("axes");
            return axes_array == std::vector<int>{2};
          });
  auto* unsqueeze2_out = pattern->NewNode(unsqueeze2_out_repr())
                             ->assert_is_op_output("unsqueeze2", "Out")
                             ->assert_is_op_input("pool2d", "X");

  auto* pool2d =
      pattern->NewNode(pool2d_repr())
          ->assert_is_op("pool2d")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto input_var = node->inputs[0]->Var();
            auto pool2d_x_shape = input_var->GetShape();
            std::vector<int> hw = {static_cast<int>(pool2d_x_shape[2]),
                                   static_cast<int>(pool2d_x_shape[3])};
            auto pool_type =
                op_desc->GetAttrIfExists<std::string>("pooling_type");
            auto ksize_array =
                op_desc->GetAttrIfExists<std::vector<int>>("ksize");
            auto strides_array =
                op_desc->GetAttrIfExists<std::vector<int>>("strides");
            auto paddings_array =
                op_desc->GetAttrIfExists<std::vector<int>>("paddings");
            return pool_type == "avg" && ksize_array == hw &&
                   strides_array == hw &&
                   paddings_array == std::vector<int>{0, 0};
          });
  auto* pool2d_out = pattern->NewNode(pool2d_out_repr())
                         ->assert_is_op_output("pool2d", "Out")
                         ->assert_is_op_input("squeeze2", "X");

  auto* squeeze2 = pattern->NewNode(squeeze2_repr())
                       ->assert_is_op("squeeze2")
                       ->assert_more([](Node* node) {
                         auto* op_desc = node->Op();
                         auto axes_array =
                             op_desc->GetAttrIfExists<std::vector<int>>("axes");
                         return axes_array == std::vector<int>{2};
                       });
  auto* squeeze2_out = pattern->NewNode(squeeze2_out_repr())
                           ->assert_is_op_output("squeeze2", "Out")
                           ->assert_is_op_input("transpose2", "X");

  unsqueeze2->LinksFrom({x}).LinksTo({unsqueeze2_out});
  pool2d->LinksFrom({unsqueeze2_out}).LinksTo({pool2d_out});
  squeeze2->LinksFrom({pool2d_out}).LinksTo({squeeze2_out});
}

}  // namespace patterns

void ReduceOpsFusePass::FuseReduceMax(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::ReduceMaxFusePattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ReduceMaxFusePass";
    // declare operator node's name
    GET_IR_NODE(x);
    GET_IR_NODE(transpose2_1);
    GET_IR_NODE(unsqueeze2);
    GET_IR_NODE(pool2d);
    GET_IR_NODE(squeeze2);
    GET_IR_NODE(transpose2_2);
    // declare variable node's name
    GET_IR_NODE(transpose2_1_out);
    GET_IR_NODE(unsqueeze2_out);
    GET_IR_NODE(pool2d_out);
    GET_IR_NODE(squeeze2_out);
    GET_IR_NODE(transpose2_2_out);

    auto* block = transpose2_1->Op()->Block();
    // Generate reshape2 op
    framework::OpDesc reduce_op_desc(block);
    reduce_op_desc.SetType("reduce_max");
    reduce_op_desc.SetInput("X", {x->Name()});
    reduce_op_desc.SetAttr("dim", std::vector<int>{-2});
    reduce_op_desc.SetAttr("reduce_all", false);
    reduce_op_desc.SetAttr("keep_dim", true);
    reduce_op_desc.SetOutput("Out", {transpose2_2_out->Name()});

    auto* reduce = graph->CreateOpNode(&reduce_op_desc);

    IR_NODE_LINK_TO(x, reduce);
    IR_NODE_LINK_TO(reduce, transpose2_2_out);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {transpose2_1,
                                                    transpose2_1_out,
                                                    unsqueeze2,
                                                    unsqueeze2_out,
                                                    pool2d,
                                                    pool2d_out,
                                                    squeeze2,
                                                    squeeze2_out,
                                                    transpose2_2};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void ReduceOpsFusePass::FuseReduceMean(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::ReduceMeanFusePattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FuseReduceMean";
    // declare operator node's name
    GET_IR_NODE(unsqueeze2);
    GET_IR_NODE(pool2d);
    GET_IR_NODE(squeeze2);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(unsqueeze2_out);
    GET_IR_NODE(pool2d_out);
    GET_IR_NODE(squeeze2_out);

    auto* block = pool2d->Op()->Block();
    // Generate reduce_mean op
    framework::OpDesc reduce_op_desc(block);
    reduce_op_desc.SetType("reduce_mean");
    reduce_op_desc.SetInput("X", {x->Name()});
    reduce_op_desc.SetAttr("dim", std::vector<int>{-1});
    reduce_op_desc.SetAttr("reduce_all", false);
    reduce_op_desc.SetAttr("keep_dim", true);
    reduce_op_desc.SetOutput("Out", {squeeze2_out->Name()});

    auto* reduce_op = graph->CreateOpNode(&reduce_op_desc);

    IR_NODE_LINK_TO(x, reduce_op);
    IR_NODE_LINK_TO(reduce_op, squeeze2_out);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {
        unsqueeze2, unsqueeze2_out, pool2d, pool2d_out, squeeze2};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void ReduceOpsFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FuseReduceMax(graph);
  FuseReduceMean(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(reduce_ops_fuse_pass, paddle::framework::ir::ReduceOpsFusePass);

REGISTER_PASS_CAPABILITY(reduce_ops_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reduce_max", 0)
            .EQ("reduce_mean", 0));
