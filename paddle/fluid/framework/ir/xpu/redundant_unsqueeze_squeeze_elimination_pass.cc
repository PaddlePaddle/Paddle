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

#include "paddle/fluid/framework/ir/xpu/redundant_unsqueeze_squeeze_elimination_pass.h"
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

struct FoldTranspose2OpsPattern : public PatternBase {
  FoldTranspose2OpsPattern(PDPattern* pattern,
                           const std::string& name_scope,
                           const std::string& act_type);

  // declare operator node's name
  PATTERN_DECL_NODE(transpose2_1);
  PATTERN_DECL_NODE(unsqueeze2);
  PATTERN_DECL_NODE(reduce_sum);
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(transpose2_2);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(transpose2_1_out);
  PATTERN_DECL_NODE(unsqueeze2_out);
  PATTERN_DECL_NODE(sum_out);
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(transpose2_2_out);

 private:
  std::string act_type_;
};

FoldTranspose2OpsPattern::FoldTranspose2OpsPattern(
    PDPattern* pattern,
    const std::string& name_scope,
    const std::string& act_type)
    : PatternBase(pattern, name_scope, name_scope), act_type_(act_type) {
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
  transpose2_1->LinksFrom({x}).LinksTo({transpose2_1_out});

  auto* unsqueeze2 =
      pattern->NewNode(unsqueeze2_repr())
          ->assert_is_op("unsqueeze2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axes_array =
                op_desc->GetAttrIfExists<std::vector<int>>("axes");
            return axes_array == std::vector<int>{-2};
          });
  auto* unsqueeze2_out = pattern->NewNode(unsqueeze2_out_repr())
                             ->assert_is_op_output("unsqueeze2", "Out")
                             ->assert_is_op_input("reduce_sum", "X");
  unsqueeze2->LinksFrom({transpose2_1_out}).LinksTo({unsqueeze2_out});

  auto* reduce_sum =
      pattern->NewNode(reduce_sum_repr())
          ->assert_is_op("reduce_sum")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto keep_dim = op_desc->GetAttrIfExists<bool>("keep_dim");
            auto dim_array = op_desc->GetAttrIfExists<std::vector<int>>("dim");
            return dim_array == std::vector<int>{-2} && !keep_dim;
          });
  auto* sum_out = pattern->NewNode(sum_out_repr())
                      ->assert_is_op_output("reduce_sum", "Out")
                      ->assert_is_op_input(act_type_, "X");
  reduce_sum->LinksFrom({unsqueeze2_out}).LinksTo({sum_out});

  auto* act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
  auto* act_out = pattern->NewNode(act_out_repr())
                      ->assert_is_op_output(act_type_, "Out")
                      ->assert_is_op_input("transpose2", "X");
  act->LinksFrom({sum_out}).LinksTo({act_out});

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
  transpose2_2->LinksFrom({act_out}).LinksTo({transpose2_2_out});
}

struct FoldGatherSqueeze2Pattern : public PatternBase {
  FoldGatherSqueeze2Pattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(unsqueeze2_op);
  PATTERN_DECL_NODE(gather_op);
  PATTERN_DECL_NODE(squeeze2_op);

  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(unsqueeze2_op_out);
  PATTERN_DECL_NODE(gather_i);
  PATTERN_DECL_NODE(gather_op_out);
  PATTERN_DECL_NODE(squeeze2_op_out);
};

FoldGatherSqueeze2Pattern::FoldGatherSqueeze2Pattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* x = pattern->NewNode(x_repr())->assert_is_op_input("unsqueeze2", "X");
  auto* unsqueeze2_op =
      pattern->NewNode(unsqueeze2_op_repr())
          ->assert_is_op("unsqueeze2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axes_array =
                op_desc->GetAttrIfExists<std::vector<int>>("axes");
            return axes_array.size() == 1;
          });
  auto* unsqueeze2_op_out = pattern->NewNode(unsqueeze2_op_out_repr())
                                ->assert_is_op_output("unsqueeze2", "Out")
                                ->assert_is_op_input("gather", "X");
  unsqueeze2_op->LinksFrom({x}).LinksTo({unsqueeze2_op_out});
  auto* gather_op = pattern->NewNode(gather_op_repr())->assert_is_op("gather");
  auto* gather_i = pattern->NewNode(gather_i_repr())
                       ->assert_is_op_input("gather", "Index")
                       ->assert_is_persistable_var()
                       ->assert_more([](Node* node) {
                         auto i_shape = node->Var()->GetShape();
                         size_t i_rank = i_shape.size();
                         return i_rank == 1;
                       });
  auto* gather_op_out = pattern->NewNode(gather_op_out_repr())
                            ->assert_is_op_output("gather", "Out")
                            ->assert_is_op_input("squeeze2", "X");
  gather_op->LinksFrom({unsqueeze2_op_out, gather_i}).LinksTo({gather_op_out});
  auto* squeeze2_op =
      pattern->NewNode(squeeze2_op_repr())
          ->assert_is_op("squeeze2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axes_array =
                op_desc->GetAttrIfExists<std::vector<int>>("axes");
            return axes_array.size() == 1;
          });
  auto* squeeze2_op_out = pattern->NewNode(squeeze2_op_out_repr())
                              ->assert_is_op_output("squeeze2", "Out");
  squeeze2_op->LinksFrom({gather_op_out}).LinksTo({squeeze2_op_out});
}

struct FoldConv1dSqueeze2Pattern : public PatternBase {
  FoldConv1dSqueeze2Pattern(PDPattern* pattern,
                            const std::string& name_scope,
                            const std::string& act_type);

  // declare operator node's name
  PATTERN_DECL_NODE(squeeze2);
  PATTERN_DECL_NODE(bn);
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(unsqueeze2);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(squeeze2_out);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_mean);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_var);
  PATTERN_DECL_NODE(bn_out);
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(bn_saved_var);
  PATTERN_DECL_NODE(bn_var_out);
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(unsqueeze2_out);

 private:
  std::string act_type_;
};

FoldConv1dSqueeze2Pattern::FoldConv1dSqueeze2Pattern(
    PDPattern* pattern,
    const std::string& name_scope,
    const std::string& act_type)
    : PatternBase(pattern, name_scope, name_scope), act_type_(act_type) {
  auto* x = pattern->NewNode(x_repr())
                ->assert_is_op_input("squeeze2", "X")
                ->assert_more([](Node* node) {
                  auto x_shape = node->Var()->GetShape();
                  size_t x_rank = x_shape.size();
                  return x_rank == 4 && x_shape[2] == 1;
                });
  auto* squeeze2 = pattern->NewNode(squeeze2_repr())
                       ->assert_is_op("squeeze2")
                       ->assert_more([](Node* node) {
                         auto* op_desc = node->Op();
                         auto axes_array =
                             op_desc->GetAttrIfExists<std::vector<int>>("axes");
                         return axes_array == std::vector<int>{-2} ||
                                axes_array == std::vector<int>{2};
                       });
  auto* squeeze2_out = pattern->NewNode(squeeze2_out_repr())
                           ->assert_is_op_output("squeeze2", "Out")
                           ->assert_is_op_input("batch_norm", "X");
  squeeze2->LinksFrom({x}).LinksTo({squeeze2_out});

  auto* bn_bias = pattern->NewNode(bn_bias_repr())
                      ->AsInput()
                      ->assert_is_persistable_var()
                      ->assert_is_op_input("batch_norm", "Bias")
                      ->assert_has_n_outputs(1);
  auto* bn_mean = pattern->NewNode(bn_mean_repr())
                      ->AsInput()
                      ->assert_is_persistable_var()
                      ->assert_is_op_input("batch_norm", "Mean")
                      ->assert_has_n_outputs(1);
  auto* bn_scale = pattern->NewNode(bn_scale_repr())
                       ->AsInput()
                       ->assert_is_persistable_var()
                       ->assert_is_op_input("batch_norm", "Scale")
                       ->assert_has_n_outputs(1);
  auto* bn_var = pattern->NewNode(bn_var_repr())
                     ->AsInput()
                     ->assert_is_persistable_var()
                     ->assert_is_op_input("batch_norm", "Variance")
                     ->assert_has_n_outputs(1);
  auto* bn = pattern->NewNode(bn_repr())->assert_is_op("batch_norm");
  auto* bn_out = pattern->NewNode(bn_out_repr())
                     ->assert_is_op_output("batch_norm", "Y")
                     ->assert_is_op_input(act_type_, "X");
  auto* bn_mean_out = pattern->NewNode(bn_mean_out_repr())
                          ->assert_is_op_output("batch_norm", "MeanOut");
  auto* bn_saved_mean = pattern->NewNode(bn_saved_mean_repr())
                            ->assert_is_op_output("batch_norm", "SavedMean");
  auto* bn_var_out = pattern->NewNode(bn_var_out_repr())
                         ->assert_is_op_output("batch_norm", "VarianceOut");
  auto* bn_saved_var = pattern->NewNode(bn_saved_var_repr())
                           ->assert_is_op_output("batch_norm", "SavedVariance");
  bn->LinksFrom({squeeze2_out, bn_bias, bn_mean, bn_scale, bn_var})
      .LinksTo({bn_out, bn_mean_out, bn_var_out, bn_saved_mean, bn_saved_var});

  auto act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
  auto act_out = pattern->NewNode(act_out_repr())
                     ->assert_is_op_output(act_type_, "Out")
                     ->assert_is_op_input("unsqueeze2", "X");
  act->LinksFrom({bn_out}).LinksTo({act_out});

  auto* unsqueeze2 =
      pattern->NewNode(unsqueeze2_repr())
          ->assert_is_op("unsqueeze2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axes_array =
                op_desc->GetAttrIfExists<std::vector<int>>("axes");
            return axes_array == std::vector<int>{-2} ||
                   axes_array == std::vector<int>{2};
          });
  auto* unsqueeze2_out = pattern->NewNode(unsqueeze2_out_repr())
                             ->assert_is_op_output("unsqueeze2", "Out");
  unsqueeze2->LinksFrom({act_out}).LinksTo({unsqueeze2_out});
}

}  // namespace patterns

void RedundantUnsqueeze2EliminationPass::FoldConv1dSqueeze2Ops(
    ir::Graph* graph, const std::string& act_type) const {
  GraphPatternDetector gpd;
  patterns::FoldConv1dSqueeze2Pattern pattern(
      gpd.mutable_pattern(), name_scope_, act_type);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FoldConv1dSqueeze2Ops";
    // declare operator node's name
    GET_IR_NODE(squeeze2);
    GET_IR_NODE(bn);
    GET_IR_NODE(act);
    GET_IR_NODE(unsqueeze2);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(squeeze2_out);
    GET_IR_NODE(bn_out);
    GET_IR_NODE(act_out);
    GET_IR_NODE(unsqueeze2_out);

    auto bn_op_desc = bn->Op();
    bn_op_desc->RenameInput(squeeze2_out->Var()->Name(), x->Var()->Name());
    bn_out->Var()->SetShape(x->Var()->GetShape());
    act_out->Var()->SetShape(x->Var()->GetShape());
    bn_op_desc->Flush();
    IR_NODE_LINK_TO(x, bn);
    // behind unsqueeze op node
    auto unsqueeze_out_link_nodes = unsqueeze2_out->outputs;
    for (auto out_link_node : unsqueeze_out_link_nodes) {
      auto op_desc = out_link_node->Op();
      op_desc->RenameInput(unsqueeze2_out->Var()->Name(),
                           act_out->Var()->Name());
      op_desc->Flush();
      IR_NODE_LINK_TO(act_out, out_link_node);
    }
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {
        squeeze2, squeeze2_out, unsqueeze2, unsqueeze2_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void RedundantUnsqueeze2EliminationPass::FoldTranspose2Ops(
    ir::Graph* graph, const std::string& act_type) const {
  GraphPatternDetector gpd;
  patterns::FoldTranspose2OpsPattern pattern(
      gpd.mutable_pattern(), name_scope_, act_type);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FoldTranspose2Ops";
    // declare operator node's name
    GET_IR_NODE(transpose2_1);
    GET_IR_NODE(unsqueeze2);
    GET_IR_NODE(reduce_sum);
    GET_IR_NODE(act);
    GET_IR_NODE(transpose2_2);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(transpose2_1_out);
    GET_IR_NODE(unsqueeze2_out);
    GET_IR_NODE(sum_out);
    GET_IR_NODE(act_out);
    GET_IR_NODE(transpose2_2_out);

    auto act_op_desc = act->Op();
    act_op_desc->RenameInput(sum_out->Var()->Name(), x->Var()->Name());
    act_out->Var()->SetShape(x->Var()->GetShape());
    act_op_desc->Flush();
    IR_NODE_LINK_TO(x, act);
    // behind unsqueeze op node
    auto final_out_link_nodes = transpose2_2_out->outputs;
    for (auto out_link_node : final_out_link_nodes) {
      auto op_desc = out_link_node->Op();
      op_desc->RenameInput(transpose2_2_out->Var()->Name(),
                           act_out->Var()->Name());
      op_desc->Flush();
      IR_NODE_LINK_TO(act_out, out_link_node);
    }
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {transpose2_1,
                                                    transpose2_1_out,
                                                    unsqueeze2,
                                                    unsqueeze2_out,
                                                    reduce_sum,
                                                    sum_out,
                                                    transpose2_2,
                                                    transpose2_2_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void RedundantUnsqueeze2EliminationPass::FoldGatherSqueeze2Ops(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::FoldGatherSqueeze2Pattern pattern(gpd.mutable_pattern(),
                                              name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FoldGatherSqueeze2Ops";
    // declare operator node's name
    GET_IR_NODE(unsqueeze2_op);
    GET_IR_NODE(gather_op);
    GET_IR_NODE(squeeze2_op);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(unsqueeze2_op_out);
    GET_IR_NODE(gather_i);
    GET_IR_NODE(gather_op_out);
    GET_IR_NODE(squeeze2_op_out);

    bool flag = true;
    auto x_shape = x->Var()->GetShape();
    auto x_rank = static_cast<int>(x_shape.size());
    std::vector<int> unsqueeze_axes_attr = PADDLE_GET_CONST(
        std::vector<int>, unsqueeze2_op->Op()->GetAttr("axes"));
    auto unsqueeze_axes = unsqueeze_axes_attr.front();
    unsqueeze_axes =
        unsqueeze_axes < 0 ? unsqueeze_axes + x_rank : unsqueeze_axes;
    auto gather_axis = PADDLE_GET_CONST(int, gather_op->Op()->GetAttr("axis"));
    gather_axis = gather_axis < 0 ? gather_axis + x_rank + 1 : gather_axis;
    std::vector<int> squeeze_axes_attr =
        PADDLE_GET_CONST(std::vector<int>, squeeze2_op->Op()->GetAttr("axes"));
    auto squeeze_axes = squeeze_axes_attr.front();
    squeeze_axes = squeeze_axes < 0 ? squeeze_axes + x_rank + 1 : squeeze_axes;
    flag &= (unsqueeze_axes >= 0 && unsqueeze_axes < x_rank);
    flag &=
        ((gather_axis == unsqueeze_axes + 1) && (squeeze_axes == gather_axis));
    if (!flag) return;
    // x->gather->squeeze2_op_out
    auto gather_op_desc = gather_op->Op();
    gather_op_desc->RenameInput(unsqueeze2_op_out->Var()->Name(),
                                x->Var()->Name());
    gather_op_desc->SetAttr("axis", gather_axis - 1);
    gather_op_out->Var()->SetShape(squeeze2_op_out->Var()->GetShape());
    gather_op_desc->Flush();
    IR_NODE_LINK_TO(x, gather_op);
    // behind squeeze op node
    auto squeeze_out_link_nodes = squeeze2_op_out->outputs;
    for (auto out_link_node : squeeze_out_link_nodes) {
      auto op_desc = out_link_node->Op();
      op_desc->RenameInput(squeeze2_op_out->Var()->Name(),
                           gather_op_out->Var()->Name());
      op_desc->Flush();
      IR_NODE_LINK_TO(gather_op_out, out_link_node);
    }
    std::unordered_set<const Node*> delete_nodes{
        squeeze2_op, squeeze2_op_out, unsqueeze2_op, unsqueeze2_op_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void RedundantUnsqueeze2EliminationPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  for (auto act_type : {"relu"}) {
    FoldTranspose2Ops(graph, act_type);
  }
  FoldGatherSqueeze2Ops(graph);
  for (auto act_type : {"leaky_relu", "elu"}) {
    FoldConv1dSqueeze2Ops(graph, act_type);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(redundant_unsqueeze_squeeze_elimination_pass,
              paddle::framework::ir::RedundantUnsqueeze2EliminationPass);

REGISTER_PASS_CAPABILITY(redundant_unsqueeze_squeeze_elimination_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "conv2d", 0));
