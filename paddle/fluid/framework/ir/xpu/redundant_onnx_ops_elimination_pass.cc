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

#include "paddle/fluid/framework/ir/xpu/redundant_onnx_ops_elimination_pass.h"
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
            std::vector<int> HW = {static_cast<int>(pool2d_x_shape[2]),
                                   static_cast<int>(pool2d_x_shape[3])};
            auto pool_type =
                op_desc->GetAttrIfExists<std::string>("pooling_type");
            auto ksize_array =
                op_desc->GetAttrIfExists<std::vector<int>>("ksize");
            auto strides_array =
                op_desc->GetAttrIfExists<std::vector<int>>("strides");
            auto paddings_array =
                op_desc->GetAttrIfExists<std::vector<int>>("paddings");
            return pool_type == "avg" && ksize_array == HW &&
                   strides_array == HW &&
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
                         return axes_array == std::vector<int>{-2};
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
            return axes_array == std::vector<int>{-2};
          });
  auto* unsqueeze2_out = pattern->NewNode(unsqueeze2_out_repr())
                             ->assert_is_op_output("unsqueeze2", "Out");
  unsqueeze2->LinksFrom({act_out}).LinksTo({unsqueeze2_out});
}

}  // namespace patterns

void RedundantOnnxOpsEliminationPass::FuseReduceMean(ir::Graph* graph) const {
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
    reduce_op_desc.SetAttr("dim", std::vector<int>{-2});
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

void RedundantOnnxOpsEliminationPass::FoldConv1dSqueeze2Ops(
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

void RedundantOnnxOpsEliminationPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  for (auto act_type : {"leaky_relu", "elu"}) {
    FoldConv1dSqueeze2Ops(graph, act_type);
  }
  FuseReduceMean(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(redundant_onnx_ops_elimination_pass,
              paddle::framework::ir::RedundantOnnxOpsEliminationPass);

REGISTER_PASS_CAPABILITY(redundant_onnx_ops_elimination_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "reduce_mean", 0));
