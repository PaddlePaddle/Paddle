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

}  // namespace patterns

void RedundantOnnxOpsEliminationPass::FoldGatherSqueeze2Ops(
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

void RedundantOnnxOpsEliminationPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  FoldGatherSqueeze2Ops(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(redundant_unsqueeze_squeeze_elimination_pass,
              paddle::framework::ir::RedundantOnnxOpsEliminationPass);

REGISTER_PASS_CAPABILITY(redundant_unsqueeze_squeeze_elimination_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "conv2d", 0));
