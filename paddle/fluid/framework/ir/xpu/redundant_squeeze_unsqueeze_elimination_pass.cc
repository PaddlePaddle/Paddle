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

#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

// Delete redundant squeeze/unsqueeze op
/*
For example:
graph:
      Input
        |
        |
     squeeze
        |
        |
   squeeze out
        |
        |
  activation(leaky_relu)
        |
        |
  activation out
        |
        |
    unsqueeze
        |
        |
      Output
------------------------------------------------------
After the pass is applied:
      Input
        |
        |
  activation(leaky_relu)
        |
        |
      Output
*/
struct SqueezeActivationUnsqueezeEliminationPattern : public PatternBase {
  SqueezeActivationUnsqueezeEliminationPattern(PDPattern* pattern,
                                               const std::string& name_scope,
                                               const std::string& act_type);
  // declare operator node's name
  PATTERN_DECL_NODE(squeeze);
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(unsqueeze);
  // declare variable node's name
  PATTERN_DECL_NODE(squeeze_input);
  PATTERN_DECL_NODE(squeeze_out);
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(unsqueeze_out);

 private:
  std::string act_type_;
};

SqueezeActivationUnsqueezeEliminationPattern::
    SqueezeActivationUnsqueezeEliminationPattern(PDPattern* pattern,
                                                 const std::string& name_scope,
                                                 const std::string& act_type)
    : PatternBase(pattern, name_scope, name_scope), act_type_(act_type) {
  // squeeze
  auto squeeze = pattern->NewNode(squeeze_repr())->assert_is_op("squeeze2");
  auto squeeze_input = pattern->NewNode(squeeze_input_repr())
                           ->assert_is_op_input("squeeze2", "X")
                           ->AsInput();
  auto squeeze_out = pattern->NewNode(squeeze_out_repr())
                         ->assert_is_op_output("squeeze2", "Out");
  squeeze->LinksFrom({squeeze_input}).LinksTo({squeeze_out});
  // activation
  auto act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
  auto act_out =
      pattern->NewNode(act_out_repr())->assert_is_op_output(act_type_, "Out");
  squeeze_out->assert_is_op_input(act_type_, "X");
  act->LinksFrom({squeeze_out}).LinksTo({act_out});
  // unsqueeze
  auto unsqueeze =
      pattern->NewNode(unsqueeze_repr())->assert_is_op("unsqueeze2");
  auto unsqueeze_out = pattern->NewNode(unsqueeze_out_repr())
                           ->assert_is_op_output("unsqueeze2", "Out")
                           ->AsOutput();
  act_out->assert_is_op_input("unsqueeze2", "X");
  unsqueeze->LinksFrom({act_out}).LinksTo({unsqueeze_out});
}

/*
Function Description:Delete redundant squeeze/unsqueeze op
Pattern: custom pattern
For example:
graph:
      Input1
        |
        |
    squeeze1
        |
        |
   squeeze1 out                                               Input2
        |                                                       |
        |                                                       |
  activation1(leaky_relu)                                     squeeze2
        |                                                       |
        |                                                       |
  activation1 out                                           squeeze2 out
        |                                                       |
        |                                                       |
         - - - - elementwise operation(elementwise_add) - - - -
                                |
                                |
                        activation2(leaky_relu)
                                |
                                |
                          activation2 out
                                |
                                |
               - - - - - - - - - - - - - - - - - - -
               |           |          |            |
               |           |          |            |
          unsqueeze 1    ......  unsqueeze n-1  unsqueeze n
               |           |          |            |
               |           |          |            |
            Output 1     ......    Output n-1    Output n

------------------------------------------------------
After the pass is applied:
      Input1
        |
        |
  activation1(leaky_relu)
        |
        |
  activation1 out                                            Input2
        |                                                       |
        |                                                       |
         - - - - elementwise operation(elementwise_add) - - - -
                                |
                                |
                        activation2(leaky_relu)
                                |
                                |
                          activation2 out
                                |
                                |
               - - - - - - - - - - - - - - - - - - -
               |           |          |            |
               |           |          |            |
            Output 1     ......    Output n-1    Output n
*/
struct CustomSqueezeUnsqueezeEliminationPattern : public PatternBase {
  CustomSqueezeUnsqueezeEliminationPattern(PDPattern* pattern,
                                           const std::string& name_scope,
                                           const std::string& act1_type,
                                           const std::string& act2_type,
                                           const std::string& elementwise_type,
                                           const bool act1_in_branch_x);
  // declare operator node's name
  PATTERN_DECL_NODE(squeeze1);
  PATTERN_DECL_NODE(squeeze2);
  PATTERN_DECL_NODE(act1);
  PATTERN_DECL_NODE(elementwise);
  PATTERN_DECL_NODE(act2);
  // declare variable node's name
  PATTERN_DECL_NODE(squeeze1_input);
  PATTERN_DECL_NODE(squeeze1_out);
  PATTERN_DECL_NODE(act1_out);
  PATTERN_DECL_NODE(squeeze2_input);
  PATTERN_DECL_NODE(squeeze2_out);
  PATTERN_DECL_NODE(elementwise_out);
  PATTERN_DECL_NODE(act2_out);

 private:
  std::string act1_type_;
  std::string act2_type_;
  std::string elementwise_type_;
  bool act1_in_branch_x_;
};

CustomSqueezeUnsqueezeEliminationPattern::
    CustomSqueezeUnsqueezeEliminationPattern(
        PDPattern* pattern,
        const std::string& name_scope,
        const std::string& act1_type,
        const std::string& act2_type,
        const std::string& elementwise_type,
        const bool act1_in_branch_x)
    : PatternBase(pattern, name_scope, name_scope),
      act1_type_(act1_type),
      act2_type_(act2_type),
      elementwise_type_(elementwise_type),
      act1_in_branch_x_(act1_in_branch_x) {
  // squeeze1
  auto squeeze1 = pattern->NewNode(squeeze1_repr())->assert_is_op("squeeze2");
  auto squeeze1_input = pattern->NewNode(squeeze1_input_repr())
                            ->assert_is_op_input("squeeze2", "X")
                            ->AsInput();
  auto squeeze1_out = pattern->NewNode(squeeze1_out_repr())
                          ->assert_is_op_output("squeeze2", "Out");
  squeeze1->LinksFrom({squeeze1_input}).LinksTo({squeeze1_out});
  // activation1
  auto act1 = pattern->NewNode(act1_repr())->assert_is_op(act1_type_);
  auto act1_out =
      pattern->NewNode(act1_out_repr())->assert_is_op_output(act1_type_, "Out");
  squeeze1_out->assert_is_op_input(act1_type_, "X");
  act1->LinksFrom({squeeze1_out}).LinksTo({act1_out});
  // squeeze2
  auto squeeze2 = pattern->NewNode(squeeze2_repr())->assert_is_op("squeeze2");
  auto squeeze2_input = pattern->NewNode(squeeze2_input_repr())
                            ->assert_is_op_input("squeeze2", "X")
                            ->AsInput();
  auto squeeze2_out = pattern->NewNode(squeeze2_out_repr())
                          ->assert_is_op_output("squeeze2", "Out");
  squeeze2->LinksFrom({squeeze2_input}).LinksTo({squeeze2_out});
  // elementwise
  auto elementwise =
      pattern->NewNode(elementwise_repr())->assert_is_op(elementwise_type_);
  auto elementwise_out = pattern->NewNode(elementwise_out_repr())
                             ->assert_is_op_output(elementwise_type_, "Out");
  if (act1_in_branch_x_) {
    act1_out->assert_is_op_input(elementwise_type_, "X");
    squeeze2_out->assert_is_op_input(elementwise_type_, "Y");
  } else {
    act1_out->assert_is_op_input(elementwise_type_, "Y");
    squeeze2_out->assert_is_op_input(elementwise_type_, "X");
  }
  elementwise->LinksFrom({act1_out, squeeze2_out}).LinksTo({elementwise_out});
  // activation2
  auto act2 = pattern->NewNode(act2_repr())->assert_is_op(act2_type_);
  auto act2_out =
      pattern->NewNode(act2_out_repr())->assert_is_op_output(act2_type_, "Out");
  elementwise_out->assert_is_op_input(act2_type_, "X");
  act2->LinksFrom({elementwise_out}).LinksTo({act2_out});
  act2_out->AsOutput();
}

}  // namespace patterns

class SqueezeActivationUnsqueezeEliminationPass : public FusePassBase {
 public:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph, const std::string& act_type) const;

  const std::string name_scope_{
      "squeeze_activation_unsqueeze_elimination_pass"};
};

void SqueezeActivationUnsqueezeEliminationPass::ApplyImpl(
    ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  std::vector<std::string> support_act_type{"relu",
                                            "sigmoid",
                                            "tanh",
                                            "gelu",
                                            "leaky_relu",
                                            "hard_swish",
                                            "hard_sigmoid",
                                            "relu6",
                                            "swish"};
  int found_subgraph_count = 0;
  for (auto act_type : support_act_type) {
    found_subgraph_count += ApplyImpl(graph, act_type);
  }
  AddStatis(found_subgraph_count);
}

int SqueezeActivationUnsqueezeEliminationPass::ApplyImpl(
    ir::Graph* graph, const std::string& act_type) const {
  GraphPatternDetector gpd;
  patterns::SqueezeActivationUnsqueezeEliminationPattern pattern(
      gpd.mutable_pattern(), name_scope_, act_type);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle squeeze activation unsqueeze elimination.";
    /* Get operator node's name */
    GET_IR_NODE(squeeze);
    GET_IR_NODE(act);
    GET_IR_NODE(unsqueeze);
    /* Get variable node's name*/
    GET_IR_NODE(squeeze_input);
    GET_IR_NODE(squeeze_out);
    GET_IR_NODE(act_out);
    GET_IR_NODE(unsqueeze_out);

    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    // Judge squeeze1 && squeeze2 op shape is same or not, if axes is same, the
    // shape is same too.
    std::vector<int> squeeze_axes =
        PADDLE_GET_CONST(std::vector<int>, squeeze->Op()->GetAttr("axes"));
    std::vector<int> unsqueeze_axes =
        PADDLE_GET_CONST(std::vector<int>, unsqueeze->Op()->GetAttr("axes"));
    bool elimination = (squeeze_axes == unsqueeze_axes);
    if (!elimination) return;
    // act
    auto act_op_desc = act->Op();
    act_op_desc->RenameInput(squeeze_out->Var()->Name(),
                             squeeze_input->Var()->Name());
    act_out->Var()->SetShape(squeeze_input->Var()->GetShape());
    act_op_desc->Flush();
    IR_NODE_LINK_TO(squeeze_input, act);
    // behind unsqueeze op node
    auto unsqueeze_out_link_nodes = unsqueeze_out->outputs;
    for (auto out_link_node : unsqueeze_out_link_nodes) {
      auto op_desc = out_link_node->Op();
      op_desc->RenameInput(unsqueeze_out->Var()->Name(),
                           act_out->Var()->Name());
      op_desc->Flush();
      IR_NODE_LINK_TO(act_out, out_link_node);
    }
    std::unordered_set<const Node*> delete_nodes{
        squeeze, squeeze_out, unsqueeze, unsqueeze_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };
  gpd(graph, handler);
  return found_subgraph_count;
}

class CustomSqueezeUnsqueezeEliminationPass : public FusePassBase {
 public:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                const std::string& act1_type,
                const std::string& act2_type,
                const std::string& elementwise_type,
                bool act1_in_branch_x) const;

  const std::string name_scope_{"custom_squeeze_unsqueeze_elimination_pass"};
};

void CustomSqueezeUnsqueezeEliminationPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  std::vector<std::string> support_act_type{"relu",
                                            "sigmoid",
                                            "tanh",
                                            "gelu",
                                            "leaky_relu",
                                            "hard_swish",
                                            "hard_sigmoid",
                                            "relu6",
                                            "swish"};
  std::vector<std::string> support_elementwise_type{"elementwise_add",
                                                    "elementwise_sub",
                                                    "elementwise_mul",
                                                    "elementwise_div"};
  int found_subgraph_count = 0;
  for (auto act1_type : support_act_type) {
    for (auto act2_type : support_act_type) {
      for (auto elementwise_type : support_elementwise_type) {
        for (auto act1_in_branch_x : {true, false}) {
          found_subgraph_count += ApplyImpl(
              graph, act1_type, act2_type, elementwise_type, act1_in_branch_x);
        }
      }
    }
  }
  AddStatis(found_subgraph_count);
}

int CustomSqueezeUnsqueezeEliminationPass::ApplyImpl(
    ir::Graph* graph,
    const std::string& act1_type,
    const std::string& act2_type,
    const std::string& elementwise_type,
    const bool act1_in_branch_x) const {
  GraphPatternDetector gpd;
  patterns::CustomSqueezeUnsqueezeEliminationPattern pattern(
      gpd.mutable_pattern(),
      name_scope_,
      act1_type,
      act2_type,
      elementwise_type,
      act1_in_branch_x);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle custom squeeze unsqueeze elimination pass.";
    /* Get operator node's name */
    GET_IR_NODE(squeeze1);
    GET_IR_NODE(squeeze2);
    GET_IR_NODE(act1);
    GET_IR_NODE(elementwise);
    GET_IR_NODE(act2);
    /* Get variable node's name*/
    GET_IR_NODE(squeeze1_input);
    GET_IR_NODE(squeeze1_out);
    GET_IR_NODE(act1_out);
    GET_IR_NODE(squeeze2_input);
    GET_IR_NODE(squeeze2_out);
    GET_IR_NODE(elementwise_out);
    GET_IR_NODE(act2_out);

    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    std::unordered_set<const Node*> delete_nodes;
    // Judge squeeze1 && squeeze2 op shape is same or not, if axes is same, the
    // shape is same too.
    std::vector<int> squeeze1_axes =
        PADDLE_GET_CONST(std::vector<int>, squeeze1->Op()->GetAttr("axes"));
    std::vector<int> squeeze2_axes =
        PADDLE_GET_CONST(std::vector<int>, squeeze2->Op()->GetAttr("axes"));
    bool elimination = (squeeze1_axes == squeeze2_axes);
    if (!elimination) return;
    // act1
    auto act1_op_desc = act1->Op();
    std::string squeeze1_input_var_name = squeeze1_input->Var()->Name();
    std::string squeeze1_out_var_name = squeeze1_out->Var()->Name();
    act1_op_desc->RenameInput(squeeze1_out_var_name, squeeze1_input_var_name);
    act1_out->Var()->SetShape(squeeze1_input->Var()->GetShape());
    act1_op_desc->Flush();
    IR_NODE_LINK_TO(squeeze1_input, act1);
    // elementwise
    auto elementwise_op_desc = elementwise->Op();
    std::string squeeze2_input_var_name = squeeze2_input->Var()->Name();
    std::string squeeze2_out_var_name = squeeze2_out->Var()->Name();
    elementwise_op_desc->RenameInput(squeeze2_out_var_name,
                                     squeeze2_input_var_name);
    elementwise_out->Var()->SetShape(squeeze2_input->Var()->GetShape());
    elementwise_op_desc->Flush();
    IR_NODE_LINK_TO(squeeze2_input, elementwise);

    std::string act2_out_var_name = act2_out->Var()->Name();
    std::vector<Node*> remove_nodes;
    auto act2_out_link_nodes = act2_out->outputs;
    for (auto out_link_node : act2_out_link_nodes) {
      auto op_desc = out_link_node->Op();
      if (op_desc->Type() == "unsqueeze2") {
        std::vector<int> unsqueeze_axes =
            PADDLE_GET_CONST(std::vector<int>, op_desc->GetAttr("axes"));
        elimination = elimination && (unsqueeze_axes == squeeze1_axes);
        if (elimination) {
          remove_nodes.push_back(out_link_node);
          delete_nodes.insert(out_link_node);
        }
      }
    }
    if (!elimination) return;
    act2_out->Var()->SetShape(elementwise_out->Var()->GetShape());
    for (auto unsqueeze_node : remove_nodes) {
      std::string unsqueeze_out_var_name =
          unsqueeze_node->Op()->Output("Out")[0];
      for (auto unsqueeze_out_node : unsqueeze_node->outputs) {
        // find unsqueeze "Out" var node
        if (unsqueeze_out_node->Name() == unsqueeze_out_var_name) {
          // Do delete operation
          delete_nodes.insert(unsqueeze_out_node);
          for (auto next_node : unsqueeze_out_node->outputs) {
            auto next_op_desc = next_node->Op();
            next_op_desc->RenameInput(unsqueeze_out_var_name,
                                      act2_out_var_name);
            next_op_desc->Flush();
            IR_NODE_LINK_TO(act2_out, next_node);
          }
        }
      }
    }

    if (elimination) {
      delete_nodes.insert(squeeze1);
      delete_nodes.insert(squeeze2);
      delete_nodes.insert(squeeze1_out);
      delete_nodes.insert(squeeze2_out);
    }
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };
  gpd(graph, handler);
  return found_subgraph_count;
}

class RedundantSqueezeUnsqueezeEliminationPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"redundant_squeeze_unsqueeze_elimination_pass"};
};

void RedundantSqueezeUnsqueezeEliminationPass::ApplyImpl(
    ir::Graph* graph) const {
  VLOG(4) << "handle redundant squeeze unsqueeze elimination.";
  SqueezeActivationUnsqueezeEliminationPass
      squeeze_activation_unsqueeze_elimination_pass;
  squeeze_activation_unsqueeze_elimination_pass.ApplyImpl(graph);
  CustomSqueezeUnsqueezeEliminationPass
      custom_squeeze_unsqueeze_elimination_pass;
  custom_squeeze_unsqueeze_elimination_pass.ApplyImpl(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(redundant_squeeze_unsqueeze_elimination_pass,
              paddle::framework::ir::RedundantSqueezeUnsqueezeEliminationPass);
REGISTER_PASS_CAPABILITY(redundant_squeeze_unsqueeze_elimination_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("squeeze2", 0)
            .LE("leaky_relu", 1)
            .EQ("unsqueeze2", 0));
