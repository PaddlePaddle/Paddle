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

#include "paddle/fluid/framework/ir/xpu/link_xpu_op_max_pass.h"
#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
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
struct LinkAddActPattern : public PatternBase {
  LinkAddActPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(fusion_op);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(ele_y);
};

LinkAddActPattern::LinkAddActPattern(PDPattern* pattern,
                                     const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* fusion_op =
      pattern->NewNode(fusion_op_repr())->assert_is_op("add_act_xpu");
  auto* x = pattern->NewNode(x_repr())->assert_is_op_input("add_act_xpu", "x");
  auto* ele_y =
      pattern->NewNode(ele_y_repr())->assert_is_op_input("add_act_xpu", "y");
  fusion_op->LinksFrom({x, ele_y});
}

struct LinkConv2dPattern : public PatternBase {
  LinkConv2dPattern(PDPattern* pattern,
                    const std::string& name_scope,
                    bool with_branch);

  // declare operator node's name
  PATTERN_DECL_NODE(fusion_op);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(filter);
  PATTERN_DECL_NODE(branch);

 private:
  bool with_branch_{false};
};

LinkConv2dPattern::LinkConv2dPattern(PDPattern* pattern,
                                     const std::string& name_scope,
                                     bool with_branch)
    : PatternBase(pattern, name_scope, name_scope), with_branch_(with_branch) {
  auto* fusion_op =
      pattern->NewNode(fusion_op_repr())->assert_is_op("conv2d_xpu");

  auto* x = pattern->NewNode(x_repr())->assert_is_op_input("conv2d_xpu", "x");
  auto* filter = pattern->NewNode(filter_repr())
                     ->assert_is_op_input("conv2d_xpu", "filter")
                     ->assert_is_persistable_var();
  PDNode* branch = nullptr;
  if (with_branch_) {
    branch = pattern->NewNode(branch_repr())
                 ->assert_is_op_input("conv2d_xpu", "branch");
    fusion_op->LinksFrom({x, branch, filter});
  } else {
    fusion_op->LinksFrom({x, filter});
  }
}

struct LinkFcPattern : public PatternBase {
  LinkFcPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(fusion_op);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(w);
};

LinkFcPattern::LinkFcPattern(PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* fusion_op = pattern->NewNode(fusion_op_repr())->assert_is_op("fc_xpu");

  auto* x = pattern->NewNode(x_repr())->assert_is_op_input("fc_xpu", "x");
  auto* w = pattern->NewNode(w_repr())
                ->assert_is_op_input("fc_xpu", "w")
                ->assert_is_persistable_var();
  fusion_op->LinksFrom({x, w});
}

}  // namespace patterns

bool LinkXPUOpMaxPass::IsQuant(Node* weight_node) const {
  auto w_dtype = param_scope()
                     ->FindVar(weight_node->Name())
                     ->GetMutable<phi::DenseTensor>()
                     ->dtype();
  return w_dtype == phi::DataType::INT8;
}

void LinkXPUOpMaxPass::LinkAddActMax(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::LinkAddActPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle LinkAddActMax";
    /* declare operator node's name */
    GET_IR_NODE(fusion_op);
    /* declare variable node's name*/
    GET_IR_NODE(x);
    GET_IR_NODE(ele_y);
    auto* fusion_op_desc = fusion_op->Op();
    auto* x_pre_op = x->inputs[0]->Op();
    if (x->inputs.size() > 0 && x->inputs[0]->IsOp() &&
        x_pre_op->HasOutput("out_max")) {
      auto preop_max_var_name = x_pre_op->Output("out_max");
      for (auto max_node : x->inputs[0]->outputs) {
        if (preop_max_var_name[0] == max_node->Name()) {
          fusion_op_desc->SetInput("x_max", {max_node->Name()});
          IR_NODE_LINK_TO(max_node, fusion_op);
        }
      }
    }
    if (ele_y->inputs.size() > 0 && ele_y->inputs[0]->IsOp() &&
        ele_y->inputs[0]->Op()->HasOutput("out_max")) {
      auto preop_max_var_name = ele_y->inputs[0]->Op()->Output("out_max");
      for (auto max_node : ele_y->inputs[0]->outputs) {
        if (preop_max_var_name[0] == max_node->Name()) {
          fusion_op_desc->SetInput("y_max", {max_node->Name()});
          IR_NODE_LINK_TO(max_node, fusion_op);
        }
      }
    }
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void LinkXPUOpMaxPass::LinkConv2dMax(ir::Graph* graph, bool with_branch) const {
  GraphPatternDetector gpd;
  patterns::LinkConv2dPattern pattern(
      gpd.mutable_pattern(), name_scope_, with_branch);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle LinkConv2dMax";
    /* get operator node's name */
    GET_IR_NODE(fusion_op);
    /* get variable node's name*/
    GET_IR_NODE(x);
    GET_IR_NODE(filter);
    GET_IR_NODE(branch);
    if (IsQuant(filter)) {
      return;
    }
    auto* fusion_op_desc = fusion_op->Op();
    bool fusion_op_has_branch = fusion_op_desc->HasInput("branch");
    if (fusion_op_has_branch) {
      if (fusion_op_has_branch != with_branch) {
        return;
      }
    }
    auto* x_pre_op = x->inputs[0]->Op();
    if (x->inputs.size() > 0 && x->inputs[0]->IsOp() &&
        x_pre_op->HasOutput("out_max")) {
      auto preop_max_var_name = x_pre_op->Output("out_max");
      for (auto max_node : x->inputs[0]->outputs) {
        if (preop_max_var_name[0] == max_node->Name()) {
          if (fusion_op_desc->HasInput("x_max")) {
            auto x_max_old_name = fusion_op_desc->Input("x_max")[0];
            fusion_op_desc->RenameInput(x_max_old_name, max_node->Name());
          } else {
            fusion_op_desc->SetInput("x_max", {max_node->Name()});
          }
          IR_NODE_LINK_TO(max_node, fusion_op);
        }
      }
    }
    if (with_branch) {
      if (branch->inputs.size() > 0 && branch->inputs[0]->IsOp() &&
          branch->inputs[0]->Op()->HasOutput("out_max")) {
        auto preop_max_var_name = branch->inputs[0]->Op()->Output("out_max");
        for (auto max_node : branch->inputs[0]->outputs) {
          if (preop_max_var_name[0] == max_node->Name()) {
            fusion_op_desc->SetInput("branch_max", {max_node->Name()});
            IR_NODE_LINK_TO(max_node, fusion_op);
          }
        }
      }
    }
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void LinkXPUOpMaxPass::LinkFcMax(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::LinkFcPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle LinkFcMax";
    /* get operator node's name */
    GET_IR_NODE(fusion_op);
    /* get variable node's name*/
    GET_IR_NODE(x);
    GET_IR_NODE(w);

    if (IsQuant(w)) return;
    auto* fusion_op_desc = fusion_op->Op();
    if (x->inputs.size() > 0) {
      auto* x_pre_op = x->inputs[0]->Op();
      if (x->inputs[0]->IsOp() && x_pre_op->HasOutput("out_max")) {
        auto preop_max_var_name = x_pre_op->Output("out_max");
        for (auto max_node : x->inputs[0]->outputs) {
          if (preop_max_var_name[0] == max_node->Name()) {
            if (fusion_op_desc->HasInput("x_max")) {
              auto x_max_old_name = fusion_op_desc->Input("x_max")[0];
              fusion_op_desc->RenameInput(x_max_old_name, max_node->Name());
            } else {
              fusion_op_desc->SetInput("x_max", {max_node->Name()});
            }
            IR_NODE_LINK_TO(max_node, fusion_op);
          }
        }
      }
    }
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void LinkXPUOpMaxPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  LinkFcMax(graph);
  for (auto with_branch : {true, false}) {
    LinkConv2dMax(graph, with_branch);
  }
  LinkAddActMax(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(link_xpu_op_max_pass, paddle::framework::ir::LinkXPUOpMaxPass);

REGISTER_PASS_CAPABILITY(link_xpu_op_max_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("fc_xpu", 0)
            .EQ("conv2d_xpu", 0)
            .EQ("add_act_xpu", 0));
