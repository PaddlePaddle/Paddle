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
/*
Origin subgraph:
          fusion_xpu_op0
            /       \
            |       |
          out0   out0_max
            |
            \
            fusion_op
Fused subgraph:
          fusion_xpu_op0
            /       \
            |       |
          out0   out0_max
            |       |
            \       /
            fusion_op

Origin subgraph1:
          fusion_xpu_op0     fusion_xpu_op1
            /       \         /          \
            |       |         |          |
          out0   out0_max    out1      out1_max
            |                 |
        (x) \                / (branch)
              fusion_xpu_op2
Fused subgraph1:
          fusion_xpu_op0     fusion_xpu_op1
            /       \         /           \
            |       |         |            |
          out0   out0_max    out1      out1_max
            |       |          |           |
        (x) \       |(x_max)   |(branch)  /(branch_max)
             \      |          |         /
              \     |          |        /
               \    |          |       /
                   fusion_xpu_op2

Origin subgraph2:
          fusion_xpu_op0     fusion_xpu_op1
            /       \         /          \
            |       |         |          |
          out0   out0_max    out1      out1_max
            |                 |
        (x) \                / (y)
              fusion_xpu_op2
Fused subgraph2:
          fusion_xpu_op0     fusion_xpu_op1
            /       \         /           \
            |       |         |            |
          out0   out0_max    out1      out1_max
            |       |          |           |
        (x) \       |(x_max)   |(y)  /(y_max)
             \      |          |         /
              \     |          |        /
               \    |          |       /
                   fusion_xpu_op2
*/
struct FusionXPUOpPattern : public PatternBase {
  FusionXPUOpPattern(PDPattern* pattern,
                     const std::string& name_scope,
                     const std::string& op_type,
                     bool with_branch,
                     bool with_ele_y);

  // declare operator node's name
  PATTERN_DECL_NODE(fusion_op);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(branch);
  PATTERN_DECL_NODE(ele_y);

 private:
  std::string op_type_;
  bool with_branch_{false};
  bool with_ele_y_{false};
};

FusionXPUOpPattern::FusionXPUOpPattern(PDPattern* pattern,
                                       const std::string& name_scope,
                                       const std::string& op_type,
                                       bool with_branch,
                                       bool with_ele_y)
    : PatternBase(pattern, name_scope, name_scope),
      op_type_(op_type),
      with_branch_(with_branch),
      with_ele_y_(with_ele_y) {
  auto* fusion_op = pattern->NewNode(fusion_op_repr())->assert_is_op(op_type_);
  auto* x = pattern->NewNode(x_repr())->assert_is_op_input(op_type_, "x");

  PDNode* branch = nullptr;
  PDNode* ele_y = nullptr;
  if (with_branch_) {
    branch =
        pattern->NewNode(branch_repr())->assert_is_op_input(op_type_, "branch");
    fusion_op->LinksFrom({x, branch});
  } else if (with_ele_y_) {
    ele_y = pattern->NewNode(ele_y_repr())->assert_is_op_input(op_type_, "y");
    fusion_op->LinksFrom({x, ele_y});
  } else {
    fusion_op->LinksFrom({x});
  }
}

}  // namespace patterns

class LinkXPUOpMaxPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                const std::string& op_type,
                bool with_branch,
                bool with_ele_y) const;

  const std::string name_scope_{"link_xpu_op_max_pass"};
};

void LinkXPUOpMaxPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  // ops with x_max/y_max/branch_max
  for (auto op_type : {"fc_xpu", "conv2d_xpu", "add_act_xpu"}) {
    for (auto with_branch : {true, false}) {
      for (auto with_ele_y : {true, false}) {
        if (with_branch && with_ele_y) continue;
        found_subgraph_count +=
            ApplyImpl(graph, op_type, with_branch, with_ele_y);
      }
    }
  }
  AddStatis(found_subgraph_count);
}

int LinkXPUOpMaxPass::ApplyImpl(ir::Graph* graph,
                                const std::string& op_type,
                                bool with_branch,
                                bool with_ele_y) const {
  GraphPatternDetector gpd;
  patterns::FusionXPUOpPattern pattern(
      gpd.mutable_pattern(), name_scope_, op_type, with_branch, with_ele_y);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle LinkXPUOpMaxPass fuse";
    /* declare operator node's name */
    GET_IR_NODE(fusion_op);
    /* declare variable node's name*/
    GET_IR_NODE(x);
    GET_IR_NODE(branch);
    GET_IR_NODE(ele_y);
    auto* fusion_op_desc = fusion_op->Op();
    if (fusion_op_desc->HasInput("x_max")) {
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
      if (with_branch && fusion_op_desc->HasInput("branch_max")) {
        auto* branch_pre_op = branch->inputs[0]->Op();
        if (branch->inputs.size() > 0 && branch->inputs[0]->IsOp() &&
            branch_pre_op->HasOutput("out_max")) {
          auto preop_max_var_name = branch_pre_op->Output("out_max");
          for (auto max_node : branch->inputs[0]->outputs) {
            if (preop_max_var_name[0] == max_node->Name()) {
              fusion_op_desc->SetInput("branch_max", {max_node->Name()});
              IR_NODE_LINK_TO(max_node, fusion_op);
            }
          }
        }
      } else if (with_ele_y && fusion_op_desc->HasInput("y_max")) {
        auto* ele_y_pre_op = ele_y->inputs[0]->Op();
        if (ele_y->inputs.size() > 0 && ele_y->inputs[0]->IsOp() &&
            ele_y_pre_op->HasOutput("out_max")) {
          auto preop_max_var_name = ele_y_pre_op->Output("out_max");
          for (auto max_node : ele_y->inputs[0]->outputs) {
            if (preop_max_var_name[0] == max_node->Name()) {
              fusion_op_desc->SetInput("y_max", {max_node->Name()});
              IR_NODE_LINK_TO(max_node, fusion_op);
            }
          }
        }
      }
      found_subgraph_count++;
    }
  };

  gpd(graph, handler);
  return found_subgraph_count;
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
