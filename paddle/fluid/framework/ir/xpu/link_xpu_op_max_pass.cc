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

struct FusionXPUOpPattern : public PatternBase {
  FusionXPUOpPattern(PDPattern* pattern,
                     const std::string& name_scope,
                     const std::string& op_type,
                     bool with_branch);

  // declare operator node's name
  PATTERN_DECL_NODE(fusion_op);
  // declare variable node's name
  PATTERN_DECL_NODE(input);
  PATTERN_DECL_NODE(branch);

 private:
  std::string op_type_;
  bool with_branch_{false};
};

FusionXPUOpPattern::FusionXPUOpPattern(PDPattern* pattern,
                                       const std::string& name_scope,
                                       const std::string& op_type,
                                       bool with_branch)
    : PatternBase(pattern, name_scope, name_scope),
      op_type_(op_type),
      with_branch_(with_branch) {
  auto* fusion_op = pattern->NewNode(fusion_op_repr())->assert_is_op(op_type_);
  auto* input =
      pattern->NewNode(input_repr())->assert_is_op_input(op_type_, "x");

  PDNode* branch = nullptr;
  if (with_branch_) {
    branch =
        pattern->NewNode(branch_repr())->assert_is_op_input(op_type_, "branch");
    fusion_op->LinksFrom({input, branch});
  } else {
    fusion_op->LinksFrom({input});
  }
}

}  // namespace patterns

class LinkXPUOpMaxPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void ApplyImpl(ir::Graph* graph,
                 const std::string& op_type,
                 bool with_branch) const;

  const std::string name_scope_{"link_xpu_op_max_pass"};
  // ops with x_max/out_max
  std::set<std::string> op_types_{"fc_xpu", "conv2d_xpu"};
};

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
*/
void LinkXPUOpMaxPass::ApplyImpl(ir::Graph* graph) const {
  Init(name_scope_, graph);
  for (auto op_type : op_types_) {
    for (auto with_branch : {true, false}) {
      ApplyImpl(graph, op_type, with_branch);
    }
  }
}

void LinkXPUOpMaxPass::ApplyImpl(ir::Graph* graph,
                                 const std::string& op_type,
                                 bool with_branch) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  GraphPatternDetector gpd;
  patterns::FusionXPUOpPattern pattern(
      gpd.mutable_pattern(), name_scope_, op_type, with_branch);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle LinkXPUOpMaxPass fuse";
    GET_IR_NODE(fusion_op);
    GET_IR_NODE(input);
    GET_IR_NODE(branch);

    auto* fusion_op_desc = fusion_op->Op();
    if (fusion_op_desc->HasAttr("has_branch")) {
      bool fusion_op_branch =
          PADDLE_GET_CONST(bool, fusion_op_desc->GetAttr("has_branch"));
      if (fusion_op_branch != with_branch) {
        return;
      }
    }
    if (input->inputs.size() > 0 && input->inputs[0]->IsOp() &&
        input->inputs[0]->Op()->HasOutput("out_max")) {
      auto input_max_name = input->inputs[0]->Op()->Output("out_max");
      for (auto max_node : input->inputs[0]->outputs) {
        if (input_max_name[0] == max_node->Name()) {
          fusion_op_desc->SetInput("x_max", {max_node->Name()});
          IR_NODE_LINK_TO(max_node, fusion_op);
          found_subgraph_count++;
        }
      }
    }

    if (with_branch) {
      if (branch->inputs.size() > 0 && branch->inputs[0]->IsOp() &&
          branch->inputs[0]->Op()->HasOutput("out_max")) {
        auto branch_max_name = branch->inputs[0]->Op()->Output("out_max");
        for (auto max_node : branch->inputs[0]->outputs) {
          if (branch_max_name[0] == max_node->Name()) {
            fusion_op_desc->SetInput("branch_max", {max_node->Name()});
            IR_NODE_LINK_TO(max_node, fusion_op);
            found_subgraph_count++;
          }
        }
      }
    }
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(link_xpu_op_max_pass, paddle::framework::ir::LinkXPUOpMaxPass);

REGISTER_PASS_CAPABILITY(link_xpu_op_max_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "fc_xpu", 0));
