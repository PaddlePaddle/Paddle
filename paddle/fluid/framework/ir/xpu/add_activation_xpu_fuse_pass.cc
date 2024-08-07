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

/*
fuse ele_add + activation block in to xpu_ele_fusion op
For example:
graph:
                    ele_x
                      |
                      |
                 elementwise_add -----ele_y
                      |
                      |
                     act
                      |
                      |
                    out_Out
------------------------------------------------------
After the pass is applied:
                    Input
                      |     ele_y
                      |    /
                      |   /
  Input_max ---- add_act_fusion ---- ele_y_max
                      |    \
                      |     \
                      |      OutputMax
                    Output
*/
struct AddActXPUPattern : public PatternBase {
  AddActXPUPattern(PDPattern* pattern,
                   const std::string& name_scope,
                   const std::string& act_type);
  // declare operator node's name
  PATTERN_DECL_NODE(ele_add);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(ele_x);
  PATTERN_DECL_NODE(ele_y);
  PATTERN_DECL_NODE(ele_out);
  PATTERN_DECL_NODE(act_out);

 private:
  std::string act_type_;
};

AddActXPUPattern::AddActXPUPattern(PDPattern* pattern,
                                   const std::string& name_scope,
                                   const std::string& act_type)
    : PatternBase(pattern, name_scope, name_scope), act_type_(act_type) {
  auto ele_add =
      pattern->NewNode(ele_add_repr())->assert_is_op("elementwise_add");
  auto ele_x = pattern->NewNode(ele_x_repr())
                   ->assert_is_op_input("elementwise_add", "X")
                   ->assert_var_not_persistable()
                   ->AsInput();
  auto ele_y = pattern->NewNode(ele_y_repr())
                   ->assert_is_op_input("elementwise_add", "Y")
                   ->assert_var_not_persistable()
                   ->AsInput();
  auto ele_out = pattern->NewNode(ele_out_repr())
                     ->assert_is_op_output("elementwise_add", "Out")
                     ->assert_has_n_outputs(1);
  ele_add->LinksFrom({ele_x, ele_y}).LinksTo({ele_out});
  ele_out->assert_is_op_input(act_type_, "X");
  auto act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
  auto act_out =
      pattern->NewNode(act_out_repr())->assert_is_op_output(act_type_, "Out");
  act->LinksFrom({ele_out}).LinksTo({act_out});
}

}  // namespace patterns

class AddActXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph, const std::string& act_type) const;

  const std::string name_scope_{"add_activation_xpu_fuse_pass"};
};

void AddActXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto act_type : {"relu", "gelu"}) {
    found_subgraph_count += ApplyImpl(graph, act_type);
  }
  AddStatis(found_subgraph_count);
}

int AddActXPUFusePass::ApplyImpl(ir::Graph* graph,
                                 const std::string& act_type) const {
  GraphPatternDetector gpd;
  patterns::AddActXPUPattern pattern(
      gpd.mutable_pattern(), name_scope_, act_type);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle AddActXPUFusePass fuse";
    /* declare operator node's name */
    GET_IR_NODE(ele_add);
    GET_IR_NODE(act);
    /* declare variable node's name*/
    GET_IR_NODE(ele_x);
    GET_IR_NODE(ele_y);
    GET_IR_NODE(ele_out);
    GET_IR_NODE(act_out);
    auto* block = ele_add->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    std::string fused_op_out_name;
    fused_op_out_name = act_out->Name();
    std::string fused_op_out_max_name = fused_op_out_name + "_max";
    VarDesc fused_op_out_max_desc(fused_op_out_max_name);
    Node* fused_op_out_max = graph->CreateVarNode(&fused_op_out_max_desc);
    // Generate add_act fused op
    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("add_act_xpu");
    // set attrs for fused op
    fused_op_desc.SetAttr("act_type", ConvertActivationType(act_type));
    fused_op_desc.SetInput("x", {ele_x->Name()});
    fused_op_desc.SetInput("y", {ele_y->Name()});
    fused_op_desc.SetOutput("out", {fused_op_out_name});
    fused_op_desc.SetOutput("out_max", {fused_op_out_max_name});
    // relink fused op
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(ele_x, fused_op);
    IR_NODE_LINK_TO(ele_y, fused_op);
    IR_NODE_LINK_TO(fused_op, act_out);
    IR_NODE_LINK_TO(fused_op, fused_op_out_max);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {ele_add, act, ele_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(add_activation_xpu_fuse_pass,
              paddle::framework::ir::AddActXPUFusePass);

REGISTER_PASS_CAPABILITY(add_activation_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "add_act_xpu", 0));
