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
                    add_x
                      |
                 elementwise_add -----add_y
                      |
                  layernorm
                      |
                    output
------------------------------------------------------
After the pass is applied:
                    add_x
                      |     add_y
                      |    /
                      |   /
   scale---- add_layernorm_fusion ---- bias
                /     |    \     \
               /      |     \     \
          variance    |      mean  z_add
                    Output
*/
struct AddLayernormXPUPattern : public PatternBase {
  AddLayernormXPUPattern(PDPattern* pattern, const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(ele_add);
  PATTERN_DECL_NODE(l_norm);
  // declare variable node's name
  PATTERN_DECL_NODE(add_x);
  PATTERN_DECL_NODE(add_y);
  PATTERN_DECL_NODE(ele_out);
  PATTERN_DECL_NODE(norm_bias);
  PATTERN_DECL_NODE(norm_scale);
  PATTERN_DECL_NODE(norm_mean);
  PATTERN_DECL_NODE(norm_variance);
  PATTERN_DECL_NODE(norm_out);
};

AddLayernormXPUPattern::AddLayernormXPUPattern(PDPattern* pattern,
                                               const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto ele_add =
      pattern->NewNode(ele_add_repr())->assert_is_op("elementwise_add");
  auto add_x = pattern->NewNode(add_x_repr())
                   ->assert_is_op_input("elementwise_add", "X")
                   ->AsInput();
  auto add_y = pattern->NewNode(add_y_repr())
                   ->assert_is_op_input("elementwise_add", "Y")
                   ->AsInput();
  auto ele_out = pattern->NewNode(ele_out_repr())
                     ->assert_is_op_output("elementwise_add", "Out")
                     ->assert_is_op_input("layer_norm", "X")
                     ->assert_has_n_outputs(1);
  ele_add->LinksFrom({add_x, add_y}).LinksTo({ele_out});
  auto l_norm = pattern->NewNode(l_norm_repr())->assert_is_op("layer_norm");
  auto norm_bias = pattern->NewNode(norm_bias_repr())
                       ->AsInput()
                       ->assert_is_persistable_var()
                       ->assert_is_op_input("layer_norm", "Bias");
  auto norm_scale = pattern->NewNode(norm_scale_repr())
                        ->AsInput()
                        ->assert_is_persistable_var()
                        ->assert_is_op_input("layer_norm", "Scale");
  auto norm_mean = pattern->NewNode(norm_mean_repr())
                       ->AsOutput()
                       ->assert_is_op_output("layer_norm", "Mean")
                       ->assert_has_n_outputs(0);
  auto norm_variance = pattern->NewNode(norm_variance_repr())
                           ->AsOutput()
                           ->assert_is_op_output("layer_norm", "Variance")
                           ->assert_has_n_outputs(0);
  auto norm_out = pattern->NewNode(norm_out_repr())
                      ->AsOutput()
                      ->assert_is_op_output("layer_norm", "Y");
  l_norm->LinksFrom({ele_out, norm_bias, norm_scale})
      .LinksTo({norm_out, norm_mean, norm_variance});
}

}  // namespace patterns

namespace {
void setIntermediateOut(OpDesc* desc,
                        const std::string& out_name,
                        const std::string& scope_name) {
  std::string new_name = scope_name + "/at." + out_name + ".new";
  desc->SetOutput(out_name, {new_name});
}

void addIntermediateOut(Node* op_node,
                        const std::string& out_name,
                        const std::string& scope_name,
                        Graph* graph) {
  std::string new_name = scope_name + "/at." + out_name + ".new";
  VarDesc out_var(new_name);
  out_var.SetPersistable(false);
  auto* node_var = graph->CreateVarNode(&out_var);
  IR_NODE_LINK_TO(op_node, node_var);
}

}  // namespace

class AddLayernormXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void FuseAddLayernorm(ir::Graph* graph) const;

  const std::string name_scope_{"add_layernorm_xpu_fuse_pass"};
};

void AddLayernormXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FuseAddLayernorm(graph);
}

void AddLayernormXPUFusePass::FuseAddLayernorm(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::AddLayernormXPUPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle AddLayernormXPUFusePass fuse";
    // declare operator node's name
    GET_IR_NODE(ele_add);
    GET_IR_NODE(l_norm);
    // declare variable node's name
    GET_IR_NODE(add_x);
    GET_IR_NODE(add_y);
    GET_IR_NODE(ele_out);
    GET_IR_NODE(norm_bias);
    GET_IR_NODE(norm_scale);
    GET_IR_NODE(norm_mean);
    GET_IR_NODE(norm_variance);
    GET_IR_NODE(norm_out);

    auto* block = l_norm->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    auto x_shape = add_x->Var()->GetShape();
    auto x_rank = x_shape.size();
    auto y_shape = add_y->Var()->GetShape();
    auto y_rank = y_shape.size();
    if (x_rank != y_rank) return;
    // delete useless node
    std::unordered_set<const Node*> delete_nodes;

    float eps = PADDLE_GET_CONST(float, l_norm->Op()->GetAttr("epsilon"));
    int begin_norm_axis =
        PADDLE_GET_CONST(int, l_norm->Op()->GetAttr("begin_norm_axis"));

    std::string fused_op_out_name;
    fused_op_out_name = norm_out->Name();
    // Generate add_layernorm fused op
    framework::OpDesc fused_op_desc(block);

    fused_op_desc.SetType("add_layernorm_xpu");
    // set attrs for fused op
    fused_op_desc.SetInput("x", {add_x->Name()});
    fused_op_desc.SetInput("y", {add_y->Name()});
    fused_op_desc.SetInput("scale", {norm_scale->Name()});
    fused_op_desc.SetInput("bias", {norm_bias->Name()});
    fused_op_desc.SetAttr("epsilon", eps);
    fused_op_desc.SetAttr("begin_norm_axis", begin_norm_axis);
    fused_op_desc.SetOutput("out", {fused_op_out_name});
    // relink fused op
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(add_x, fused_op);
    IR_NODE_LINK_TO(add_y, fused_op);
    IR_NODE_LINK_TO(norm_scale, fused_op);
    IR_NODE_LINK_TO(norm_bias, fused_op);
    IR_NODE_LINK_TO(fused_op, norm_out);

    delete_nodes.insert({ele_add, l_norm, ele_out, norm_mean, norm_variance});
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(add_layernorm_xpu_fuse_pass,
              paddle::framework::ir::AddLayernormXPUFusePass);

REGISTER_PASS_CAPABILITY(add_layernorm_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "add_layernorm_xpu", 0));
