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
fuse ele_mul + ele_add
For example:
graph:
  mul_x         Y            mul_y
    \        /    \          /
     \     scale   \        /
      \    /        \      /
      ele_mul        ele_mul
          \          /
           \        /
            ele_add
------------------------------------------------------
After the pass is applied:
  mul_x       Y    mul_y
    \        /     /
     \      /     /
      \    /     /
      ele_mul_add
*/
struct EltMulAddXPUPattern : public PatternBase {
  EltMulAddXPUPattern(PDPattern* pattern, const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(ele_add);
  PATTERN_DECL_NODE(ele_mul_l);
  PATTERN_DECL_NODE(ele_mul_r);
  // declare variable node's name
  // scale
  PATTERN_DECL_NODE(scale_in);
  PATTERN_DECL_NODE(scale_out);
  // mul1
  PATTERN_DECL_NODE(mul1_x);
  PATTERN_DECL_NODE(mul1_out);
  // mul2
  PATTERN_DECL_NODE(mul2_y);
  PATTERN_DECL_NODE(mul2_out);
  // add
  PATTERN_DECL_NODE(add_out);
};

EltMulAddXPUPattern::EltMulAddXPUPattern(PDPattern* pattern,
                                         const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  // scale
  auto scale_in = pattern->NewNode(scale_in_repr())
                      ->assert_is_op_input("scale", "X")
                      ->assert_is_op_input("elementwise_mul", "X");
  auto scale_out = pattern->NewNode(scale_out_repr())
                       ->assert_is_op_output("scale", "Out")
                       ->assert_is_op_input("elementwise_mul", "X");
  auto scale =
      pattern->NewNode(scale_repr())
          ->assert_is_op("scale")
          ->assert_more([](Node* n) {
            auto bias_val = PADDLE_GET_CONST(float, n->Op()->GetAttr("bias"));
            auto scale_val = PADDLE_GET_CONST(float, n->Op()->GetAttr("scale"));
            return bias_val == 1 && scale_val == -1;
          });
  // mul1
  auto mul1_x = pattern->NewNode(mul1_x_repr())
                    ->assert_is_op_input("elementwise_mul", "Y");
  auto ele_mul_l = pattern->NewNode(ele_mul_l_repr())
                       ->assert_is_op("elementwise_mul")
                       ->assert_more([](Node* node) {
                         auto node1 = node->inputs[0];
                         auto node2 = node->inputs[1];
                         auto node1_shape = node1->Var()->GetShape();
                         auto node2_shape = node2->Var()->GetShape();
                         if (node1_shape.size() != node2_shape.size() &&
                             node1_shape.size() != 2)
                           return false;
                         for (size_t i = 0; i < node1_shape.size(); i++) {
                           if (node1_shape[i] != node2_shape[i] &&
                               (node1_shape[i] != 1 && node2_shape[i] != 1)) {
                             return false;
                           }
                         }
                         return true;
                       });
  auto mul1_out = pattern->NewNode(mul1_out_repr())
                      ->assert_is_op_output("elementwise_mul", "Out")
                      ->assert_is_op_input("elementwise_add", "X");
  // mul2
  auto mul2_y = pattern->NewNode(mul2_y_repr())
                    ->assert_is_op_input("elementwise_mul", "Y");
  auto ele_mul_r = pattern->NewNode(ele_mul_r_repr())
                       ->assert_is_op("elementwise_mul")
                       ->assert_more([](Node* node) {
                         auto node1 = node->inputs[0];
                         auto node2 = node->inputs[1];
                         auto node1_shape = node1->Var()->GetShape();
                         auto node2_shape = node2->Var()->GetShape();
                         if (node1_shape.size() != node2_shape.size() &&
                             node1_shape.size() != 2)
                           return false;
                         for (size_t i = 0; i < node1_shape.size(); i++) {
                           if (node1_shape[i] != node2_shape[i] &&
                               (node1_shape[i] != 1 && node2_shape[i] != 1)) {
                             return false;
                           }
                         }
                         return true;
                       });
  auto mul2_out = pattern->NewNode(mul2_out_repr())
                      ->assert_is_op_output("elementwise_mul", "Out")
                      ->assert_is_op_input("elementwise_add", "Y");
  // add
  auto ele_add =
      pattern->NewNode(ele_add_repr())
          ->assert_is_op("elementwise_add")
          ->assert_more([](Node* node) {
            // ele_add ->in -> ele_mul -> in
            auto node_in1 = node->inputs[0];
            auto node_in2 = node->inputs[1];
            if (node_in1->inputs.size() == 1 &&
                node_in1->inputs[0]->Op()->Type() == "elementwise_mul" &&
                node_in2->inputs.size() == 1 &&
                node_in2->inputs[0]->Op()->Type() == "elementwise_mul") {
              auto mul1 = node_in1->inputs[0];
              auto mul2 = node_in2->inputs[0];
              auto shape1 = mul1->inputs[1]->Var()->GetShape();
              auto shape2 = mul2->inputs[1]->Var()->GetShape();
              return shape1 == shape2;
            }
            return false;
          });
  auto add_out = pattern->NewNode(add_out_repr())
                     ->assert_is_op_output("elementwise_add", "Out");

  scale->LinksFrom({scale_in}).LinksTo({scale_out});
  ele_mul_l->LinksFrom({mul1_x, scale_out}).LinksTo({mul1_out});
  ele_mul_r->LinksFrom({scale_in, mul2_y}).LinksTo({mul2_out});
  ele_add->LinksFrom({mul1_out, mul2_out}).LinksTo({add_out});
}
}  // namespace patterns

class EltMulAddXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl_(ir::Graph* graph) const;

  const std::string name_scope_{"elementwise_mul_add_xpu_fuse_pass"};
};

void EltMulAddXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  int found_subgraph_count = 0;
  found_subgraph_count += ApplyImpl_(graph);
  AddStatis(found_subgraph_count);
}

int EltMulAddXPUFusePass::ApplyImpl_(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::EltMulAddXPUPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle EltMulAddXPUFusePass fuse";
    // declare operator node's name
    GET_IR_NODE(scale);
    GET_IR_NODE(ele_add);
    GET_IR_NODE(ele_mul_l);
    GET_IR_NODE(ele_mul_r);
    // declare variable node's name
    // scale
    GET_IR_NODE(scale_in);
    GET_IR_NODE(scale_out);
    // mul1
    GET_IR_NODE(mul1_x);
    GET_IR_NODE(mul1_out);
    // mul2
    GET_IR_NODE(mul2_y);
    GET_IR_NODE(mul2_out);
    // add
    GET_IR_NODE(add_out);

    auto* block = ele_add->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));
    framework::OpDesc mul_add_op_desc(block);
    mul_add_op_desc.SetType("elementwise_mul_mul_add_xpu");
    mul_add_op_desc.SetInput("X", {mul1_x->Name()});
    mul_add_op_desc.SetInput("Y", {mul2_y->Name()});
    mul_add_op_desc.SetInput("Z", {scale_in->Name()});
    mul_add_op_desc.SetOutput("Out", {add_out->Name()});
    auto mul_add_op_xpu = graph->CreateOpNode(&mul_add_op_desc);
    IR_NODE_LINK_TO(mul1_x, mul_add_op_xpu);
    IR_NODE_LINK_TO(mul2_y, mul_add_op_xpu);
    IR_NODE_LINK_TO(scale_in, mul_add_op_xpu);
    IR_NODE_LINK_TO(mul_add_op_xpu, add_out);
    std::unordered_set<const Node*> delete_nodes = {
        scale, scale_out, ele_mul_l, ele_mul_r, ele_add, mul1_out, mul2_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(elementwise_mul_add_xpu_fuse_pass,
              paddle::framework::ir::EltMulAddXPUFusePass);

REGISTER_PASS_CAPABILITY(elementwise_mul_add_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "elementwise_mul_mul_add_xpu", 0));
