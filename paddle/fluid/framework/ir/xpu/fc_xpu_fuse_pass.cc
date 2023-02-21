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

struct FcXPUPattern : public PatternBase {
  FcXPUPattern(PDPattern* pattern,
               const std::string& name_scope,
               const std::string& mul_type,
               bool with_bias,
               const std::string& act_type);

  // declare operator node's name
  PATTERN_DECL_NODE(mul);
  PATTERN_DECL_NODE(add);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(mul_x);
  PATTERN_DECL_NODE(mul_w);
  PATTERN_DECL_NODE(mul_out);
  PATTERN_DECL_NODE(bias);
  PATTERN_DECL_NODE(add_out);
  PATTERN_DECL_NODE(act_out);

 private:
  std::string mul_type_;
  bool with_bias_{false};
  std::string act_type_;
};

FcXPUPattern::FcXPUPattern(PDPattern* pattern,
                           const std::string& name_scope,
                           const std::string& mul_type,
                           bool with_bias,
                           const std::string& act_type)
    : PatternBase(pattern, name_scope, name_scope),
      mul_type_(mul_type),
      with_bias_(with_bias),
      act_type_(act_type) {
  auto* mul_x = pattern->NewNode(mul_x_repr())
                    ->assert_is_op_input(mul_type_, "X")
                    ->assert_var_not_persistable();
  auto* mul_w = pattern->NewNode(mul_w_repr())
                    ->assert_is_op_input(mul_type_, "Y")
                    ->assert_is_persistable_var()
                    ->assert_more([](Node* node) {
                      return node->Var()->GetShape().size() == 2;
                    });
  auto* mul =
      pattern->NewNode(mul_repr())
          ->assert_is_op(mul_type_)
          ->assert_more([](Node* node) {
            auto op_type = node->Op()->Type();
            if (op_type == "matmul") {
              return !PADDLE_GET_CONST(bool,
                                       node->Op()->GetAttr("transpose_X"));
            } else if (op_type == "matmul_v2") {
              return !PADDLE_GET_CONST(bool, node->Op()->GetAttr("trans_x"));
            } else {
              return true;
            }
          });
  auto* mul_out = pattern->NewNode(mul_out_repr())
                      ->assert_is_op_output(mul_type_, "Out")
                      ->assert_var_not_persistable();
  mul->LinksFrom({mul_x, mul_w}).LinksTo({mul_out});
  PDNode* bias = nullptr;
  PDNode* add = nullptr;
  PDNode* add_out = nullptr;
  PDNode* act = nullptr;
  PDNode* act_out = nullptr;
  if (with_bias_) {
    mul_out->assert_is_op_input("elementwise_add", "X");
    bias = pattern->NewNode(bias_repr())
               ->assert_is_op_input("elementwise_add", "Y")
               ->assert_is_persistable_var();
    add = pattern->NewNode(add_repr())->assert_is_op("elementwise_add");
    add_out = pattern->NewNode(add_out_repr())
                  ->assert_is_op_output("elementwise_add", "Out")
                  ->assert_var_not_persistable();
    add->LinksFrom({mul_out, bias}).LinksTo({add_out});
  } else {
    add_out = mul_out;
  }
  if (!act_type_.empty()) {
    add_out->assert_is_op_input(act_type_, "X");
    act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
    act_out = pattern->NewNode(act_out_repr())
                  ->assert_is_op_output(act_type_, "Out")
                  ->assert_var_not_persistable();
    act->LinksFrom({add_out}).LinksTo({act_out});
  }
}

}  // namespace patterns

/*
1. fuse mul/matmul/matmul_v2 + add + act into fc_xpu
2. add is optional
3. act is optional

Origin subgraph:
          mul_x  mul_w
             \     /
              \   /
               mul
                |
                |
             mul_out  bias
                \      /
                 \    /
             elementwise_add
                   |
                   |
           elementwise_add_out
                   |
                   |
                  act
                   |
                   |
                act_out

Fused subgraph:
        mul_x mul_w bias mul_w_max
          \     |    /       |
           \    |   /        |
            \   |  /         |
             fc_xpu-----------
                |
                |
             act_out
*/
class FcXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void ApplyImpl(ir::Graph* graph,
                 const std::string& mul_type,
                 bool with_bias,
                 const std::string& act_type) const;

  const std::string name_scope_{"fc_xpu_fuse_pass"};
};

void FcXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  for (auto mul_type : {"mul", "matmul", "matmul_v2"}) {
    for (auto with_bias : {true, false}) {
      for (auto act_type : {
               "relu",
               "gelu",
               "",
           }) {
        ApplyImpl(graph, mul_type, with_bias, act_type);
      }
    }
  }
}

void FcXPUFusePass::ApplyImpl(ir::Graph* graph,
                              const std::string& mul_type,
                              bool with_bias,
                              const std::string& act_type) const {
  GraphPatternDetector gpd;
  patterns::FcXPUPattern pattern(
      gpd.mutable_pattern(), name_scope_, mul_type, with_bias, act_type);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FcXPUFusePass fuse";
    GET_IR_NODE(mul_x);
    GET_IR_NODE(mul_w);
    GET_IR_NODE(mul);
    GET_IR_NODE(mul_out);
    GET_IR_NODE(bias);
    GET_IR_NODE(add);
    GET_IR_NODE(add_out);
    GET_IR_NODE(act);
    GET_IR_NODE(act_out);
    auto* block = mul->Op()->Block();
    auto* scope = param_scope();

    auto mul_w_name = mul_w->Name();
    auto mul_w_tensor =
        scope->FindVar(mul_w_name)->GetMutable<phi::DenseTensor>();
    // 1. Transform weight to int16/int31
    // 2. Avoid transform repeatly, because weight may be shared with other ops.
    // TODO(zhupengyang): support int31
    std::string mul_w_max_name = mul_w_name + "_max";
    Node* mul_w_max = nullptr;
    if (mul_w_tensor->dtype() != phi::DataType::INT16) {
      // Create weight_max node
      VarDesc mul_w_max_desc(mul_w_max_name);
      mul_w_max_desc.SetPersistable(true);
      mul_w_max = graph->CreateVarNode(&mul_w_max_desc);
      // Create weight_max var/tensor
      auto mul_w_max_var = block->Var(mul_w_max_name);
      mul_w_max_var->SetPersistable(true);
      auto mul_w_max_tensor =
          scope->Var(mul_w_max_name)->GetMutable<phi::DenseTensor>();
      bool transpose_w = false;
      if (mul_type == "matmul") {
        transpose_w = PADDLE_GET_CONST(bool, mul->Op()->GetAttr("transpose_Y"));
      } else if (mul_type == "matmul_v2") {
        transpose_w = PADDLE_GET_CONST(bool, mul->Op()->GetAttr("trans_y"));
      }
      QuantWeight<int16_t>(mul_w_tensor, mul_w_max_tensor, !transpose_w);
    }

    // Generate fc_xpu op
    framework::OpDesc fc_xpu_op_desc(block);
    fc_xpu_op_desc.SetType("fc_xpu");
    fc_xpu_op_desc.SetInput("x", {mul_x->Name()});
    fc_xpu_op_desc.SetInput("w", {mul_w->Name()});
    fc_xpu_op_desc.SetInput("w_max", {mul_w_max_name});
    if (bias) {
      fc_xpu_op_desc.SetInput("bias", {bias->Name()});
    }
    fc_xpu_op_desc.SetAttr(
        "in_num_col_dims",
        static_cast<int>(mul_x->Var()->GetShape().size() - 1));
    if (mul_type == "mul") {
      fc_xpu_op_desc.SetAttr(
          "in_num_col_dims",
          PADDLE_GET_CONST(int, mul->Op()->GetAttr("x_num_col_dims")));
    }
    fc_xpu_op_desc.SetAttr("transpose_x", false);
    fc_xpu_op_desc.SetAttr("alpha", 1.f);
    fc_xpu_op_desc.SetAttr("beta", 0.f);
    if (mul_type == "matmul") {
      fc_xpu_op_desc.SetAttr(
          "alpha", PADDLE_GET_CONST(float, mul->Op()->GetAttr("alpha")));
      fc_xpu_op_desc.SetAttr(
          "beta", PADDLE_GET_CONST(float, mul->Op()->GetAttr("beta")));
    }
    fc_xpu_op_desc.SetAttr("act_type", 0);
    fc_xpu_op_desc.SetAttr("act_alpha", 0.f);
    if (act) {
      fc_xpu_op_desc.SetAttr("act_type", ConvertActivationType(act_type));
      if (act_type == "leaky_relu") {
        fc_xpu_op_desc.SetAttr(
            "act_alpha", PADDLE_GET_CONST(float, act->Op()->GetAttr("alpha")));
      } else if (act_type == "hard_sigmoid") {
        fc_xpu_op_desc.SetAttr(
            "act_alpha", PADDLE_GET_CONST(float, act->Op()->GetAttr("slope")));
      }
    }
    if (act_out) {
      fc_xpu_op_desc.SetOutput("out", {act_out->Name()});
    } else if (add_out) {
      fc_xpu_op_desc.SetOutput("out", {add_out->Name()});
    } else {
      fc_xpu_op_desc.SetOutput("out", {mul_out->Name()});
    }
    auto* fc_xpu = graph->CreateOpNode(&fc_xpu_op_desc);
    SAFE_IR_NODE_LINK_TO(mul_x, fc_xpu);
    SAFE_IR_NODE_LINK_TO(mul_w, fc_xpu);
    SAFE_IR_NODE_LINK_TO(mul_w_max, fc_xpu);
    SAFE_IR_NODE_LINK_TO(bias, fc_xpu);
    if (act_out) {
      SAFE_IR_NODE_LINK_TO(fc_xpu, act_out);
    } else if (add_out) {
      SAFE_IR_NODE_LINK_TO(fc_xpu, add_out);
    } else {
      SAFE_IR_NODE_LINK_TO(fc_xpu, mul_out);
    }

    // delete useless node
    std::unordered_set<const Node*> delete_nodes;
    if (act != nullptr && add != nullptr) {
      delete_nodes = {mul, mul_out, add, add_out, act};
    } else if (act) {
      delete_nodes = {mul, mul_out, act};
    } else if (add) {
      delete_nodes = {mul, mul_out, add};
    } else {
      delete_nodes = {mul};
    }
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_xpu_fuse_pass, paddle::framework::ir::FcXPUFusePass);

REGISTER_PASS_CAPABILITY(fc_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "fc_xpu", 0));
