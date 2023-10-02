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

struct FcXPUPattern : public PatternBase {
  FcXPUPattern(PDPattern* pattern,
               const std::string& name_scope,
               const std::string& mul_type,
               bool with_bias,
               bool with_bn,
               const std::string& act_type);

  // declare operator node's name
  PATTERN_DECL_NODE(mul);
  PATTERN_DECL_NODE(add);
  PATTERN_DECL_NODE(bn);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(mul_x);
  PATTERN_DECL_NODE(mul_w);
  PATTERN_DECL_NODE(mul_out);
  PATTERN_DECL_NODE(bias);
  PATTERN_DECL_NODE(add_out);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_mean);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_var);
  PATTERN_DECL_NODE(bn_out);
  PATTERN_DECL_NODE(bn_var_out);
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_saved_var);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(act_out);

 private:
  std::string mul_type_;
  bool with_bias_{false};
  bool with_bn_{false};
  std::string act_type_;
};

FcXPUPattern::FcXPUPattern(PDPattern* pattern,
                           const std::string& name_scope,
                           const std::string& mul_type,
                           bool with_bias,
                           bool with_bn,
                           const std::string& act_type)
    : PatternBase(pattern, name_scope, name_scope),
      mul_type_(mul_type),
      with_bias_(with_bias),
      with_bn_(with_bn),
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
    if (with_bn_ || !act_type_.empty()) {
      add_out->assert_has_n_outputs(1);
    }
    add->LinksFrom({mul_out, bias}).LinksTo({add_out});
  } else {
    add_out = mul_out;
  }
  PDNode* bn = nullptr;
  PDNode* bn_bias = nullptr;
  PDNode* bn_mean = nullptr;
  PDNode* bn_scale = nullptr;
  PDNode* bn_var = nullptr;
  PDNode* bn_out = nullptr;
  PDNode* bn_mean_out = nullptr;
  PDNode* bn_saved_mean = nullptr;
  PDNode* bn_var_out = nullptr;
  PDNode* bn_saved_var = nullptr;
  if (with_bn_) {
    add_out->assert_is_op_input("batch_norm", "X");
    bn_bias = pattern->NewNode(bn_bias_repr())
                  ->assert_is_op_input("batch_norm", "Bias")
                  ->assert_has_n_outputs(1);
    bn_mean = pattern->NewNode(bn_mean_repr())
                  ->assert_is_op_input("batch_norm", "Mean")
                  ->assert_has_n_outputs(1);
    bn_scale = pattern->NewNode(bn_scale_repr())
                   ->assert_is_op_input("batch_norm", "Scale")
                   ->assert_has_n_outputs(1);
    bn_var = pattern->NewNode(bn_var_repr())
                 ->assert_is_op_input("batch_norm", "Variance")
                 ->assert_has_n_outputs(1);
    bn = pattern->NewNode(bn_repr())->assert_is_op("batch_norm");
    bn_out =
        pattern->NewNode(bn_out_repr())->assert_is_op_output("batch_norm", "Y");
    if (!act_type_.empty()) {
      bn_out->assert_has_n_outputs(1);
    }
    bn_mean_out = pattern->NewNode(bn_mean_out_repr())
                      ->assert_is_op_output("batch_norm", "MeanOut");
    bn_saved_mean = pattern->NewNode(bn_saved_mean_repr())
                        ->assert_is_op_output("batch_norm", "SavedMean");
    bn_var_out = pattern->NewNode(bn_var_out_repr())
                     ->assert_is_op_output("batch_norm", "VarianceOut");
    bn_saved_var = pattern->NewNode(bn_saved_var_repr())
                       ->assert_is_op_output("batch_norm", "SavedVariance");
    bn->LinksFrom({add_out, bn_bias, bn_mean, bn_scale, bn_var})
        .LinksTo(
            {bn_out, bn_mean_out, bn_var_out, bn_saved_mean, bn_saved_var});
  } else {
    bn_out = add_out;
  }
  if (!act_type_.empty()) {
    bn_out->assert_is_op_input(act_type_, "X");
    act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
    act_out = pattern->NewNode(act_out_repr())
                  ->assert_is_op_output(act_type_, "Out")
                  ->assert_var_not_persistable();
    act->LinksFrom({bn_out}).LinksTo({act_out});
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
               batch_norm
                   |
                   |
             batch_norm_out
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
              |  \
              |   \
         act_out  out_max
*/
class FcXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                const std::string& mul_type,
                bool with_bias,
                bool with_bn,
                const std::string& act_type) const;

  const std::string name_scope_{"fc_xpu_fuse_pass"};
};

void FcXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto mul_type : {"mul", "matmul", "matmul_v2"}) {
    for (auto with_bias : {true, false}) {
      for (auto with_bn : {true, false}) {
        for (auto act_type : {
                 "relu",
                 "gelu",
                 "tanh",
                 "sigmoid",
                 "swish",
                 "relu6",
                 "leaky_relu",
                 "",
             }) {
          found_subgraph_count +=
              ApplyImpl(graph, mul_type, with_bias, with_bn, act_type);
        }
      }
    }
  }
  AddStatis(found_subgraph_count);
}

int FcXPUFusePass::ApplyImpl(ir::Graph* graph,
                             const std::string& mul_type,
                             bool with_bias,
                             bool with_bn,
                             const std::string& act_type) const {
  GraphPatternDetector gpd;
  patterns::FcXPUPattern pattern(gpd.mutable_pattern(),
                                 name_scope_,
                                 mul_type,
                                 with_bias,
                                 with_bn,
                                 act_type);

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
    GET_IR_NODE(bn);
    GET_IR_NODE(bn_bias);
    GET_IR_NODE(bn_mean);
    GET_IR_NODE(bn_scale);
    GET_IR_NODE(bn_var);
    GET_IR_NODE(bn_out);
    GET_IR_NODE(bn_var_out);
    GET_IR_NODE(bn_mean_out);
    GET_IR_NODE(bn_saved_var);
    GET_IR_NODE(bn_saved_mean);
    GET_IR_NODE(act);
    GET_IR_NODE(act_out);
    auto* block = mul->Op()->Block();
    auto* scope = param_scope();

    auto* filter_t =
        scope->FindVar(mul_w->Name())->GetMutable<phi::DenseTensor>();
    // weight fp16 --> fp32
    auto filter_dtype = filter_t->dtype();
    int out_dtype = proto::VarType::Type::VarType_Type_FP32;
    if (filter_dtype == phi::DataType::FLOAT16) {
      out_dtype = proto::VarType::Type::VarType_Type_FP16;
      CastToFp32(filter_t, nullptr);
    }
    auto filter_dims = filter_t->dims();

    bool transpose_w = false;
    if (mul_type == "matmul") {
      transpose_w = PADDLE_GET_CONST(bool, mul->Op()->GetAttr("transpose_Y"));
    } else if (mul_type == "matmul_v2") {
      transpose_w = PADDLE_GET_CONST(bool, mul->Op()->GetAttr("trans_y"));
    }

    bool has_bias = with_bn || with_bias;
    Node* fusion_bias_node = nullptr;
    if (has_bias) {
      if (bias != nullptr) {
        PrepareBias(graph, scope, block, bias, &fusion_bias_node);
      }
      if (bn != nullptr) {
        auto bn_bias_t =
            scope->Var(bn_bias->Name())->GetMutable<phi::DenseTensor>();
        auto bn_scale_t =
            scope->Var(bn_scale->Name())->GetMutable<phi::DenseTensor>();
        auto bn_mean_t =
            scope->Var(bn_mean->Name())->GetMutable<phi::DenseTensor>();
        auto bn_var_t =
            scope->Var(bn_var->Name())->GetMutable<phi::DenseTensor>();
        float* mul_w_ptr = filter_t->data<float>();
        float* bn_scale_ptr = bn_scale_t->data<float>();
        float* bn_bias_ptr = bn_bias_t->data<float>();
        float* bn_mean_ptr = bn_mean_t->data<float>();
        float* bn_var_ptr = bn_var_t->data<float>();
        auto mean_len = bn_mean_t->numel();
        auto filter_h = filter_dims[0];
        auto filter_w = filter_dims[1];
        float epsilon = PADDLE_GET_CONST(float, bn->Op()->GetAttr("epsilon"));
        if (fusion_bias_node == nullptr) {  // prev node is conv
          PrepareBias(graph, scope, block, bn_bias, &fusion_bias_node);
        }
        auto fusion_bias_t = scope->Var(fusion_bias_node->Name())
                                 ->GetMutable<phi::DenseTensor>();
        float* fusion_bias_ptr = fusion_bias_t->data<float>();
        // recompute bias and weights
        if (bias == nullptr) {
          for (int i = 0; i < mean_len; ++i) {
            bn_scale_ptr[i] = bn_scale_ptr[i] / sqrtf(bn_var_ptr[i] + epsilon);
            fusion_bias_ptr[i] += (0.f - bn_mean_ptr[i]) * bn_scale_ptr[i];
            for (int j = 0; j < filter_h; j++) {
              mul_w_ptr[j * filter_w + i] *= bn_scale_ptr[i];
            }
          }
        } else {
          for (int i = 0; i < mean_len; ++i) {
            bn_scale_ptr[i] = bn_scale_ptr[i] / sqrtf(bn_var_ptr[i] + epsilon);
            bn_bias_ptr[i] +=
                (fusion_bias_ptr[i] - bn_mean_ptr[i]) * bn_scale_ptr[i];
            for (int j = 0; j < filter_h; j++) {
              mul_w_ptr[j * filter_w + i] *= bn_scale_ptr[i];
            }
          }
          memcpy(fusion_bias_ptr, bn_bias_ptr, mean_len * sizeof(float));
        }
      }
    }

    Node* mul_w_int16 = nullptr;
    Node* mul_w_max = nullptr;
    PrepareWeight<int16_t>(
        graph, scope, block, mul_w, &mul_w_int16, &mul_w_max, !transpose_w);

    std::string fc_out_name;
    if (act_out) {
      fc_out_name = act_out->Name();
    } else if (bn) {
      fc_out_name = bn_out->Name();
    } else if (add_out) {
      fc_out_name = add_out->Name();
    } else {
      fc_out_name = mul_out->Name();
    }
    std::string fc_out_max_name = fc_out_name + "_max";
    VarDesc fc_out_max_desc(fc_out_max_name);
    Node* fc_out_max = graph->CreateVarNode(&fc_out_max_desc);

    // Generate fc_xpu op
    framework::OpDesc fc_xpu_op_desc(block);
    fc_xpu_op_desc.SetType("fc_xpu");
    fc_xpu_op_desc.SetInput("x", {mul_x->Name()});
    fc_xpu_op_desc.SetInput("w", {mul_w_int16->Name()});
    fc_xpu_op_desc.SetInput("w_max", {mul_w_max->Name()});
    if (has_bias) {
      fc_xpu_op_desc.SetInput("bias", {fusion_bias_node->Name()});
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
    fc_xpu_op_desc.SetAttr("out_dtype", out_dtype);
    fc_xpu_op_desc.SetOutput("out", {fc_out_name});
    fc_xpu_op_desc.SetOutput("out_max", {fc_out_max_name});
    auto* fc_xpu = graph->CreateOpNode(&fc_xpu_op_desc);
    IR_NODE_LINK_TO(mul_x, fc_xpu);
    IR_NODE_LINK_TO(mul_w_int16, fc_xpu);
    IR_NODE_LINK_TO(mul_w_max, fc_xpu);
    if (bias || bn) {
      SAFE_IR_NODE_LINK_TO(fusion_bias_node, fc_xpu);
    }
    if (act_out) {
      IR_NODE_LINK_TO(fc_xpu, act_out);
    } else if (bn_out) {
      IR_NODE_LINK_TO(fc_xpu, bn_out);
    } else if (add_out) {
      IR_NODE_LINK_TO(fc_xpu, add_out);
    } else {
      IR_NODE_LINK_TO(fc_xpu, mul_out);
    }
    IR_NODE_LINK_TO(fc_xpu, fc_out_max);

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
    if (bn != nullptr) {
      delete_nodes.insert(bn);
      delete_nodes.insert(bn_bias);
      delete_nodes.insert(bn_var);
      delete_nodes.insert(bn_mean);
      delete_nodes.insert(bn_scale);
      delete_nodes.insert(bn_var_out);
      delete_nodes.insert(bn_mean_out);
      delete_nodes.insert(bn_saved_var);
      delete_nodes.insert(bn_saved_mean);
    }
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_xpu_fuse_pass, paddle::framework::ir::FcXPUFusePass);

REGISTER_PASS_CAPABILITY(fc_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "fc_xpu", 0));
