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
#include "paddle/fluid/framework/ir/quantize_helper.h"
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
  mul->LinksFrom({mul_x}).LinksTo({mul_out});
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

  void CreateFusionWeightsAndBias(
      ir::Graph* graph,
      Scope* scope,
      BlockDesc* block,
      std::string mul_type,
      const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
      std::map<std::string, Node*>* fusion_nodes_map,
      bool with_bias,
      bool with_bn,
      std::string op_weights_precision,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
      const;

  void CreateFusionOutputs(
      ir::Graph* graph,
      Scope* scope,
      BlockDesc* block,
      const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
      std::map<std::string, Node*>* fusion_nodes_map,
      std::string op_weights_precision,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
      const;

  void CreateFusionInputs(
      ir::Graph* graph,
      Scope* scope,
      BlockDesc* block,
      const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
      std::map<std::string, Node*>* fusion_nodes_map,
      std::string op_weights_precision,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
      const;

  Node* GetNodeFromNodesMap(
      const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
      std::string pattern_node_name,
      std::string node_name) const;

  void CreateTheReplicatedWeights(
      ir::Graph* graph,
      Scope* scope,
      BlockDesc* block,
      const std::map<std::string, std::map<std::string, Node*>>& nodes_map)
      const;

  const std::string name_scope_{"fc_xpu_fuse_pass"};
};

void FcXPUFusePass::CreateTheReplicatedWeights(
    ir::Graph* graph,
    Scope* scope,
    BlockDesc* block,
    const std::map<std::string, std::map<std::string, Node*>>& nodes_map)
    const {
  // Get Node
  auto* mul = GetNodeFromNodesMap(nodes_map, "mul", "mul");
  PADDLE_ENFORCE_EQ(
      mul != nullptr,
      true,
      common::errors::InvalidArgument("mul node ptr can not be null"));
  auto mul_w_name = mul->Op()->Input("Y")[0];
  std::string replicated_w_name = mul_w_name + "_copy_" +
                                  std::to_string(block->ID()) + "_" +
                                  std::to_string(mul->id());
  auto* replicated_w_var = scope->FindVar(replicated_w_name);
  if (replicated_w_var == nullptr) {
    auto* filter_tensor =
        scope->FindVar(mul_w_name)->GetMutable<phi::DenseTensor>();
    phi::DenseTensor replicated_filter_tensor;
    Assign(*filter_tensor, &replicated_filter_tensor);

    VarDesc replicated_filter_desc(replicated_w_name);
    replicated_filter_desc.SetPersistable(true);
    replicated_filter_desc.SetShape(
        common::vectorize(replicated_filter_tensor.dims()));
    replicated_filter_desc.SetDataType(
        framework::TransToProtoVarType(replicated_filter_tensor.dtype()));
    graph->CreateVarNode(&replicated_filter_desc);
    auto* block_replicated_filter_desc = block->Var(replicated_w_name);
    block_replicated_filter_desc->SetPersistable(
        replicated_filter_desc.Persistable());
    block_replicated_filter_desc->SetShape(replicated_filter_desc.GetShape());
    block_replicated_filter_desc->SetDataType(
        replicated_filter_desc.GetDataType());
    Assign(replicated_filter_tensor,
           scope->Var(replicated_w_name)->GetMutable<phi::DenseTensor>());
  }
}

Node* FcXPUFusePass::GetNodeFromNodesMap(
    const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
    std::string pattern_node_name,
    std::string node_name) const {
  auto iter = nodes_map.find(pattern_node_name);
  PADDLE_ENFORCE_EQ(
      iter != nodes_map.end(),
      true,
      common::errors::InvalidArgument("nodes_map[%s] not found in nodes_map",
                                      pattern_node_name.c_str()));
  auto node_map = iter->second;
  auto node_iter = node_map.find(node_name);
  PADDLE_ENFORCE_EQ(node_iter != node_map.end(),
                    true,
                    common::errors::InvalidArgument(
                        "nodes_map[%s][%s] not found in nodes_map",
                        pattern_node_name.c_str(),
                        node_name.c_str()));
  return node_iter->second;
}

void FcXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
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

void FcXPUFusePass::CreateFusionWeightsAndBias(
    ir::Graph* graph,
    Scope* scope,
    BlockDesc* block,
    std::string mul_type,
    const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
    std::map<std::string, Node*>* fusion_nodes_map,
    bool with_bias,
    bool with_bn,
    std::string op_weights_precision,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  // Get Node
  auto* mul = GetNodeFromNodesMap(nodes_map, "mul", "mul");
  PADDLE_ENFORCE_EQ(
      mul != nullptr,
      true,
      common::errors::InvalidArgument("mul node ptr can not be null"));
  auto mul_w_name = mul->Op()->Input("Y")[0];
  Node* mul_w = FindNodeWithName(graph, mul_w_name);
  CreateTheReplicatedWeights(graph, scope, block, nodes_map);
  std::string replicated_w_name = mul_w_name + "_copy_" +
                                  std::to_string(block->ID()) + "_" +
                                  std::to_string(mul->id());
  auto* mul_w_replicated_node = FindNodeWithName(graph, replicated_w_name);
  // transfilter fp16 --> fp32
  auto* filter_t = scope->FindVar(mul_w_replicated_node->Name())
                       ->GetMutable<phi::DenseTensor>();
  auto filter_dtype = filter_t->dtype();
  if (filter_dtype == phi::DataType::FLOAT16) {
    CastToFp32(filter_t, nullptr);
  }

  bool transpose_w = false;
  if (mul_type == "matmul") {
    transpose_w = PADDLE_GET_CONST(bool, mul->Op()->GetAttr("transpose_Y"));
  } else if (mul_type == "matmul_v2") {
    transpose_w = PADDLE_GET_CONST(bool, mul->Op()->GetAttr("trans_y"));
  }
  // Get Weight scale in int8 scene
  std::vector<float> weight_scale{};
  if (AreScalesPresentForNodes(var_quant_scales, {mul_w})) {
    weight_scale = GetScaleVecValueForNode(var_quant_scales, mul_w);
  }
  // Create fusion_bias_node
  Node* fusion_bias_node = nullptr;
  if (with_bias) {
    auto* ew_bias_add_bias =
        GetNodeFromNodesMap(nodes_map, "ew_bias_add", "ew_bias_add_bias");
    PADDLE_ENFORCE_EQ(ew_bias_add_bias != nullptr,
                      true,
                      common::errors::InvalidArgument(
                          "ew_bias_add_bias node ptr can not be null"));
    PrepareBias(graph, scope, block, ew_bias_add_bias, &fusion_bias_node);
  }

  if (with_bn) {
    auto* bn = GetNodeFromNodesMap(nodes_map, "bn", "bn");
    PADDLE_ENFORCE_EQ(
        bn != nullptr,
        true,
        common::errors::InvalidArgument("bn node ptr can not be null"));
    auto* bn_bias = GetNodeFromNodesMap(nodes_map, "bn", "bn_bias");
    PADDLE_ENFORCE_EQ(
        bn_bias != nullptr,
        true,
        common::errors::InvalidArgument("bn_bias node ptr can not be null"));
    auto* bn_scale = GetNodeFromNodesMap(nodes_map, "bn", "bn_scale");
    PADDLE_ENFORCE_EQ(
        bn_scale != nullptr,
        true,
        common::errors::InvalidArgument("bn_scale node ptr can not be null"));
    auto* bn_var = GetNodeFromNodesMap(nodes_map, "bn", "bn_var");
    PADDLE_ENFORCE_EQ(
        bn_var != nullptr,
        true,
        common::errors::InvalidArgument("bn_var node ptr can not be null"));
    auto* bn_mean = GetNodeFromNodesMap(nodes_map, "bn", "bn_mean");
    PADDLE_ENFORCE_EQ(
        bn_mean != nullptr,
        true,
        common::errors::InvalidArgument("bn_mean node ptr can not be null"));

    auto bn_bias_t =
        scope->Var(bn_bias->Name())->GetMutable<phi::DenseTensor>();
    auto bn_scale_t =
        scope->Var(bn_scale->Name())->GetMutable<phi::DenseTensor>();
    auto bn_mean_t =
        scope->Var(bn_mean->Name())->GetMutable<phi::DenseTensor>();
    auto bn_var_t = scope->Var(bn_var->Name())->GetMutable<phi::DenseTensor>();
    float* bn_scale_ptr = bn_scale_t->data<float>();
    float* bn_bias_ptr = bn_bias_t->data<float>();
    float* bn_mean_ptr = bn_mean_t->data<float>();
    float* bn_var_ptr = bn_var_t->data<float>();
    auto mean_len = bn_mean_t->numel();
    auto filter_h = filter_t->dims()[0];
    auto filter_w = filter_t->dims()[1];
    float epsilon = PADDLE_GET_CONST(float, bn->Op()->GetAttr("epsilon"));
    if (!with_bias) {
      PrepareBias(graph, scope, block, bn_bias, &fusion_bias_node);
    }

    auto fusion_bias_t =
        scope->Var(fusion_bias_node->Name())->GetMutable<phi::DenseTensor>();
    float* fusion_bias_ptr = fusion_bias_t->data<float>();
    // recompute bias and weights
    for (int i = 0; i < mean_len; ++i) {
      bn_scale_ptr[i] = bn_scale_ptr[i] / sqrtf(bn_var_ptr[i] + epsilon);
    }
    // recompute the weights
    if (op_weights_precision != "int8") {
      float* filter_ptr = filter_t->data<float>();
      for (int i = 0; i < mean_len; ++i) {
        for (int j = 0; j < filter_h; j++) {
          filter_ptr[j * filter_w + i] *= bn_scale_ptr[i];
        }
      }
    } else {
      int8_t* filter_ptr = filter_t->data<int8_t>();
      PADDLE_ENFORCE_EQ(
          weight_scale.size(),
          mean_len,
          common::errors::InvalidArgument(
              "Weight max_scale size must equal batch_norm scale/mean size."));
      for (int i = 0; i < mean_len; i++) {
        weight_scale[i] *= fabs(bn_scale_ptr[i]);
      }
      for (int i = 0; i < mean_len; i++) {
        if (bn_scale_ptr[i] < 0) {
          for (int j = 0; j < filter_h; ++j) {
            filter_ptr[j * filter_w + i] *= -1;
          }
        }
      }
    }
    // recompute bias
    if (!with_bias) {
      for (int i = 0; i < mean_len; ++i) {
        fusion_bias_ptr[i] += (0.0f - bn_mean_ptr[i]) * bn_scale_ptr[i];
      }
    } else {
      for (int i = 0; i < mean_len; ++i) {
        fusion_bias_ptr[i] =
            bn_bias_ptr[i] +
            (fusion_bias_ptr[i] - bn_mean_ptr[i]) * bn_scale_ptr[i];
      }
    }
  }

  (*fusion_nodes_map)["bias"] = fusion_bias_node;

  Node* filter_intx = nullptr;
  Node* filter_max = nullptr;
  Node* scale_max = nullptr;

  std::map<std::string, int> default_type;
  default_type.insert(std::make_pair("fc", -1));
  auto quant_post_type =
      Has("quant_post_dynamic_weight_methods")
          ? Get<std::map<std::string, int>>("quant_post_dynamic_weight_methods")
          : default_type;

  for (auto it = quant_post_type.begin(); it != quant_post_type.end(); ++it) {
    VLOG(5) << "Key:" << it->first;
    VLOG(5) << "Value:" << it->second;
  }

  if (op_weights_precision != "int8") {
    if (quant_post_type.find("fc") != quant_post_type.end() &&
            quant_post_type.find("fc")->second == 2 ||
        quant_post_type.find("fc") != quant_post_type.end() &&
            quant_post_type.find("fc")->second == -1) {
      VLOG(5) << "Use int16 per-tensor weight";
      PrepareWeight<float, int16_t>(graph,
                                    scope,
                                    block,
                                    mul_w_replicated_node,
                                    &filter_intx,
                                    &filter_max,
                                    &scale_max,
                                    !transpose_w,
                                    weight_scale,
                                    false);
    } else if (quant_post_type.find("fc") != quant_post_type.end() &&
               quant_post_type.find("fc")->second == 3) {
      VLOG(5) << "Use int16 per-channel weight";
      PrepareWeight<float, int16_t>(graph,
                                    scope,
                                    block,
                                    mul_w_replicated_node,
                                    &filter_intx,
                                    &filter_max,
                                    &scale_max,
                                    !transpose_w,
                                    weight_scale,
                                    true);
    } else if (quant_post_type.find("fc") != quant_post_type.end() &&
               quant_post_type.find("fc")->second == 4) {
      VLOG(5) << "Use int31 per-tensor weight";
      PrepareWeight<float, float>(graph,
                                  scope,
                                  block,
                                  mul_w_replicated_node,
                                  &filter_intx,
                                  &filter_max,
                                  &scale_max,
                                  !transpose_w,
                                  weight_scale,
                                  false);
    } else if (quant_post_type.find("fc") != quant_post_type.end() &&
                   quant_post_type.find("fc")->second == 0 ||
               quant_post_type.find("fc") != quant_post_type.end() &&
                   quant_post_type.find("fc")->second == 1) {
      VLOG(5) << "Unsupported int8 post quant!";
    } else {
      VLOG(5) << "Unsupported type weight by non-int8!";
    }
  } else {
    VLOG(5) << "Use int8  quant weight";
    PrepareWeight<int8_t, int8_t>(graph,
                                  scope,
                                  block,
                                  mul_w_replicated_node,
                                  &filter_intx,
                                  &filter_max,
                                  &scale_max,
                                  !transpose_w,
                                  weight_scale,
                                  false);
  }

  (*fusion_nodes_map)["w"] = filter_intx;
  (*fusion_nodes_map)["w_max"] = filter_max;
  (*fusion_nodes_map)["scale_max"] = scale_max;
}

void FcXPUFusePass::CreateFusionOutputs(
    ir::Graph* graph,
    Scope* scope,
    BlockDesc* block,
    const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
    std::map<std::string, Node*>* fusion_nodes_map,
    std::string op_weights_precision,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  auto* mul = GetNodeFromNodesMap(nodes_map, "mul", "mul");
  PADDLE_ENFORCE_EQ(
      mul != nullptr,
      true,
      common::errors::InvalidArgument("mul node ptr can not be null"));
  // output && output max
  std::string fc_xpu_out_name;
  Node* fc_out_var_node = nullptr;

  auto* bn = GetNodeFromNodesMap(nodes_map, "bn", "bn");
  auto* ew_bias_add =
      GetNodeFromNodesMap(nodes_map, "ew_bias_add", "ew_bias_add");
  auto* act = GetNodeFromNodesMap(nodes_map, "act", "act");
  if (act) {
    auto* act_out = GetNodeFromNodesMap(nodes_map, "act", "act_out");
    PADDLE_ENFORCE_EQ(
        act_out != nullptr,
        true,
        common::errors::InvalidArgument("act_out node ptr can not be null"));
    fc_xpu_out_name = act_out->Name();
    fc_out_var_node = act_out;
  } else if (bn) {
    auto* bn_out = GetNodeFromNodesMap(nodes_map, "bn", "bn_out");
    PADDLE_ENFORCE_EQ(
        bn_out != nullptr,
        true,
        common::errors::InvalidArgument("bn_out node ptr can not be null"));
    fc_xpu_out_name = bn_out->Name();
    fc_out_var_node = bn_out;
  } else if (ew_bias_add) {
    auto* ew_bias_add_out =
        GetNodeFromNodesMap(nodes_map, "ew_bias_add", "ew_bias_add_out");
    PADDLE_ENFORCE_EQ(ew_bias_add_out != nullptr,
                      true,
                      common::errors::InvalidArgument(
                          "ew_bias_add_out node ptr can not be null"));
    fc_xpu_out_name = ew_bias_add_out->Name();
    fc_out_var_node = ew_bias_add_out;
  } else {
    auto* mul_out = GetNodeFromNodesMap(nodes_map, "mul", "mul_out");
    PADDLE_ENFORCE_EQ(
        mul_out != nullptr,
        true,
        common::errors::InvalidArgument("mul_out node ptr can not be null"));
    fc_xpu_out_name = mul_out->Name();
    fc_out_var_node = mul_out;
  }
  (*fusion_nodes_map)["out"] = fc_out_var_node;

  // Create out max in
  if (op_weights_precision == "int8" &&
      AreScalesPresentForNodes(var_quant_scales, {fc_out_var_node})) {
    std::string fc_out_max_in_name = fc_xpu_out_name + "_max_in";
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    VarDesc fc_out_max_in_desc(fc_out_max_in_name);
    fc_out_max_in_desc.SetPersistable(true);
    fc_out_max_in_desc.SetShape({static_cast<int64_t>(max_ptr_size)});
    fc_out_max_in_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    Node* fc_xpu_out_max_in = graph->CreateVarNode(&fc_out_max_in_desc);
    auto* block_out_max_in_desc = block->Var(fc_out_max_in_name);
    block_out_max_in_desc->SetPersistable(fc_out_max_in_desc.Persistable());
    block_out_max_in_desc->SetShape(fc_out_max_in_desc.GetShape());
    block_out_max_in_desc->SetDataType(fc_out_max_in_desc.GetDataType());

    float output_scale =
        GetScaleValueForNode(var_quant_scales, fc_out_var_node);
    phi::DenseTensor out_max_in_cpu_tensor;
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    out_max_in_cpu_tensor.set_type(phi::DataType::FLOAT32);
    out_max_in_cpu_tensor.Resize({max_ptr_size});
    std::vector<float> output_scales(max_ptr_size, output_scale);
    memcpy(cpu_ctx->Alloc<float>(&out_max_in_cpu_tensor),
           output_scales.data(),
           max_ptr_size * sizeof(float));
    Assign(out_max_in_cpu_tensor,
           scope->Var(fc_out_max_in_name)->GetMutable<phi::DenseTensor>());
    (*fusion_nodes_map)["out_max_in"] = fc_xpu_out_max_in;
  }

  // Create out max
  std::string fc_out_max_name = fc_xpu_out_name + "_max";
  VarDesc fc_out_max_desc(fc_out_max_name);
  Node* fc_xpu_out_max = graph->CreateVarNode(&fc_out_max_desc);
  (*fusion_nodes_map)["out_max"] = fc_xpu_out_max;
}

void FcXPUFusePass::CreateFusionInputs(
    ir::Graph* graph,
    Scope* scope,
    BlockDesc* block,
    const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
    std::map<std::string, Node*>* fusion_nodes_map,
    std::string op_weights_precision,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  // Get Node
  auto* mul = GetNodeFromNodesMap(nodes_map, "mul", "mul");
  PADDLE_ENFORCE_EQ(
      mul != nullptr,
      true,
      common::errors::InvalidArgument("mul node ptr can not be null"));
  auto* mul_x = GetNodeFromNodesMap(nodes_map, "mul", "mul_x");
  PADDLE_ENFORCE_EQ(
      mul_x != nullptr,
      true,
      common::errors::InvalidArgument("mul_x node ptr can not be null"));
  // x max
  std::string mul_x_max_name = mul_x->Name() + "_input_max";
  Node* mul_x_max = nullptr;
  if (op_weights_precision == "int8") {
    PADDLE_ENFORCE_EQ(AreScalesPresentForNodes(var_quant_scales, {mul_x}),
                      true,
                      common::errors::InvalidArgument(
                          "When fc op is running in int8 precision, the scales "
                          "of input var should be present in!"));
    float input_scale = GetScaleValueForNode(var_quant_scales, mul_x);
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    VarDesc x_max_desc(mul_x_max_name);
    x_max_desc.SetPersistable(
        true);  // Need depends on ir_params_sync_among_devices_pass copy to xpu
                // device
    x_max_desc.SetShape({static_cast<int64_t>(max_ptr_size)});
    x_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    mul_x_max = graph->CreateVarNode(&x_max_desc);
    auto input_max_tensor =
        scope->Var(mul_x_max_name)->GetMutable<phi::DenseTensor>();
    input_max_tensor->set_type(phi::DataType::FLOAT32);
    input_max_tensor->Resize({max_ptr_size});
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    std::vector<float> input_scales(max_ptr_size, input_scale);
    memcpy(cpu_ctx->Alloc<float>(input_max_tensor),
           input_scales.data(),
           max_ptr_size * sizeof(float));
  }
  (*fusion_nodes_map)["x"] = mul_x;
  (*fusion_nodes_map)["x_max"] = mul_x_max;
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
  auto* scope = param_scope();
  std::unordered_map<std::string, std::vector<float>> var_quant_scales =
      GetQuantInfoFromTheGraph(graph, "has_quant_info", "var_quant_scales");
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FcXPUFusePass fuse";
    GET_IR_NODE(mul_x);
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
    std::map<std::string, std::map<std::string, Node*>> nodes_map;
    nodes_map.insert(
        {"mul", {{"mul", mul}, {"mul_x", mul_x}, {"mul_out", mul_out}}});
    nodes_map.insert({"ew_bias_add",
                      {{"ew_bias_add", add},
                       {"ew_bias_add_bias", bias},
                       {"ew_bias_add_out", add_out}}});
    nodes_map.insert({"bn",
                      {{"bn", bn},
                       {"bn_bias", bn_bias},
                       {"bn_mean", bn_mean},
                       {"bn_scale", bn_scale},
                       {"bn_var", bn_var},
                       {"bn_out", bn_out},
                       {"bn_var_out", bn_var_out},
                       {"bn_mean_out", bn_mean_out},
                       {"bn_saved_var", bn_saved_var},
                       {"bn_saved_mean", bn_saved_mean}}});
    nodes_map.insert({"act", {{"act", act}, {"act_out", act_out}}});

    std::map<std::string, Node*> fusion_nodes_map{{"x", nullptr},
                                                  {"x_max", nullptr},
                                                  {"w", nullptr},
                                                  {"w_max", nullptr},
                                                  {"bias", nullptr},
                                                  {"scale_max", nullptr},
                                                  {"out_max_in", nullptr},
                                                  {"out", nullptr},
                                                  {"out_max", nullptr}};
    auto mul_w_name = mul->Op()->Input("Y")[0];
    Node* mul_w = FindNodeWithName(graph, mul_w_name);
    if (!mul_w->Var()->Persistable() || mul_w->Var()->GetShape().size() != 2) {
      return;
    }
    auto filter_data_type = scope->FindVar(mul->Op()->Input("Y")[0])
                                ->GetMutable<phi::DenseTensor>()
                                ->dtype();
    std::string op_weights_precision = "float32";
    if (filter_data_type == phi::DataType::INT8) {
      op_weights_precision = "int8";
    } else if (filter_data_type == phi::DataType::FLOAT16) {
      op_weights_precision = "float16";
    }
    if (op_weights_precision == "float32" &&
        AreScalesPresentForNodes(&var_quant_scales, {mul_w})) {
      // convert weight to int8
      auto* var = scope->FindVar(mul_w_name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          common::errors::NotFound(
              "The input persistable [%s] var of [%s] op is not found.",
              mul_w_name));
      auto* weight_tensor = var->GetMutable<phi::DenseTensor>();
      float* fp32_weight_data = weight_tensor->data<float>();
      std::vector<int8_t> weight_data;
      weight_data.resize(weight_tensor->numel());
      for (int i = 0; i < weight_tensor->numel(); i++) {
        weight_data[i] = static_cast<int8_t>(fp32_weight_data[i]);
      }
      const auto weight_dims = weight_tensor->dims();
      weight_tensor->clear();  // clear int weight
      weight_tensor->set_type(phi::DataType::INT8);
      weight_tensor->Resize(common::make_ddim(common::vectorize(weight_dims)));
      auto* cpu_ctx = static_cast<phi::CPUContext*>(
          phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
      auto* new_weight_data = cpu_ctx->Alloc<int8_t>(weight_tensor);
      memcpy(new_weight_data,
             weight_data.data(),
             weight_tensor->numel() * sizeof(int8_t));
      op_weights_precision = "int8";
    }

    VLOG(4) << "FC fusion fuse pass is running on " << op_weights_precision
            << " precision!";
    auto* block = mul->Op()->Block();
    CreateFusionWeightsAndBias(graph,
                               scope,
                               block,
                               mul_type,
                               nodes_map,
                               &fusion_nodes_map,
                               with_bias,
                               with_bn,
                               op_weights_precision,
                               &var_quant_scales);
    CreateFusionInputs(graph,
                       scope,
                       block,
                       nodes_map,
                       &fusion_nodes_map,
                       op_weights_precision,
                       &var_quant_scales);
    CreateFusionOutputs(graph,
                        scope,
                        block,
                        nodes_map,
                        &fusion_nodes_map,
                        op_weights_precision,
                        &var_quant_scales);

    // Generate fc_xpu op
    framework::OpDesc fc_xpu_op_desc(block);
    fc_xpu_op_desc.SetType("fc_xpu");
    fc_xpu_op_desc.SetInput("x", {fusion_nodes_map["x"]->Name()});
    if (fusion_nodes_map["x_max"]) {
      fc_xpu_op_desc.SetInput("x_max", {fusion_nodes_map["x_max"]->Name()});
    }
    fc_xpu_op_desc.SetInput("w", {fusion_nodes_map["w"]->Name()});
    fc_xpu_op_desc.SetInput("w_max", {fusion_nodes_map["w_max"]->Name()});
    if (fusion_nodes_map["bias"]) {
      fc_xpu_op_desc.SetInput("bias", {fusion_nodes_map["bias"]->Name()});
    }
    if (fusion_nodes_map["scale_max"]) {
      fc_xpu_op_desc.SetInput("scale_max",
                              {fusion_nodes_map["scale_max"]->Name()});
    }
    if (fusion_nodes_map["out_max_in"]) {
      fc_xpu_op_desc.SetInput("out_max_in",
                              {fusion_nodes_map["out_max_in"]->Name()});
    }
    fc_xpu_op_desc.SetOutput("out", {fusion_nodes_map["out"]->Name()});
    fc_xpu_op_desc.SetOutput("out_max", {fusion_nodes_map["out_max"]->Name()});
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
    // out_dtype is same to input precision
    fc_xpu_op_desc.SetAttr("out_dtype",
                           fusion_nodes_map["x"]->Var()->GetDataType());
    auto* fc_xpu = graph->CreateOpNode(&fc_xpu_op_desc);
    IR_NODE_LINK_TO(fusion_nodes_map["x"], fc_xpu);
    if (fusion_nodes_map["x_max"]) {
      IR_NODE_LINK_TO(fusion_nodes_map["x_max"], fc_xpu);
    }
    IR_NODE_LINK_TO(fusion_nodes_map["w"], fc_xpu);
    IR_NODE_LINK_TO(fusion_nodes_map["w_max"], fc_xpu);
    if (fusion_nodes_map["scale_max"]) {
      IR_NODE_LINK_TO(fusion_nodes_map["scale_max"], fc_xpu);
    }
    if (fusion_nodes_map["bias"]) {
      IR_NODE_LINK_TO(fusion_nodes_map["bias"], fc_xpu);
    }
    if (fusion_nodes_map["out_max_in"]) {
      IR_NODE_LINK_TO(fusion_nodes_map["out_max_in"], fc_xpu);
    }
    IR_NODE_LINK_TO(fc_xpu, fusion_nodes_map["out"]);
    IR_NODE_LINK_TO(fc_xpu, fusion_nodes_map["out_max"]);

    // delete useless node
    std::unordered_set<const Node*> delete_nodes;
    if (mul != nullptr) {
      delete_nodes.insert(mul);
    }
    if (bn != nullptr) {
      delete_nodes.insert(bn);
    }
    if (add != nullptr) {
      delete_nodes.insert(add);
    }
    if (act != nullptr) {
      delete_nodes.insert(act);
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
