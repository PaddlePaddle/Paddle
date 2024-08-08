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

#include <map>
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

struct Conv2dXPUPattern : public PatternBase {
  Conv2dXPUPattern(PDPattern* pattern,
                   const std::string& name_scope,
                   const std::string& conv_type,
                   const std::string& act_type,
                   bool with_conv_bias,
                   bool with_bn,
                   bool with_scale,
                   bool with_branch_x,
                   bool with_branch_y);
  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(ew_bias_add);
  PATTERN_DECL_NODE(bn);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(ew_branch_add);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(input);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(ew_bias_add_y);
  PATTERN_DECL_NODE(ew_bias_add_out);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_mean);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_var);
  PATTERN_DECL_NODE(bn_out);
  PATTERN_DECL_NODE(bn_var_out);
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_saved_var);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(ew_branch_add_in);
  PATTERN_DECL_NODE(ew_branch_add_out);
  PATTERN_DECL_NODE(act_out);

 private:
  std::string conv_type_;
  std::string act_type_;
  bool with_conv_bias_{false};
  bool with_bn_{false};
  bool with_scale_{false};
  bool with_branch_{false};
  bool with_branch_x_{false};
  bool with_branch_y_{false};
};

Conv2dXPUPattern::Conv2dXPUPattern(PDPattern* pattern,
                                   const std::string& name_scope,
                                   const std::string& conv_type,
                                   const std::string& act_type,
                                   bool with_conv_bias,
                                   bool with_bn,
                                   bool with_scale,
                                   bool with_branch_x,
                                   bool with_branch_y)
    : PatternBase(pattern, name_scope, name_scope),
      conv_type_(conv_type),
      act_type_(act_type),
      with_conv_bias_(with_conv_bias),
      with_bn_(with_bn),
      with_scale_(with_scale),
      with_branch_(with_branch_x || with_branch_y),
      with_branch_x_(with_branch_x),
      with_branch_y_(with_branch_y) {
  auto conv = pattern->NewNode(conv_repr())->assert_is_op(conv_type_);
  auto input = pattern->NewNode(input_repr())
                   ->assert_is_op_input(conv_type_, "Input")
                   ->AsInput()
                   ->assert_more([](Node* node) {
                     return node->Var()->GetShape().size() == 4;
                   });
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output(conv_type_, "Output")
                      ->assert_has_n_outputs(1);
  conv->LinksFrom({input}).LinksTo({conv_out});
  // ew_bias_add op
  PDNode* ew_bias_add = nullptr;
  PDNode* ew_bias_add_y = nullptr;
  PDNode* ew_bias_add_out = nullptr;
  if (with_conv_bias_) {
    conv_out->assert_is_op_input("elementwise_add", "X");
    ew_bias_add_y = pattern->NewNode(ew_bias_add_y_repr())
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->assert_is_persistable_var()
                        ->assert_has_n_outputs(1)
                        ->assert_more([](Node* node) {
                          return node->Var()->GetShape().size() == 1;
                        });
    ew_bias_add =
        pattern->NewNode(ew_bias_add_repr())->assert_is_op("elementwise_add");
    ew_bias_add_out = pattern->NewNode(ew_bias_add_out_repr())
                          ->assert_is_op_output("elementwise_add", "Out");
    if (with_bn_ || with_scale_ || with_branch_ || !act_type_.empty()) {
      ew_bias_add_out->assert_has_n_outputs(1);
    }
    ew_bias_add->LinksFrom({conv_out, ew_bias_add_y})
        .LinksTo({ew_bias_add_out});
  } else {
    ew_bias_add_out = conv_out;
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
  PDNode* ew_branch_add = nullptr;
  PDNode* ew_branch_add_in = nullptr;
  PDNode* ew_branch_add_out = nullptr;
  PDNode* scale = nullptr;
  PDNode* scale_out = nullptr;
  PDNode* act = nullptr;
  PDNode* act_out = nullptr;
  // batch_norm op
  if (with_bn_) {
    ew_bias_add_out->assert_is_op_input("batch_norm", "X");
    bn_bias = pattern->NewNode(bn_bias_repr())
                  ->AsInput()
                  ->assert_is_persistable_var()
                  ->assert_is_op_input("batch_norm", "Bias")
                  ->assert_has_n_outputs(1);
    bn_mean = pattern->NewNode(bn_mean_repr())
                  ->AsInput()
                  ->assert_is_persistable_var()
                  ->assert_is_op_input("batch_norm", "Mean")
                  ->assert_has_n_outputs(1);
    bn_scale = pattern->NewNode(bn_scale_repr())
                   ->AsInput()
                   ->assert_is_persistable_var()
                   ->assert_is_op_input("batch_norm", "Scale")
                   ->assert_has_n_outputs(1);
    bn_var = pattern->NewNode(bn_var_repr())
                 ->AsInput()
                 ->assert_is_persistable_var()
                 ->assert_is_op_input("batch_norm", "Variance")
                 ->assert_has_n_outputs(1);
    bn = pattern->NewNode(bn_repr())->assert_is_op("batch_norm");
    bn_out =
        pattern->NewNode(bn_out_repr())->assert_is_op_output("batch_norm", "Y");
    if (with_scale_ || with_branch_ || !act_type_.empty()) {
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
    bn->LinksFrom({ew_bias_add_out, bn_bias, bn_mean, bn_scale, bn_var})
        .LinksTo(
            {bn_out, bn_mean_out, bn_var_out, bn_saved_mean, bn_saved_var});
  } else {
    bn_out = ew_bias_add_out;
  }
  // scale op
  if (with_scale_) {
    bn_out->assert_is_op_input("scale", "X");
    scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
    scale_out =
        pattern->NewNode(scale_out_repr())->assert_is_op_output("scale", "Out");
    if (with_bn_ || !act_type_.empty()) {
      scale_out->assert_has_n_outputs(1);
    }
    scale->LinksFrom({bn_out}).LinksTo({scale_out});
  } else {
    scale_out = bn_out;
  }
  // ew_branch_add op
  if (with_branch_) {
    if (with_branch_x_) {
      scale_out->assert_is_op_input("elementwise_add", "Y");
      ew_branch_add_in = pattern->NewNode(ew_branch_add_in_repr())
                             ->assert_is_op_input("elementwise_add", "X")
                             ->AsInput();
    } else if (with_branch_y_) {
      bn_out->assert_is_op_input("elementwise_add", "X");
      ew_branch_add_in = pattern->NewNode(ew_branch_add_in_repr())
                             ->assert_is_op_input("elementwise_add", "Y")
                             ->AsInput();
    }
    ew_branch_add = pattern->NewNode(ew_branch_add_repr())
                        ->assert_is_op("elementwise_add")
                        ->assert_more([](Node* node) {
                          if (node->inputs.size() != 2) {
                            return false;
                          }
                          return node->inputs[0]->Var()->GetShape() ==
                                 node->inputs[1]->Var()->GetShape();
                        });
    ew_branch_add_out = pattern->NewNode(ew_branch_add_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out");
    if (!act_type_.empty()) {
      ew_branch_add_out->assert_has_n_outputs(1);
    }
    ew_branch_add->LinksFrom({bn_out, ew_branch_add_in})
        .LinksTo({ew_branch_add_out});
  } else {
    ew_branch_add_out = scale_out;
  }
  // act op
  if (!act_type_.empty()) {
    ew_branch_add_out->assert_is_op_input(act_type_, "X");
    act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
    act_out =
        pattern->NewNode(act_out_repr())->assert_is_op_output(act_type_, "Out");
    act->LinksFrom({ew_branch_add_out}).LinksTo({act_out});
  } else {
    act_out = ew_branch_add_out;
  }
  act_out->AsOutput();
}

}  // namespace patterns

/*
fuse conv2d block in resnet50-like model to xpu_conv2d op
For example:
graph[1]: sub block
                    in_Input
                      |
                      |
                    conv2d----in_Filter
                      |
                      |
                 elementwise_add -----conv_Bias
                      |
                      |
                 batch_norm ------in_Bias
                      |
                      |
                     act
                      |
                      |
                    out_Out
------------------------------------------------------
graph[2]: sub block
                    in_Input
                      |
                      |
                    conv2d----in_Filter
                      |
                      |
                 batch_norm ------in_Bias
                      |
                      |
                    out_Out
------------------------------------------------------
graph[3]: sub block
                    in_Input
                      |
                      |
                    conv2d----in_Filter
                      |
                      |
       in_X       batch_norm ------in_Bias
            \         |
              \       |
               elementwise_add
                      |
                      |
                     act
                      |
                      |
                    out_Out
------------------------------------------------------
graph[4]: sub block
                    in_Input
                      |
                      |
                    conv2d----in_Filter
                      |
                      |
               elementwise_add ------in_Bias
                      |
                      |
                     act
                      |
                      |
                    out_Out
------------------------------------------------------
After the pass is applied:
                    in_Input
       in_Filter      |     in_FilterMax
                 \    |    /
                  \   |   /
  in_Branch ------- __xpu__conv2d ------ in_Bias
                       |    \
                       |     \
                       |      out_OutputMax
                    out_Output
*/
class Conv2dXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                const std::string& conv_type,
                const std::string& act_type,
                bool with_conv_bias,
                bool with_bn,
                bool with_scale,
                bool with_branch_x,
                bool with_branch_y) const;

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

  void CreateFusionWeightsAndBias(
      ir::Graph* graph,
      Scope* scope,
      BlockDesc* block,
      const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
      std::map<std::string, Node*>* fusion_nodes_map,
      bool with_conv_bias,
      bool with_bn,
      bool with_scale,
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

  void CreateFusionBranch(
      ir::Graph* graph,
      Scope* scope,
      BlockDesc* block,
      const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
      std::map<std::string, Node*>* fusion_nodes_map,
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
      std::string act_type,
      std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
      const;

  const std::string name_scope_{"conv2d_xpu_fuse_pass"};
};

void Conv2dXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto conv_type : {"conv2d", "depthwise_conv2d"}) {
    for (auto with_conv_bias : {true, false}) {
      for (auto with_bn : {true, false}) {
        for (auto with_scale : {true, false}) {
          for (auto with_branch_x : {true, false}) {
            for (auto with_branch_y : {true, false}) {
              for (auto act_type : {
                       "relu",
                       "sigmoid",
                       "tanh",
                       "gelu",
                       "leaky_relu",
                       "hard_swish",
                       "hard_sigmoid",
                       "relu6",
                       "swish",
                       "",
                   }) {
                if (with_branch_x && with_branch_y) continue;
                found_subgraph_count += ApplyImpl(graph,
                                                  conv_type,
                                                  act_type,
                                                  with_conv_bias,
                                                  with_bn,
                                                  with_scale,
                                                  with_branch_x,
                                                  with_branch_y);
              }
            }
          }
        }
      }
    }
  }
  AddStatis(found_subgraph_count);
}

Node* Conv2dXPUFusePass::GetNodeFromNodesMap(
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

void Conv2dXPUFusePass::CreateTheReplicatedWeights(
    ir::Graph* graph,
    Scope* scope,
    BlockDesc* block,
    const std::map<std::string, std::map<std::string, Node*>>& nodes_map)
    const {
  // Get Node
  auto* conv = GetNodeFromNodesMap(nodes_map, "conv", "conv");
  PADDLE_ENFORCE_EQ(
      conv != nullptr,
      true,
      common::errors::InvalidArgument("conv node ptr can not be null"));
  auto conv_filter_name = conv->Op()->Input("Filter")[0];
  std::string replicated_filter_name = conv_filter_name + "_copy_" +
                                       std::to_string(block->ID()) + "_" +
                                       std::to_string(conv->id());
  auto* replicated_filter_var = scope->FindVar(replicated_filter_name);
  if (replicated_filter_var == nullptr) {
    auto* filter_tensor =
        scope->FindVar(conv_filter_name)->GetMutable<phi::DenseTensor>();
    phi::DenseTensor replicated_filter_tensor;
    Assign(*filter_tensor, &replicated_filter_tensor);

    VarDesc replicated_filter_desc(replicated_filter_name);
    replicated_filter_desc.SetPersistable(true);
    replicated_filter_desc.SetShape(
        common::vectorize(replicated_filter_tensor.dims()));
    replicated_filter_desc.SetDataType(
        framework::TransToProtoVarType(replicated_filter_tensor.dtype()));
    graph->CreateVarNode(&replicated_filter_desc);
    auto* block_replicated_filter_desc = block->Var(replicated_filter_name);
    block_replicated_filter_desc->SetPersistable(
        replicated_filter_desc.Persistable());
    block_replicated_filter_desc->SetShape(replicated_filter_desc.GetShape());
    block_replicated_filter_desc->SetDataType(
        replicated_filter_desc.GetDataType());
    Assign(replicated_filter_tensor,
           scope->Var(replicated_filter_name)->GetMutable<phi::DenseTensor>());
  }
}

void Conv2dXPUFusePass::CreateFusionWeightsAndBias(
    ir::Graph* graph,
    Scope* scope,
    BlockDesc* block,
    const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
    std::map<std::string, Node*>* fusion_nodes_map,
    bool with_conv_bias,
    bool with_bn,
    bool with_scale,
    std::string op_weights_precision,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  // Get Node
  auto* conv = GetNodeFromNodesMap(nodes_map, "conv", "conv");
  PADDLE_ENFORCE_EQ(
      conv != nullptr,
      true,
      common::errors::InvalidArgument("conv node ptr can not be null"));
  auto conv_filter_name = conv->Op()->Input("Filter")[0];
  Node* conv_filter = FindNodeWithName(graph, conv_filter_name);
  CreateTheReplicatedWeights(graph, scope, block, nodes_map);
  std::string replicated_filter_name = conv_filter_name + "_copy_" +
                                       std::to_string(block->ID()) + "_" +
                                       std::to_string(conv->id());
  auto* conv_filter_replicated_node =
      FindNodeWithName(graph, replicated_filter_name);
  auto* filter_t =
      scope->FindVar(replicated_filter_name)->GetMutable<phi::DenseTensor>();
  auto filter_len = filter_t->numel();
  auto filter_dtype = filter_t->dtype();
  // transfilter fp16 --> fp32
  if (filter_dtype == phi::DataType::FLOAT16) {
    CastToFp32(filter_t, nullptr);
  }

  // Get Weight scale in int8 scene
  std::vector<float> weight_scale{};
  if (AreScalesPresentForNodes(var_quant_scales, {conv_filter})) {
    weight_scale = GetScaleVecValueForNode(var_quant_scales, conv_filter);
  }
  // Create fusion_bias_node
  auto filter_dims = filter_t->dims();
  Node* fusion_bias_node = nullptr;
  if (with_conv_bias) {
    auto* ew_bias_add_y =
        GetNodeFromNodesMap(nodes_map, "ew_bias_add", "ew_bias_add_y");
    PADDLE_ENFORCE_EQ(ew_bias_add_y != nullptr,
                      true,
                      common::errors::InvalidArgument(
                          "ew_bias_add_y node ptr can not be null"));
    auto* ew_bias_add_y_t =
        scope->FindVar(ew_bias_add_y->Name())->GetMutable<phi::DenseTensor>();
    auto ew_bias_add_y_dims = ew_bias_add_y_t->dims();
    PADDLE_ENFORCE_EQ(
        filter_dims[0],
        ew_bias_add_y_dims[0],
        common::errors::InvalidArgument("the shape[%d] of elewise bias tensor "
                                        "must equal out_channel[%d] of conv",
                                        ew_bias_add_y_dims[0],
                                        filter_dims[0]));
    PrepareBias(graph, scope, block, ew_bias_add_y, &fusion_bias_node);
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
    PADDLE_ENFORCE_EQ(
        filter_dims[0],
        bn_bias_t->dims()[0],
        common::errors::InvalidArgument("the shape[%d] of bn bias tensor "
                                        "must equal out_channel[%d] of conv",
                                        bn_bias_t->dims()[0],
                                        filter_dims[0]));
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
    auto filter_stride = filter_len / mean_len;
    float epsilon = PADDLE_GET_CONST(float, bn->Op()->GetAttr("epsilon"));
    if (!with_conv_bias) {  // prev node is conv
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
        for (int j = 0; j < filter_stride; j++) {
          filter_ptr[i * filter_stride + j] *= bn_scale_ptr[i];
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
          for (int j = 0; j < filter_stride; ++j) {
            filter_ptr[i * filter_stride + j] *= -1;
          }
        }
      }
    }
    // recompute bias
    if (!with_conv_bias) {
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
  // deal with scale op
  if (with_scale) {
    auto* scale = GetNodeFromNodesMap(nodes_map, "scale", "scale");
    PADDLE_ENFORCE_EQ(
        scale != nullptr,
        true,
        common::errors::InvalidArgument("scale node ptr can not be null"));
    auto bias_len = filter_dims[0];
    float scale_val_ = 1.f;
    float bias_val_ = 0.f;
    scale_val_ = PADDLE_GET_CONST(float, scale->Op()->GetAttr("scale"));
    bias_val_ = PADDLE_GET_CONST(float, scale->Op()->GetAttr("bias"));
    bool bias_after_scale_ =
        PADDLE_GET_CONST(bool, scale->Op()->GetAttr("bias_after_scale"));
    // recompute bias as scale op
    auto fusion_bias_t =
        scope->GetVar(fusion_bias_node->Name())->GetMutable<phi::DenseTensor>();
    float* fusion_bias_ptr = fusion_bias_t->data<float>();
    for (int i = 0; i < bias_len; ++i) {
      if (bias_after_scale_) {
        fusion_bias_ptr[i] = fusion_bias_ptr[i] * scale_val_ + bias_val_;
      } else {
        fusion_bias_ptr[i] = (fusion_bias_ptr[i] + bias_val_) * scale_val_;
      }
    }
    // recompute weight as scale op
    if (op_weights_precision != "int8") {
      float* filter_ptr = filter_t->data<float>();
      for (int i = 0; i < filter_len; ++i) {
        filter_ptr[i] *= scale_val_;
      }
    } else {
      for (size_t i = 0; i < weight_scale.size(); i++) {
        weight_scale[i] *= scale_val_;
      }
    }
  }

  (*fusion_nodes_map)["bias"] = fusion_bias_node;

  Node* filter_intx = nullptr;
  Node* filter_max = nullptr;
  Node* scale_max = nullptr;

  std::map<std::string, int> default_type;
  default_type.insert(std::make_pair("conv2d", -1));
  auto quant_post_type =
      Has("quant_post_dynamic_weight_methods")
          ? Get<std::map<std::string, int>>("quant_post_dynamic_weight_methods")
          : default_type;

  for (auto it = quant_post_type.begin(); it != quant_post_type.end(); ++it) {
    VLOG(5) << "Key:" << it->first;
    VLOG(5) << "Value:" << it->second;
  }

  if (op_weights_precision != "int8") {
    if (quant_post_type.find("conv2d") != quant_post_type.end() &&
            quant_post_type.find("conv2d")->second == 2 ||
        quant_post_type.find("conv2d") != quant_post_type.end() &&
            quant_post_type.find("conv2d")->second == -1) {
      VLOG(5) << "Use int16 per-tensor weight";
      PrepareWeight<float, int16_t>(graph,
                                    scope,
                                    block,
                                    conv_filter_replicated_node,
                                    &filter_intx,
                                    &filter_max,
                                    &scale_max,
                                    false,
                                    weight_scale,
                                    false);
    } else if (quant_post_type.find("conv2d") != quant_post_type.end() &&
               quant_post_type.find("conv2d")->second == 3) {
      VLOG(5) << "Use int16 per-channel weight";
      PrepareWeight<float, int16_t>(graph,
                                    scope,
                                    block,
                                    conv_filter_replicated_node,
                                    &filter_intx,
                                    &filter_max,
                                    &scale_max,
                                    false,
                                    weight_scale,
                                    true);
    } else if (quant_post_type.find("conv2d") != quant_post_type.end() &&
               quant_post_type.find("conv2d")->second == 4) {
      VLOG(5) << "Use int31 per-tensor weight";
      PrepareWeight<float, float>(graph,
                                  scope,
                                  block,
                                  conv_filter_replicated_node,
                                  &filter_intx,
                                  &filter_max,
                                  &scale_max,
                                  false,
                                  weight_scale,
                                  false);
    } else if (quant_post_type.find("conv2d") != quant_post_type.end() &&
                   quant_post_type.find("conv2d")->second == 0 ||
               quant_post_type.find("conv2d") != quant_post_type.end() &&
                   quant_post_type.find("conv2d")->second == 1) {
      VLOG(5) << "Unsupported int8 post quant !";
    } else {
      VLOG(5) << "Unsupported type weight by non-int8!";
    }

  } else {
    VLOG(5) << "Use int8 quant weight";
    PrepareWeight<int8_t, int8_t>(graph,
                                  scope,
                                  block,
                                  conv_filter_replicated_node,
                                  &filter_intx,
                                  &filter_max,
                                  &scale_max,
                                  false,
                                  weight_scale);
  }

  (*fusion_nodes_map)["filter"] = filter_intx;
  (*fusion_nodes_map)["filter_max"] = filter_max;
  (*fusion_nodes_map)["scale_max"] = scale_max;
}

void Conv2dXPUFusePass::CreateFusionInputs(
    ir::Graph* graph,
    Scope* scope,
    BlockDesc* block,
    const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
    std::map<std::string, Node*>* fusion_nodes_map,
    std::string op_weights_precision,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  // Get Node
  auto* conv = GetNodeFromNodesMap(nodes_map, "conv", "conv");
  PADDLE_ENFORCE_EQ(
      conv != nullptr,
      true,
      common::errors::InvalidArgument("conv node ptr can not be null"));
  auto* input = GetNodeFromNodesMap(nodes_map, "conv", "input");
  PADDLE_ENFORCE_EQ(
      input != nullptr,
      true,
      common::errors::InvalidArgument("conv input node ptr can not be null"));
  // input max
  std::string conv_input_max_name = input->Name() + "_input_max";
  Node* conv2d_xpu_input_max = nullptr;
  if (op_weights_precision == "int8") {
    PADDLE_ENFORCE_EQ(AreScalesPresentForNodes(var_quant_scales, {input}),
                      true,
                      common::errors::InvalidArgument(
                          "When conv op is running in int8 precision, the "
                          "scales of input var should be present in!"));
    float input_scale = GetScaleValueForNode(var_quant_scales, input);
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    VarDesc conv_input_max_desc(conv_input_max_name);
    conv_input_max_desc.SetPersistable(true);
    conv_input_max_desc.SetShape({static_cast<int64_t>(max_ptr_size)});
    conv_input_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    conv2d_xpu_input_max = graph->CreateVarNode(&conv_input_max_desc);
    auto input_max_tensor =
        scope->Var(conv_input_max_name)->GetMutable<phi::DenseTensor>();
    input_max_tensor->set_type(phi::DataType::FLOAT32);
    input_max_tensor->Resize({max_ptr_size});
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    std::vector<float> input_scales(max_ptr_size, input_scale);
    memcpy(cpu_ctx->Alloc<float>(input_max_tensor),
           input_scales.data(),
           max_ptr_size * sizeof(float));
  }
  (*fusion_nodes_map)["x"] = input;
  (*fusion_nodes_map)["x_max"] = conv2d_xpu_input_max;
}

void Conv2dXPUFusePass::CreateFusionBranch(
    ir::Graph* graph,
    Scope* scope,
    BlockDesc* block,
    const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
    std::map<std::string, Node*>* fusion_nodes_map,
    std::string op_weights_precision,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  // Get Node
  auto* ew_branch_add =
      GetNodeFromNodesMap(nodes_map, "ew_branch_add", "ew_branch_add");
  if (ew_branch_add) {
    auto* ew_branch_add_in =
        GetNodeFromNodesMap(nodes_map, "ew_branch_add", "ew_branch_add_in");
    PADDLE_ENFORCE_EQ(ew_branch_add_in != nullptr,
                      true,
                      common::errors::InvalidArgument(
                          "ew_branch_add_in node ptr can not be null"));
    (*fusion_nodes_map)["branch"] = ew_branch_add_in;
    // ew_branch_add_max
    std::string ew_branch_add_max_name =
        ew_branch_add_in->Name() + "branch_max";
    Node* ew_branch_add_max = FindNodeWithName(graph, ew_branch_add_max_name);
    if (op_weights_precision == "int8" && !ew_branch_add_max) {
      int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
      VarDesc ew_branch_add_in_max_desc(ew_branch_add_max_name);
      ew_branch_add_in_max_desc.SetPersistable(true);
      ew_branch_add_in_max_desc.SetShape({static_cast<int64_t>(max_ptr_size)});
      ew_branch_add_in_max_desc.SetDataType(
          proto::VarType::Type::VarType_Type_FP32);
      ew_branch_add_max = graph->CreateVarNode(&ew_branch_add_in_max_desc);
      PADDLE_ENFORCE_EQ(
          AreScalesPresentForNodes(var_quant_scales, {ew_branch_add_in}),
          true,
          common::errors::InvalidArgument(
              "When conv op is running in int8 precision with branch add, the "
              "scales of branch var should be present in!"));
      float ew_branch_add_scale =
          GetScaleValueForNode(var_quant_scales, ew_branch_add_in);
      auto* conv = GetNodeFromNodesMap(nodes_map, "conv", "conv");
      PADDLE_ENFORCE_EQ(
          conv != nullptr,
          true,
          common::errors::InvalidArgument("conv node ptr can not be null"));
      auto ew_branch_add_max_tensor =
          scope->Var(ew_branch_add_max_name)->GetMutable<phi::DenseTensor>();
      ew_branch_add_max_tensor->set_type(phi::DataType::FLOAT32);
      ew_branch_add_max_tensor->Resize({max_ptr_size});
      auto* cpu_ctx = static_cast<phi::CPUContext*>(
          phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
      std::vector<float> ew_branch_add_scales(max_ptr_size,
                                              ew_branch_add_scale);
      memcpy(cpu_ctx->Alloc<float>(ew_branch_add_max_tensor),
             ew_branch_add_scales.data(),
             max_ptr_size * sizeof(float));
    }
    (*fusion_nodes_map)["branch_max"] = ew_branch_add_max;
  }
}

void Conv2dXPUFusePass::CreateFusionOutputs(
    ir::Graph* graph,
    Scope* scope,
    BlockDesc* block,
    const std::map<std::string, std::map<std::string, Node*>>& nodes_map,
    std::map<std::string, Node*>* fusion_nodes_map,
    std::string op_weights_precision,
    std::string act_type,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  auto* conv = GetNodeFromNodesMap(nodes_map, "conv", "conv");
  PADDLE_ENFORCE_EQ(
      conv != nullptr,
      true,
      common::errors::InvalidArgument("conv node ptr can not be null"));
  // output && output max
  std::string conv2d_xpu_out_name;
  Node* conv2d_out_var_node = nullptr;

  auto* ew_branch_add =
      GetNodeFromNodesMap(nodes_map, "ew_branch_add", "ew_branch_add");
  auto* bn = GetNodeFromNodesMap(nodes_map, "bn", "bn");
  auto* scale = GetNodeFromNodesMap(nodes_map, "scale", "scale");
  auto* ew_bias_add =
      GetNodeFromNodesMap(nodes_map, "ew_bias_add", "ew_bias_add");
  if (!act_type.empty()) {
    auto* act_out = GetNodeFromNodesMap(nodes_map, "act", "act_out");
    PADDLE_ENFORCE_EQ(
        act_out != nullptr,
        true,
        common::errors::InvalidArgument("act_out node ptr can not be null"));
    conv2d_xpu_out_name = act_out->Name();
    conv2d_out_var_node = act_out;
    auto* act = GetNodeFromNodesMap(nodes_map, "act", "act");
    PADDLE_ENFORCE_EQ(
        act != nullptr,
        true,
        common::errors::InvalidArgument("act node ptr can not be null"));
  } else if (ew_branch_add) {
    auto* ew_branch_add_out =
        GetNodeFromNodesMap(nodes_map, "ew_branch_add", "ew_branch_add_out");
    PADDLE_ENFORCE_EQ(ew_branch_add_out != nullptr,
                      true,
                      common::errors::InvalidArgument(
                          "ew_branch_add_out node ptr can not be null"));
    conv2d_xpu_out_name = ew_branch_add_out->Name();
    conv2d_out_var_node = ew_branch_add_out;
    PADDLE_ENFORCE_EQ(ew_branch_add != nullptr,
                      true,
                      common::errors::InvalidArgument(
                          "ew_branch_add node ptr can not be null"));
  } else if (scale) {
    auto* scale_out = GetNodeFromNodesMap(nodes_map, "scale", "scale_out");
    PADDLE_ENFORCE_EQ(
        scale_out != nullptr,
        true,
        common::errors::InvalidArgument("scale_out node ptr can not be null"));
    conv2d_xpu_out_name = scale_out->Name();
    conv2d_out_var_node = scale_out;
  } else if (bn) {
    auto* bn_out = GetNodeFromNodesMap(nodes_map, "bn", "bn_out");
    PADDLE_ENFORCE_EQ(
        bn_out != nullptr,
        true,
        common::errors::InvalidArgument("bn_out node ptr can not be null"));
    conv2d_xpu_out_name = bn_out->Name();
    conv2d_out_var_node = bn_out;
  } else if (ew_bias_add) {
    auto* ew_bias_add_out =
        GetNodeFromNodesMap(nodes_map, "ew_bias_add", "ew_bias_add_out");
    PADDLE_ENFORCE_EQ(ew_bias_add_out != nullptr,
                      true,
                      common::errors::InvalidArgument(
                          "ew_bias_add_out node ptr can not be null"));
    conv2d_xpu_out_name = ew_bias_add_out->Name();
    conv2d_out_var_node = ew_bias_add_out;
  } else {
    auto* conv_out = GetNodeFromNodesMap(nodes_map, "conv", "conv_out");
    PADDLE_ENFORCE_EQ(
        conv_out != nullptr,
        true,
        common::errors::InvalidArgument("conv_out node ptr can not be null"));
    conv2d_xpu_out_name = conv_out->Name();
    conv2d_out_var_node = conv_out;
    auto* conv = GetNodeFromNodesMap(nodes_map, "conv", "conv");
    PADDLE_ENFORCE_EQ(
        conv != nullptr,
        true,
        common::errors::InvalidArgument("conv node ptr can not be null"));
  }
  (*fusion_nodes_map)["out"] = conv2d_out_var_node;

  // Create out max in
  if (op_weights_precision == "int8" &&
      AreScalesPresentForNodes(var_quant_scales, {conv2d_out_var_node})) {
    std::string conv_out_max_in_name = conv2d_xpu_out_name + "_max_in";
    int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
    VarDesc conv_out_max_in_desc(conv_out_max_in_name);
    conv_out_max_in_desc.SetPersistable(true);
    conv_out_max_in_desc.SetShape({static_cast<int64_t>(max_ptr_size)});
    conv_out_max_in_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    Node* conv2d_xpu_out_max_in = graph->CreateVarNode(&conv_out_max_in_desc);
    auto* block_out_max_in_desc = block->Var(conv_out_max_in_name);
    block_out_max_in_desc->SetPersistable(conv_out_max_in_desc.Persistable());
    block_out_max_in_desc->SetShape(conv_out_max_in_desc.GetShape());
    block_out_max_in_desc->SetDataType(conv_out_max_in_desc.GetDataType());

    float output_scale =
        GetScaleValueForNode(var_quant_scales, conv2d_out_var_node);
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
           scope->Var(conv_out_max_in_name)->GetMutable<phi::DenseTensor>());
    (*fusion_nodes_map)["out_max_in"] = conv2d_xpu_out_max_in;
  }

  // Create out max
  std::string conv_out_max_name = conv2d_xpu_out_name + "_max";
  VarDesc conv_out_max_desc(conv_out_max_name);
  Node* conv2d_xpu_out_max = graph->CreateVarNode(&conv_out_max_desc);
  (*fusion_nodes_map)["out_max"] = conv2d_xpu_out_max;
}

int Conv2dXPUFusePass::ApplyImpl(ir::Graph* graph,
                                 const std::string& conv_type,
                                 const std::string& act_type,
                                 bool with_conv_bias,
                                 bool with_bn,
                                 bool with_scale,
                                 bool with_branch_x,
                                 bool with_branch_y) const {
  GraphPatternDetector gpd;
  patterns::Conv2dXPUPattern pattern(gpd.mutable_pattern(),
                                     name_scope_,
                                     conv_type,
                                     act_type,
                                     with_conv_bias,
                                     with_bn,
                                     with_scale,
                                     with_branch_x,
                                     with_branch_y);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
  std::unordered_map<std::string, std::vector<float>> var_quant_scales =
      GetQuantInfoFromTheGraph(graph, "has_quant_info", "var_quant_scales");
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle Conv2dXPUFusePass fuse";
    std::map<std::string, std::map<std::string, Node*>> nodes_map;
    GET_IR_NODE(conv);
    GET_IR_NODE(ew_bias_add);
    GET_IR_NODE(bn);
    GET_IR_NODE(scale);
    GET_IR_NODE(ew_branch_add);
    GET_IR_NODE(act);
    /* Get variable node's name*/
    GET_IR_NODE(input);
    GET_IR_NODE(conv_out);
    GET_IR_NODE(ew_bias_add_y);
    GET_IR_NODE(ew_bias_add_out);
    GET_IR_NODE(bn_bias);
    GET_IR_NODE(bn_mean);
    GET_IR_NODE(bn_scale);
    GET_IR_NODE(bn_var);
    GET_IR_NODE(bn_out);
    GET_IR_NODE(bn_var_out);
    GET_IR_NODE(bn_mean_out);
    GET_IR_NODE(bn_saved_var);
    GET_IR_NODE(bn_saved_mean);
    GET_IR_NODE(scale_out);
    GET_IR_NODE(ew_branch_add_in);
    GET_IR_NODE(ew_branch_add_out);
    GET_IR_NODE(act_out);

    nodes_map.insert(
        {"conv", {{"conv", conv}, {"input", input}, {"conv_out", conv_out}}});
    nodes_map.insert({"ew_bias_add",
                      {{"ew_bias_add", ew_bias_add},
                       {"ew_bias_add_y", ew_bias_add_y},
                       {"ew_bias_add_out", ew_bias_add_out}}});
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
    nodes_map.insert({"scale", {{"scale", scale}, {"scale_out", scale_out}}});
    nodes_map.insert({"ew_branch_add",
                      {{"ew_branch_add", ew_branch_add},
                       {"ew_branch_add_in", ew_branch_add_in},
                       {"ew_branch_add_out", ew_branch_add_out}}});
    nodes_map.insert({"act", {{"act", act}, {"act_out", act_out}}});

    std::map<std::string, Node*> fusion_nodes_map{{"x", nullptr},
                                                  {"x_max", nullptr},
                                                  {"filter", nullptr},
                                                  {"filter_max", nullptr},
                                                  {"bias", nullptr},
                                                  {"branch", nullptr},
                                                  {"branch_max", nullptr},
                                                  {"scale_max", nullptr},
                                                  {"out_max_in", nullptr},
                                                  {"out", nullptr},
                                                  {"out_max", nullptr}};
    auto filter_name_0 = conv->Op()->Input("Filter")[0];
    Node* filter_node = FindNodeWithName(graph, filter_name_0);
    if (!filter_node->Var()->Persistable()) return;

    auto filter_data_type =
        scope->FindVar(filter_name_0)->GetMutable<phi::DenseTensor>()->dtype();
    std::string op_weights_precision = "float32";
    if (filter_data_type == phi::DataType::INT8) {
      op_weights_precision = "int8";
    } else if (filter_data_type == phi::DataType::FLOAT16) {
      op_weights_precision = "float16";
    }
    VLOG(4) << "Conv2d fusion fuse pass is running on " << op_weights_precision
            << " precision!";
    auto* block = conv->Op()->Block();
    CreateFusionWeightsAndBias(graph,
                               scope,
                               block,
                               nodes_map,
                               &fusion_nodes_map,
                               with_conv_bias,
                               with_bn,
                               with_scale,
                               op_weights_precision,
                               &var_quant_scales);
    CreateFusionInputs(graph,
                       scope,
                       block,
                       nodes_map,
                       &fusion_nodes_map,
                       op_weights_precision,
                       &var_quant_scales);
    CreateFusionBranch(graph,
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
                        act_type,
                        &var_quant_scales);

    framework::OpDesc conv2d_xpu_op_desc(block);
    conv2d_xpu_op_desc.SetType("conv2d_xpu");
    conv2d_xpu_op_desc.SetInput("x", {fusion_nodes_map["x"]->Name()});
    if (fusion_nodes_map["x_max"]) {
      conv2d_xpu_op_desc.SetInput("x_max", {fusion_nodes_map["x_max"]->Name()});
    }
    conv2d_xpu_op_desc.SetInput("filter", {fusion_nodes_map["filter"]->Name()});
    conv2d_xpu_op_desc.SetInput("filter_max",
                                {fusion_nodes_map["filter_max"]->Name()});
    if (fusion_nodes_map["scale_max"]) {
      conv2d_xpu_op_desc.SetInput("scale_max",
                                  {fusion_nodes_map["scale_max"]->Name()});
    }
    if (fusion_nodes_map["out_max_in"]) {
      conv2d_xpu_op_desc.SetInput("out_max_in",
                                  {fusion_nodes_map["out_max_in"]->Name()});
    }
    conv2d_xpu_op_desc.SetOutput("out", {fusion_nodes_map["out"]->Name()});
    conv2d_xpu_op_desc.SetOutput("out_max",
                                 {fusion_nodes_map["out_max"]->Name()});
    if (with_conv_bias || with_bn) {
      PADDLE_ENFORCE_EQ(
          fusion_nodes_map["bias"] != nullptr,
          true,
          common::errors::InvalidArgument(
              "fusion_nodes_map['bias'] node ptr can not be null"));
      conv2d_xpu_op_desc.SetInput("bias", {fusion_nodes_map["bias"]->Name()});
    }
    // set ew_branch_add input node
    if (ew_branch_add != nullptr) {
      PADDLE_ENFORCE_EQ(
          fusion_nodes_map["branch"] != nullptr,
          true,
          common::errors::InvalidArgument(
              "fusion_nodes_map['branch'] node ptr can not be null"));
      conv2d_xpu_op_desc.SetInput("branch",
                                  {fusion_nodes_map["branch"]->Name()});
      if (fusion_nodes_map["branch_max"]) {
        conv2d_xpu_op_desc.SetInput("branch_max",
                                    {fusion_nodes_map["branch_max"]->Name()});
      }
    }
    // set attrs of conv2d_xpu
    float act_param_ = 0.0f;
    if (!act_type.empty()) {
      if (act_type == "leaky_relu") {
        act_param_ = PADDLE_GET_CONST(float, act->Op()->GetAttr("alpha"));
      } else if (act_type == "hard_sigmoid") {
        act_param_ = PADDLE_GET_CONST(float, act->Op()->GetAttr("slope"));
      }
    }
    conv2d_xpu_op_desc.SetAttr("act_type", ConvertActivationType(act_type));
    conv2d_xpu_op_desc.SetAttr("act_param", act_param_);
    conv2d_xpu_op_desc.SetAttr(
        "padding_algorithm",
        conv->Op()->GetAttrIfExists<std::string>("padding_algorithm"));
    auto conv_paddings =
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("paddings"));
    if (conv_paddings.size() == 2) {
      for (int i = 0; i < 2; i++) {
        int copy_pad = *(conv_paddings.begin() + 2 * i);
        conv_paddings.insert(conv_paddings.begin() + 2 * i + 1, copy_pad);
      }
    }
    PADDLE_ENFORCE_EQ(conv_paddings.size(),
                      4UL,
                      common::errors::InvalidArgument(
                          "padding length should be 4, but received %d, ",
                          conv_paddings.size()));
    conv2d_xpu_op_desc.SetAttr(
        "dilations",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("dilations")));
    conv2d_xpu_op_desc.SetAttr(
        "groups", PADDLE_GET_CONST(int, conv->Op()->GetAttr("groups")));
    conv2d_xpu_op_desc.SetAttr(
        "strides",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("strides")));
    conv2d_xpu_op_desc.SetAttr("paddings", conv_paddings);
    // out_dtype is same to input precision
    conv2d_xpu_op_desc.SetAttr("out_dtype",
                               fusion_nodes_map["x"]->Var()->GetDataType());

    // Link node
    auto* conv2d_xpu = graph->CreateOpNode(&conv2d_xpu_op_desc);
    IR_NODE_LINK_TO(fusion_nodes_map["x"], conv2d_xpu);
    if (fusion_nodes_map["x_max"]) {
      IR_NODE_LINK_TO(fusion_nodes_map["x_max"], conv2d_xpu);
    }
    IR_NODE_LINK_TO(fusion_nodes_map["filter"], conv2d_xpu);
    IR_NODE_LINK_TO(fusion_nodes_map["filter_max"], conv2d_xpu);
    if (fusion_nodes_map["scale_max"]) {
      IR_NODE_LINK_TO(fusion_nodes_map["scale_max"], conv2d_xpu);
    }
    if (fusion_nodes_map["bias"]) {
      SAFE_IR_NODE_LINK_TO(fusion_nodes_map["bias"], conv2d_xpu);
    }
    if (fusion_nodes_map["branch"]) {
      IR_NODE_LINK_TO(fusion_nodes_map["branch"], conv2d_xpu);
    }
    if (fusion_nodes_map["branch_max"]) {
      IR_NODE_LINK_TO(fusion_nodes_map["branch_max"], conv2d_xpu);
    }
    if (fusion_nodes_map["out_max_in"]) {
      IR_NODE_LINK_TO(fusion_nodes_map["out_max_in"], conv2d_xpu);
    }
    IR_NODE_LINK_TO(conv2d_xpu, fusion_nodes_map["out"]);
    IR_NODE_LINK_TO(conv2d_xpu, fusion_nodes_map["out_max"]);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes;
    if (conv != nullptr) {
      delete_nodes.insert(conv);
    }
    if (scale != nullptr) {
      delete_nodes.insert(scale);
    }
    if (bn != nullptr) {
      delete_nodes.insert(bn);
    }
    if (ew_bias_add != nullptr) {
      delete_nodes.insert(ew_bias_add);
    }
    if (ew_branch_add != nullptr) {
      delete_nodes.insert(ew_branch_add);
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

REGISTER_PASS(conv2d_xpu_fuse_pass, paddle::framework::ir::Conv2dXPUFusePass);

REGISTER_PASS_CAPABILITY(conv2d_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "conv2d_xpu", 0));
