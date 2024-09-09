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

struct Conv2dTransposeXPUPattern : public PatternBase {
  Conv2dTransposeXPUPattern(PDPattern* pattern,
                            const std::string& name_scope,
                            const std::string& act_type,
                            bool with_ew_bias,
                            bool with_bn);
  // operator
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(ew_bias_add);
  PATTERN_DECL_NODE(bn);
  PATTERN_DECL_NODE(act);
  // conv param
  PATTERN_DECL_NODE(input);
  PATTERN_DECL_NODE(conv_filter);
  PATTERN_DECL_NODE(conv_out);
  // ew param
  PATTERN_DECL_NODE(ew_bias_add_y);
  PATTERN_DECL_NODE(ew_bias_add_out);
  // bn param
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_mean);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_var);
  PATTERN_DECL_NODE(bn_out);
  PATTERN_DECL_NODE(bn_var_out);
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_saved_var);
  PATTERN_DECL_NODE(bn_saved_mean);
  // act param
  PATTERN_DECL_NODE(act_out);

 private:
  std::string act_type_;
  bool with_bn_;
  bool with_ew_bias_;
};

Conv2dTransposeXPUPattern::Conv2dTransposeXPUPattern(
    PDPattern* pattern,
    const std::string& name_scope,
    const std::string& act_type,
    bool with_ew_bias,
    bool with_bn)
    : PatternBase(pattern, name_scope, name_scope),
      act_type_(act_type),
      with_bn_(with_bn),
      with_ew_bias_(with_ew_bias) {
  // deconv op
  auto conv = pattern->NewNode(conv_repr())->assert_is_op("conv2d_transpose");
  auto input = pattern->NewNode(input_repr())
                   ->assert_is_op_input("conv2d_transpose", "Input")
                   ->AsInput()
                   ->assert_more([](Node* node) {
                     return node->Var()->GetShape().size() == 4;
                   });
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input("conv2d_transpose", "Filter")
                         ->AsInput()
                         ->assert_is_persistable_var();
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output("conv2d_transpose", "Output")
                      ->assert_has_n_outputs(1);
  conv->LinksFrom({input, conv_filter}).LinksTo({conv_out});

  // elementwise op
  PDNode* ew_bias_add = nullptr;
  PDNode* ew_bias_add_y = nullptr;
  PDNode* ew_bias_add_out = nullptr;
  if (with_ew_bias_) {
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
    if (with_bn_ || !act_type_.empty()) {
      ew_bias_add_out->assert_has_n_outputs(1);
    }
    ew_bias_add->LinksFrom({conv_out, ew_bias_add_y})
        .LinksTo({ew_bias_add_out});
  } else {
    ew_bias_add_out = conv_out;
  }

  // batch_norm op
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
    bn->LinksFrom({ew_bias_add_out, bn_bias, bn_mean, bn_scale, bn_var})
        .LinksTo(
            {bn_out, bn_mean_out, bn_var_out, bn_saved_mean, bn_saved_var});
  } else {
    bn_out = ew_bias_add_out;
  }

  // act
  PDNode* act = nullptr;
  PDNode* act_out = nullptr;
  if (!act_type_.empty()) {
    bn_out->assert_is_op_input(act_type_, "X");
    act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
    act_out =
        pattern->NewNode(act_out_repr())->assert_is_op_output(act_type_, "Out");
    act->LinksFrom({bn_out}).LinksTo({act_out});
  } else {
    act_out = bn_out;
  }
  act_out->AsOutput();
}
}  // namespace patterns

/* fuse conv2d block in resnet50-like model to xpu_conv2d op    */
/* For example:                                                 */
/* graph[1]: sub block                                          */
/*                     in_Input                                 */
/*                       |                                      */
/*                       |                                      */
/*                conv2d_transpose----in_Filter                 */
/*                       |                                      */
/*                       |                                      */
/*                  elementwise_add -----ew_add                 */
/*                       |                                      */
/*                       |                                      */
/*                   batch_norm ------in_Bias                   */
/*                       |                                      */
/*                       |                                      */
/*                      act                                     */
/*                       |                                      */
/*                       |                                      */
/*                     out_Out                                  */
/*                                                              */
class Conv2dTransposeXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                const std::string& act_type,
                bool with_ew_bias,
                bool with_bn) const;

  const std::string name_scope_{"conv2d_transpose_xpu_fuse_pass"};
};

void Conv2dTransposeXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto with_bn : {true, false}) {
    for (auto with_ew_bias : {true, false}) {
      for (auto act_type : {"relu", ""}) {
        found_subgraph_count +=
            ApplyImpl(graph, act_type, with_ew_bias, with_bn);
      }
    }
  }
  AddStatis(found_subgraph_count);
}

int Conv2dTransposeXPUFusePass::ApplyImpl(ir::Graph* graph,
                                          const std::string& act_type,
                                          bool with_ew_bias,
                                          bool with_bn) const {
  GraphPatternDetector gpd;
  patterns::Conv2dTransposeXPUPattern pattern(
      gpd.mutable_pattern(), name_scope_, act_type, with_ew_bias, with_bn);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle Conv2dTransposeXPUFusePass fuse";
    /* declare operator node's name */
    GET_IR_NODE(conv);
    GET_IR_NODE(ew_bias_add);
    GET_IR_NODE(bn);
    GET_IR_NODE(act);
    /* declare variable node's name*/
    GET_IR_NODE(input);
    GET_IR_NODE(conv_filter);
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
    GET_IR_NODE(act_out);
    auto* block = conv->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));

    // recompute bias and weight for conv2d_transpose_xpu op
    auto* filter_t =
        scope->FindVar(conv_filter->Name())->GetMutable<phi::DenseTensor>();
    // conv_filter fp16 --> fp32
    auto tensor_type = filter_t->dtype();
    if (tensor_type == phi::DataType::FLOAT16) {
      CastToFp32(filter_t, nullptr);
    }
    auto filter_dims = filter_t->dims();
    bool has_bias = with_bn || with_ew_bias;
    Node* fusion_bias_node = nullptr;
    int groups = PADDLE_GET_CONST(int, conv->Op()->GetAttr("groups"));
    int out_c = filter_dims[1] * groups;

    // ew bias
    if (with_ew_bias) {
      auto* ew_bias_add_y_t =
          scope->FindVar(ew_bias_add_y->Name())->GetMutable<phi::DenseTensor>();
      auto ew_bias_add_y_dims = ew_bias_add_y_t->dims();
      PADDLE_ENFORCE_EQ(out_c,
                        ew_bias_add_y_dims[0],
                        common::errors::InvalidArgument(
                            "the shape[%d] of elewise bias tensor "
                            "must equal out_channel[%d] of conv",
                            ew_bias_add_y_dims[0],
                            out_c));
      PrepareBias(graph, scope, block, ew_bias_add_y, &fusion_bias_node);
    }
    // bn
    if (with_bn) {
      auto bn_bias_t =
          scope->Var(bn_bias->Name())->GetMutable<phi::DenseTensor>();
      PADDLE_ENFORCE_EQ(
          out_c,
          bn_bias_t->dims()[0],
          common::errors::InvalidArgument("the shape[%d] of bn bias tensor "
                                          "must equal out_channel[%d] of conv",
                                          bn_bias_t->dims()[0],
                                          out_c));
      auto bn_scale_t =
          scope->Var(bn_scale->Name())->GetMutable<phi::DenseTensor>();
      auto bn_mean_t =
          scope->Var(bn_mean->Name())->GetMutable<phi::DenseTensor>();
      auto bn_var_t =
          scope->Var(bn_var->Name())->GetMutable<phi::DenseTensor>();
      float* filter_ptr = filter_t->data<float>();
      float* bn_scale_ptr = bn_scale_t->data<float>();
      float* bn_bias_ptr = bn_bias_t->data<float>();
      float* bn_mean_ptr = bn_mean_t->data<float>();
      float* bn_var_ptr = bn_var_t->data<float>();
      auto mean_len = bn_mean_t->numel();  // oc

      float epsilon = PADDLE_GET_CONST(float, bn->Op()->GetAttr("epsilon"));
      // bias
      if (fusion_bias_node) {
        auto fusion_bias_t = scope->Var(fusion_bias_node->Name())
                                 ->GetMutable<phi::DenseTensor>();
        float* fusion_bias_ptr = fusion_bias_t->data<float>();
        for (int i = 0; i < mean_len; ++i) {
          bn_scale_ptr[i] = bn_scale_ptr[i] / sqrtf(bn_var_ptr[i] + epsilon);
          fusion_bias_ptr[i] =
              bn_bias_ptr[i] +
              (fusion_bias_ptr[i] - bn_mean_ptr[i]) * bn_scale_ptr[i];
        }
      } else {
        PrepareBias(graph, scope, block, bn_bias, &fusion_bias_node);
        auto fusion_bias_t = scope->Var(fusion_bias_node->Name())
                                 ->GetMutable<phi::DenseTensor>();
        float* fusion_bias_ptr = fusion_bias_t->data<float>();
        for (int i = 0; i < mean_len; ++i) {
          bn_scale_ptr[i] = bn_scale_ptr[i] / sqrtf(bn_var_ptr[i] + epsilon);
          fusion_bias_ptr[i] += (0.0f - bn_mean_ptr[i]) * bn_scale_ptr[i];
        }
      }
      // compute new conv_weight, weight is ic-oc/g-h-w
      int cout_group = filter_dims[1];
      int cin_group = filter_dims[0] / groups;
      int c_size = cout_group * filter_dims[2] * filter_dims[3];
      int hw = filter_dims[2] * filter_dims[3];
      for (int g = 0; g < groups; g++) {
        for (int k = 0; k < cin_group; ++k) {
          for (int i = 0; i < cout_group; ++i) {
            auto ptr_row =
                filter_ptr + g * cin_group * c_size + k * c_size + i * hw;
            for (int j = 0; j < hw; ++j) {
              ptr_row[j] *= bn_scale_ptr[g * cout_group + i];
            }
          }
        }
      }
    }
    // filter max
    Node* filter_int16 = nullptr;
    Node* filter_max = nullptr;
    Node* scale_max = nullptr;
    PrepareWeight<float, int16_t>(graph,
                                  scope,
                                  block,
                                  conv_filter,
                                  &filter_int16,
                                  &filter_max,
                                  &scale_max,
                                  false,
                                  std::vector<float>({}));
    // output && output max
    std::string conv2d_xpu_out_name;
    if (!act_type.empty()) {
      conv2d_xpu_out_name = act_out->Name();
    } else if (with_bn) {
      conv2d_xpu_out_name = bn_out->Name();
    } else if (with_ew_bias) {
      conv2d_xpu_out_name = ew_bias_add_out->Name();
    } else {
      conv2d_xpu_out_name = conv_out->Name();
    }
    std::string conv_out_max_name = conv2d_xpu_out_name + "_max";
    VarDesc conv_out_max_desc(conv_out_max_name);
    Node* conv2d_xpu_out_max = graph->CreateVarNode(&conv_out_max_desc);
    // Generate conv2d_xpu op
    framework::OpDesc conv2d_xpu_op_desc(block);
    // set input&output var
    conv2d_xpu_op_desc.SetType("conv2d_transpose_xpu");
    conv2d_xpu_op_desc.SetInput("x", {input->Name()});
    conv2d_xpu_op_desc.SetInput("filter", {filter_int16->Name()});
    conv2d_xpu_op_desc.SetInput("filter_max", {filter_max->Name()});
    conv2d_xpu_op_desc.SetOutput("out", {conv2d_xpu_out_name});
    conv2d_xpu_op_desc.SetOutput("out_max", {conv_out_max_name});
    // set fusion_bias input node
    if (has_bias) {
      conv2d_xpu_op_desc.SetInput("bias", {fusion_bias_node->Name()});
    }
    conv2d_xpu_op_desc.SetAttr("has_bias", has_bias);
    // set attrs of conv2d_xpu
    if (!act_type.empty()) {
      conv2d_xpu_op_desc.SetAttr("with_act", true);
    } else {
      conv2d_xpu_op_desc.SetAttr("with_act", false);
    }
    conv2d_xpu_op_desc.SetAttr("act_type", act_type);
    conv2d_xpu_op_desc.SetAttr(
        "padding_algorithm",
        conv->Op()->GetAttrIfExists<std::string>("padding_algorithm"));
    conv2d_xpu_op_desc.SetAttr(
        "output_size",
        conv->Op()->GetAttrIfExists<std::vector<int>>("output_size"));
    conv2d_xpu_op_desc.SetAttr(
        "output_padding",
        conv->Op()->GetAttrIfExists<std::vector<int>>("output_padding"));
    conv2d_xpu_op_desc.SetAttr(
        "dilations",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("dilations")));
    conv2d_xpu_op_desc.SetAttr(
        "paddings",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("paddings")));
    conv2d_xpu_op_desc.SetAttr(
        "groups", PADDLE_GET_CONST(int, conv->Op()->GetAttr("groups")));
    conv2d_xpu_op_desc.SetAttr(
        "strides",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("strides")));
    conv2d_xpu_op_desc.SetAttr(
        "data_format", conv->Op()->GetAttrIfExists<std::string>("data_format"));

    auto* conv2d_xpu = graph->CreateOpNode(&conv2d_xpu_op_desc);
    IR_NODE_LINK_TO(input, conv2d_xpu);
    IR_NODE_LINK_TO(filter_int16, conv2d_xpu);
    IR_NODE_LINK_TO(filter_max, conv2d_xpu);
    if (has_bias) {
      SAFE_IR_NODE_LINK_TO(fusion_bias_node, conv2d_xpu);
    }
    if (act_out) {
      IR_NODE_LINK_TO(conv2d_xpu, act_out);
    } else if (bn_out) {
      IR_NODE_LINK_TO(conv2d_xpu, bn_out);
    } else if (ew_bias_add_out) {
      IR_NODE_LINK_TO(conv2d_xpu, ew_bias_add_out);
    } else {
      IR_NODE_LINK_TO(conv2d_xpu, conv_out);
    }
    IR_NODE_LINK_TO(conv2d_xpu, conv2d_xpu_out_max);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {conv};
    if (act != nullptr) {
      delete_nodes.insert(act);
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
    if (ew_bias_add) {
      delete_nodes.insert(ew_bias_add);
      delete_nodes.insert(ew_bias_add_y);
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

REGISTER_PASS(conv2d_transpose_xpu_fuse_pass,
              paddle::framework::ir::Conv2dTransposeXPUFusePass);

REGISTER_PASS_CAPABILITY(conv2d_transpose_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "conv2d_transpose_xpu", 0));
