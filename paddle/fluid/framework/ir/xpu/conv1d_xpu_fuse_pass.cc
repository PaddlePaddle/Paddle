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

struct Conv1dXPUPattern : public PatternBase {
  Conv1dXPUPattern(PDPattern* pattern,
                   const std::string& name_scope,
                   const std::string& conv_type,
                   const std::string& act_type,
                   bool with_conv_bias,
                   bool with_bn,
                   bool with_branch_x,
                   bool with_branch_y);
  // declare operator node's name
  PATTERN_DECL_NODE(unsqueeze2);
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(ew_bias_add);
  PATTERN_DECL_NODE(squeeze2);
  PATTERN_DECL_NODE(bn);
  PATTERN_DECL_NODE(ew_branch_add);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(conv_input);
  PATTERN_DECL_NODE(conv_filter);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(ew_bias_add_y);
  PATTERN_DECL_NODE(ew_bias_add_out);
  PATTERN_DECL_NODE(squeeze2_out);
  PATTERN_DECL_NODE(bn_bias);
  PATTERN_DECL_NODE(bn_mean);
  PATTERN_DECL_NODE(bn_scale);
  PATTERN_DECL_NODE(bn_var);
  PATTERN_DECL_NODE(bn_out);
  PATTERN_DECL_NODE(bn_var_out);
  PATTERN_DECL_NODE(bn_mean_out);
  PATTERN_DECL_NODE(bn_saved_var);
  PATTERN_DECL_NODE(bn_saved_mean);
  PATTERN_DECL_NODE(ew_branch_add_in);
  PATTERN_DECL_NODE(ew_branch_add_out);
  PATTERN_DECL_NODE(act_out);

 private:
  std::string conv_type_;
  std::string act_type_;
  bool with_conv_bias_{false};
  bool with_bn_{false};
  bool with_branch_{false};
  bool with_branch_x_{false};
  bool with_branch_y_{false};
};

Conv1dXPUPattern::Conv1dXPUPattern(PDPattern* pattern,
                                   const std::string& name_scope,
                                   const std::string& conv_type,
                                   const std::string& act_type,
                                   bool with_conv_bias,
                                   bool with_bn,
                                   bool with_branch_x,
                                   bool with_branch_y)
    : PatternBase(pattern, name_scope, name_scope),
      conv_type_(conv_type),
      act_type_(act_type),
      with_conv_bias_(with_conv_bias),
      with_bn_(with_bn),
      with_branch_(with_branch_x || with_branch_y),
      with_branch_x_(with_branch_x),
      with_branch_y_(with_branch_y) {
  auto x = pattern->NewNode(x_repr())
               ->assert_is_op_input("unsqueeze2", "X")
               ->assert_more([](Node* node) {
                 auto x_shape = node->Var()->GetShape();
                 size_t x_rank = x_shape.size();
                 return x_rank == 3;
               });
  auto unsqueeze2 =
      pattern->NewNode(unsqueeze2_repr())
          ->assert_is_op("unsqueeze2")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto axes_array =
                op_desc->GetAttrIfExists<std::vector<int>>("axes");
            return axes_array == std::vector<int>{-2} ||
                   axes_array == std::vector<int>{2};
          });
  auto conv_input = pattern->NewNode(conv_input_repr())
                        ->assert_is_op_output("unsqueeze2", "Out")
                        ->assert_is_op_input(conv_type_, "Input");
  unsqueeze2->LinksFrom({x}).LinksTo({conv_input});
  auto conv = pattern->NewNode(conv_repr())->assert_is_op(conv_type_);
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input(conv_type_, "Filter")
                         ->assert_is_persistable_var()
                         ->assert_more([](Node* node) {
                           auto filter_shape = node->Var()->GetShape();
                           size_t filter_rank = filter_shape.size();
                           return filter_rank == 4 && filter_shape[2] == 1;
                         });
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output(conv_type_, "Output")
                      ->assert_has_n_outputs(1);
  conv->LinksFrom({conv_input, conv_filter}).LinksTo({conv_out});
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
    if (with_bn_ || with_branch_ || !act_type_.empty()) {
      ew_bias_add_out->assert_has_n_outputs(1);
    }
    ew_bias_add->LinksFrom({conv_out, ew_bias_add_y})
        .LinksTo({ew_bias_add_out});
  } else {
    ew_bias_add_out = conv_out;
  }
  // squeeze2 op
  ew_bias_add_out->assert_is_op_input("squeeze2", "X");
  auto squeeze2 = pattern->NewNode(squeeze2_repr())
                      ->assert_is_op("squeeze2")
                      ->assert_more([](Node* node) {
                        auto* op_desc = node->Op();
                        auto axes_array =
                            op_desc->GetAttrIfExists<std::vector<int>>("axes");
                        return axes_array == std::vector<int>{-2} ||
                               axes_array == std::vector<int>{2};
                      });
  auto squeeze2_out = pattern->NewNode(squeeze2_out_repr())
                          ->assert_is_op_output("squeeze2", "Out");
  squeeze2->LinksFrom({ew_bias_add_out}).LinksTo({squeeze2_out});
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
  PDNode* act = nullptr;
  PDNode* act_out = nullptr;
  // batch_norm op
  if (with_bn_) {
    squeeze2_out->assert_is_op_input("batch_norm", "X");
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
    if (with_branch_ || !act_type_.empty()) {
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
    bn->LinksFrom({squeeze2_out, bn_bias, bn_mean, bn_scale, bn_var})
        .LinksTo(
            {bn_out, bn_mean_out, bn_var_out, bn_saved_mean, bn_saved_var});
  } else {
    bn_out = squeeze2_out;
  }
  // ew_branch_add op
  if (with_branch_) {
    if (with_branch_x_) {
      bn_out->assert_is_op_input("elementwise_add", "Y");
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
    ew_branch_add_out = bn_out;
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
For example:
graph[1]: sub block
                    in_Input
                      |
                      |
                    conv2d----in_Filter
                      |
                      |
                 elementwise_add -----conv2d_Bias
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
  in_Branch ------- conv1d_xpu ------ in_Bias
                       |    \
                       |     \
                       |      out_OutputMax
                    out_Output
*/
class Conv1dXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                const std::string& conv_type,
                const std::string& act_type,
                bool with_conv_bias,
                bool with_bn,
                bool with_branch_x,
                bool with_branch_y) const;

  const std::string name_scope_{"conv1d_xpu_fuse_pass"};
};

void Conv1dXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto conv_type : {"conv2d"}) {
    for (auto with_conv_bias : {true}) {
      for (auto with_bn : {true, false}) {
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
                                                with_branch_x,
                                                with_branch_y);
            }
          }
        }
      }
    }
  }
  AddStatis(found_subgraph_count);
}

int Conv1dXPUFusePass::ApplyImpl(ir::Graph* graph,
                                 const std::string& conv_type,
                                 const std::string& act_type,
                                 bool with_conv_bias,
                                 bool with_bn,
                                 bool with_branch_x,
                                 bool with_branch_y) const {
  GraphPatternDetector gpd;
  patterns::Conv1dXPUPattern pattern(gpd.mutable_pattern(),
                                     name_scope_,
                                     conv_type,
                                     act_type,
                                     with_conv_bias,
                                     with_bn,
                                     with_branch_x,
                                     with_branch_y);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle Conv1dXPUFusePass fuse";
    /* declare operator node's name */
    GET_IR_NODE(unsqueeze2);
    GET_IR_NODE(conv);
    GET_IR_NODE(ew_bias_add);
    GET_IR_NODE(squeeze2);
    GET_IR_NODE(bn);
    GET_IR_NODE(ew_branch_add);
    GET_IR_NODE(act);
    /* declare variable node's name*/
    GET_IR_NODE(x);
    GET_IR_NODE(conv_input);
    GET_IR_NODE(conv_filter);
    GET_IR_NODE(conv_out);
    GET_IR_NODE(ew_bias_add_y);
    GET_IR_NODE(squeeze2_out);
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
    GET_IR_NODE(ew_branch_add_in);
    GET_IR_NODE(ew_branch_add_out);
    GET_IR_NODE(act_out);
    auto* block = conv->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));

    // recompute bias and weight for conv1d_xpu op
    // update shape of conv_filter
    VLOG(4) << "--- deal with conv_filter";
    auto* filter_t =
        scope->GetVar(conv_filter->Name())->GetMutable<phi::DenseTensor>();
    auto filter_dims = filter_t->dims();
    auto original_f_dims =
        common::make_ddim({filter_dims[0], filter_dims[1], filter_dims[3]});
    filter_t->Resize(original_f_dims);
    filter_dims = original_f_dims;
    // conv_filter fp16 --> fp32
    auto tensor_type = filter_t->dtype();
    if (tensor_type == phi::DataType::FLOAT16) {
      CastToFp32(filter_t, nullptr);
    }
    bool has_bias = with_bn || with_conv_bias;
    // Create conv_fusion_bias (conv bias) variable
    Node* fusion_bias_node = nullptr;
    if (has_bias) {
      if (with_conv_bias) {
        auto* ew_bias_add_y_t = scope->GetVar(ew_bias_add_y->Name())
                                    ->GetMutable<phi::DenseTensor>();
        auto ew_bias_add_y_dims = ew_bias_add_y_t->dims();
        PADDLE_ENFORCE_EQ(filter_dims[0],
                          ew_bias_add_y_dims[0],
                          common::errors::InvalidArgument(
                              "the shape[%d] of elewise bias tensor "
                              "must equal out_channel[%d] of conv",
                              ew_bias_add_y_dims[0],
                              filter_dims[0]));
        PrepareBias(graph, scope, block, ew_bias_add_y, &fusion_bias_node);
      }
      if (with_bn) {
        auto bn_bias_t =
            scope->Var(bn_bias->Name())->GetMutable<phi::DenseTensor>();
        PADDLE_ENFORCE_EQ(filter_dims[0],
                          bn_bias_t->dims()[0],
                          common::errors::InvalidArgument(
                              "the shape[%d] of bn bias tensor "
                              "must equal out_channel[%d] of conv",
                              bn_bias_t->dims()[0],
                              filter_dims[0]));
        auto bn_scale_t =
            scope->Var(bn_scale->Name())->GetMutable<phi::DenseTensor>();
        auto bn_mean_t =
            scope->Var(bn_mean->Name())->GetMutable<phi::DenseTensor>();
        auto bn_var_t =
            scope->Var(bn_var->Name())->GetMutable<phi::DenseTensor>();
        float* filter_ptr = filter_t->mutable_data<float>(phi::CPUPlace());
        float* bn_scale_ptr = bn_scale_t->mutable_data<float>(phi::CPUPlace());
        float* bn_bias_ptr = bn_bias_t->mutable_data<float>(phi::CPUPlace());
        float* bn_mean_ptr = bn_mean_t->mutable_data<float>(phi::CPUPlace());
        float* bn_var_ptr = bn_var_t->mutable_data<float>(phi::CPUPlace());
        auto mean_len = bn_mean_t->numel();
        auto filter_len = filter_t->numel();
        auto filter_stride = filter_len / mean_len;
        float epsilon = PADDLE_GET_CONST(float, bn->Op()->GetAttr("epsilon"));
        if (!with_conv_bias) {  // prev node is conv
          PrepareBias(graph, scope, block, bn_bias, &fusion_bias_node);
        }
        auto fusion_bias_t = scope->Var(fusion_bias_node->Name())
                                 ->GetMutable<phi::DenseTensor>();
        float* fusion_bias_ptr =
            fusion_bias_t->mutable_data<float>(phi::CPUPlace());
        // recompute bias and weights
        if (!with_conv_bias) {  // prev node is conv
          for (int i = 0; i < mean_len; ++i) {
            bn_scale_ptr[i] = bn_scale_ptr[i] / sqrtf(bn_var_ptr[i] + epsilon);
            fusion_bias_ptr[i] += (0.0f - bn_mean_ptr[i]) * bn_scale_ptr[i];
            for (int j = 0; j < filter_stride; j++) {
              filter_ptr[i * filter_stride + j] *= bn_scale_ptr[i];
            }
          }
        } else {
          for (int i = 0; i < mean_len; ++i) {
            bn_scale_ptr[i] = bn_scale_ptr[i] / sqrtf(bn_var_ptr[i] + epsilon);
            fusion_bias_ptr[i] =
                bn_bias_ptr[i] +
                (fusion_bias_ptr[i] - bn_mean_ptr[i]) * bn_scale_ptr[i];
            for (int j = 0; j < filter_stride; j++) {
              filter_ptr[i * filter_stride + j] *= bn_scale_ptr[i];
            }
          }
        }
      }
    }
    VLOG(4) << "--- deal with name";
    // filter max
    Node* filter_int16 = nullptr;
    Node* filter_max = nullptr;
    PrepareWeight<int16_t>(
        graph, scope, block, conv_filter, &filter_int16, &filter_max, false);
    // output && output max
    VLOG(4) << "--- output && output max";
    std::string conv1d_xpu_out_name;
    if (act) {
      conv1d_xpu_out_name = act_out->Name();
    } else if (ew_branch_add) {
      conv1d_xpu_out_name = ew_branch_add_out->Name();
    } else if (bn) {
      conv1d_xpu_out_name = bn_out->Name();
    } else if (squeeze2) {
      conv1d_xpu_out_name = squeeze2_out->Name();
    } else if (ew_bias_add) {
      conv1d_xpu_out_name = ew_bias_add_out->Name();
    } else {
      conv1d_xpu_out_name = conv_out->Name();
    }
    std::string conv1d_out_max_name = conv1d_xpu_out_name + "_max";
    VarDesc conv1d_out_max_desc(conv1d_out_max_name);
    Node* conv1d_xpu_out_max = graph->CreateVarNode(&conv1d_out_max_desc);
    // Generate conv1d_xpu op
    framework::OpDesc conv1d_xpu_op_desc(block);
    // set input&output var
    conv1d_xpu_op_desc.SetType("conv1d_xpu");
    conv1d_xpu_op_desc.SetInput("x", {x->Name()});
    conv1d_xpu_op_desc.SetInput("filter", {filter_int16->Name()});
    conv1d_xpu_op_desc.SetInput("filter_max", {filter_max->Name()});
    conv1d_xpu_op_desc.SetOutput("out", {conv1d_xpu_out_name});
    conv1d_xpu_op_desc.SetOutput("out_max", {conv1d_out_max_name});
    // set fusion_bias input node
    if (has_bias) {
      conv1d_xpu_op_desc.SetInput("bias", {fusion_bias_node->Name()});
    }
    // set ew_branch_add input node
    if (ew_branch_add != nullptr) {
      conv1d_xpu_op_desc.SetInput("branch", {ew_branch_add_in->Name()});
    }
    // set attrs of conv1d_xpu
    float act_param_ = 0.0f;
    if (!act_type.empty()) {
      if (act_type == "leaky_relu") {
        act_param_ = PADDLE_GET_CONST(float, act->Op()->GetAttr("alpha"));
      } else if (act_type == "hard_sigmoid") {
        act_param_ = PADDLE_GET_CONST(float, act->Op()->GetAttr("slope"));
      }
    }
    conv1d_xpu_op_desc.SetAttr("act_type", ConvertActivationType(act_type));
    conv1d_xpu_op_desc.SetAttr("act_param", act_param_);
    conv1d_xpu_op_desc.SetAttr(
        "groups", PADDLE_GET_CONST(int, conv->Op()->GetAttr("groups")));
    conv1d_xpu_op_desc.SetAttr(
        "padding_algorithm",
        conv->Op()->GetAttrIfExists<std::string>("padding_algorithm"));
    auto conv_paddings =
        conv->Op()->GetAttrIfExists<std::vector<int>>("paddings");
    if (conv_paddings.size() == 2) {
      if (conv_paddings[0] == 0) {
        conv_paddings[0] = conv_paddings[1];
      }
    }
    std::vector<int> paddings = {conv_paddings[0], conv_paddings[1]};
    conv1d_xpu_op_desc.SetAttr("paddings", paddings);
    auto conv_dilations =
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("dilations"));
    int dilations_w = 1;
    if (conv_dilations.size() == 2) {
      dilations_w = conv_dilations[1];
    }
    conv1d_xpu_op_desc.SetAttr("dilations", dilations_w);
    auto conv_strides =
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("strides"));
    int stride_w = 1;
    if (conv_strides.size() == 2) {
      stride_w = conv_strides[1];
    }
    conv1d_xpu_op_desc.SetAttr("strides", stride_w);
    // update graph pattern after fuse
    std::unordered_set<const Node*> delete_nodes = {
        conv, conv_out, ew_bias_add, ew_bias_add_y, ew_bias_add_out, squeeze2};
    // for x->unsqueeze-->conv2d pattern
    //                 |->conv2d
    if (conv_input->outputs.size() == 1) {
      IR_NODE_UNLINK(x, unsqueeze2);
      auto x_link_in_nodes = x->inputs;
      for (auto x_link_in_node : x_link_in_nodes) {
        auto op_desc = x_link_in_node->Op();
        op_desc->Flush();
      }
      delete_nodes.insert(unsqueeze2);
      delete_nodes.insert(conv_input);
    } else {
      IR_NODE_UNLINK(conv_input, conv);
      unsqueeze2->Op()->Flush();
    }
    auto* conv1d_xpu = graph->CreateOpNode(&conv1d_xpu_op_desc);
    IR_NODE_LINK_TO(x, conv1d_xpu);
    IR_NODE_LINK_TO(filter_int16, conv1d_xpu);
    IR_NODE_LINK_TO(filter_max, conv1d_xpu);
    if (has_bias) {
      SAFE_IR_NODE_LINK_TO(fusion_bias_node, conv1d_xpu);
    }
    if (ew_branch_add_in) {
      IR_NODE_LINK_TO(ew_branch_add_in, conv1d_xpu);
    }
    if (act_out) {
      IR_NODE_LINK_TO(conv1d_xpu, act_out);
    } else if (ew_branch_add_out) {
      IR_NODE_LINK_TO(conv1d_xpu, ew_branch_add_out);
    } else if (bn_out) {
      IR_NODE_LINK_TO(conv1d_xpu, bn_out);
    } else {
      IR_NODE_LINK_TO(conv1d_xpu, squeeze2_out);
    }
    IR_NODE_LINK_TO(conv1d_xpu, conv1d_xpu_out_max);
    // delete useless node
    if (act != nullptr) {
      delete_nodes.insert(act);
    }
    if (ew_branch_add != nullptr) {
      delete_nodes.insert(ew_branch_add);
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

REGISTER_PASS(conv1d_xpu_fuse_pass, paddle::framework::ir::Conv1dXPUFusePass);

REGISTER_PASS_CAPABILITY(conv1d_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "conv1d_xpu", 0));
