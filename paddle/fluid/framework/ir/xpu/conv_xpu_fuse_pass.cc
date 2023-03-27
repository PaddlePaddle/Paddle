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

struct ConvXPUPattern : public PatternBase {
  ConvXPUPattern(PDPattern* pattern,
                 const std::string& name_scope,
                 const std::string& conv_type,
                 const std::string& act_type,
                 bool with_conv_bias,
                 bool with_bn,
                 bool with_branch_x,
                 bool with_branch_y);
  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(ew_bias_add);
  PATTERN_DECL_NODE(bn);
  PATTERN_DECL_NODE(ew_branch_add);
  PATTERN_DECL_NODE(act);
  // declare variable node's name
  PATTERN_DECL_NODE(input);
  PATTERN_DECL_NODE(conv_filter);
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

ConvXPUPattern::ConvXPUPattern(PDPattern* pattern,
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
  LOG(INFO) << "---------conv_type is: " << conv_type;
  auto conv = pattern->NewNode(conv_repr())->assert_is_op(conv_type_);
  auto input = pattern->NewNode(input_repr())
                   ->assert_is_op_input(conv_type_, "Input")
                   ->AsInput();
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input(conv_type_, "Filter")
                         ->AsInput();
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output(conv_type_, "Output")
                      ->assert_var_not_persistable();
  conv->LinksFrom({input, conv_filter}).LinksTo({conv_out});
  // ew_bias_add op
  PDNode* ew_bias_add = nullptr;
  PDNode* ew_bias_add_y = nullptr;
  PDNode* ew_bias_add_out = nullptr;
  if (with_conv_bias_) {
    LOG(INFO) << "--------------pattern---------------with_conv_bias";
    conv_out->assert_is_op_input("elementwise_add", "X");
    ew_bias_add_y = pattern->NewNode(ew_bias_add_y_repr())
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->assert_is_persistable_var()
                        ->assert_has_n_outputs(1);
    ew_bias_add =
        pattern->NewNode(ew_bias_add_repr())->assert_is_op("elementwise_add");
    ew_bias_add_out = pattern->NewNode(ew_bias_add_out_repr())
                          ->assert_is_op_output("elementwise_add", "Out")
                          ->assert_var_not_persistable();
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
  PDNode* act = nullptr;
  PDNode* act_out = nullptr;
  // batch_norm op
  if (with_bn_) {
    LOG(INFO) << "-------------pattern-------------with_bn";
    ew_bias_add_out->assert_is_op_input("batch_norm", "X");
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
    bn_out = pattern->NewNode(bn_out_repr())
                 ->assert_is_op_output("batch_norm", "Y")
                 ->assert_var_not_persistable();
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
  // ew_branch_add op
  if (with_branch_) {
    if (with_branch_x_) {
      LOG(INFO) << "-------------pattern-------------with_branch_x";
      bn_out->assert_is_op_input("elementwise_add", "Y");
      ew_branch_add_in = pattern->NewNode(ew_branch_add_in_repr())
                             ->assert_is_op_input("elementwise_add", "X")
                             ->assert_is_persistable_var()
                             ->AsInput()
                             ->assert_more([](Node* node) {
                               return node->Var()->GetShape().size() == 4;
                             });
    } else if (with_branch_y_) {
      LOG(INFO) << "-------------pattern-------------with_branch_y";
      bn_out->assert_is_op_input("elementwise_add", "X");
      ew_branch_add_in = pattern->NewNode(ew_branch_add_in_repr())
                             ->assert_is_op_input("elementwise_add", "Y")
                             ->assert_is_persistable_var()
                             ->AsInput()
                             ->assert_more([](Node* node) {
                               return node->Var()->GetShape().size() == 4;
                             });
    }
    ew_branch_add =
        pattern->NewNode(ew_branch_add_repr())->assert_is_op("elementwise_add");
    ew_branch_add_out = pattern->NewNode(ew_branch_add_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_var_not_persistable();
    ew_branch_add->LinksFrom({bn_out, ew_branch_add_in})
        .LinksTo({ew_branch_add_out});
  } else {
    ew_branch_add_out = bn_out;
  }
  // act op
  if (!act_type_.empty()) {
    LOG(INFO) << "-------------pattern-------------with_act:" << act_type_;
    ew_branch_add_out->assert_is_op_input(act_type_, "X");
    act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
    act_out = pattern->NewNode(act_out_repr())
                  ->assert_is_op_output(act_type_, "Out")
                  ->assert_var_not_persistable();
    act->LinksFrom({ew_branch_add_out}).LinksTo({act_out});
  }
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
class ConvXPUFusePass : public FusePassBase {
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

  const std::string name_scope_{"conv_xpu_fuse_pass"};
};

void ConvXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto conv_type : {"conv2d", "depthwise_conv2d"}) {
    for (auto with_conv_bias : {true, false}) {
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
  LOG(INFO) << "------------Detected subgraph num is: " << found_subgraph_count;
}

int ConvXPUFusePass::ApplyImpl(ir::Graph* graph,
                               const std::string& conv_type,
                               const std::string& act_type,
                               bool with_conv_bias,
                               bool with_bn,
                               bool with_branch_x,
                               bool with_branch_y) const {
  GraphPatternDetector gpd;
  LOG(INFO) << "------------start Pattern Detecting!";

  patterns::ConvXPUPattern pattern(gpd.mutable_pattern(),
                                   name_scope_,
                                   conv_type,
                                   act_type,
                                   with_conv_bias,
                                   with_bn,
                                   with_branch_x,
                                   with_branch_y);
  LOG(INFO) << "---------------pattern dot is:"
            << gpd.mutable_pattern()->DotString();
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ConvXPUFusePass fuse";
    LOG(INFO) << "-------act:type: " << act_type
              << " with_conv_bias: " << with_conv_bias
              << " with_bn: " << with_bn << " with_branch_x: " << with_branch_x
              << " with_branch_y: " << with_branch_y;
    /* declare operator node's name */
    GET_IR_NODE(conv);
    GET_IR_NODE(ew_bias_add);
    GET_IR_NODE(bn);
    GET_IR_NODE(ew_branch_add);
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
    GET_IR_NODE(ew_branch_add_in);
    GET_IR_NODE(ew_branch_add_out);
    GET_IR_NODE(act_out);
    auto* block = conv->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

    // recompute bias and weight for conv_xpu op
    auto* filter_t =
        scope->FindVar(conv_filter->Name())->GetMutable<phi::DenseTensor>();
    auto filter_dims = filter_t->dims();
    bool has_bias = with_bn || with_conv_bias;
    // Create conv_fusion_bias (conv bias) variable
    Node* fusion_bias_node = nullptr;
    if (has_bias) {
      LOG(INFO) << "------------deal with fusion bias var";
      if (ew_bias_add != nullptr) {
        auto* ew_bias_add_y_t = scope->FindVar(ew_bias_add_y->Name())
                                    ->GetMutable<phi::DenseTensor>();
        auto ew_bias_add_y_dims = ew_bias_add_y_t->dims();
        PADDLE_ENFORCE_EQ(filter_dims[0],
                          ew_bias_add_y_dims[0],
                          platform::errors::InvalidArgument(
                              "dims of batch "
                              "must equal out_channel of conv"));
        PrepareBias(graph, scope, block, ew_bias_add_y, &fusion_bias_node);
      }
      if (bn != nullptr) {
        auto bn_bias_t =
            scope->Var(bn_bias->Name())->GetMutable<phi::DenseTensor>();
        PADDLE_ENFORCE_EQ(filter_dims[0],
                          bn_bias_t->dims()[0],
                          platform::errors::InvalidArgument(
                              "dims of batch bias"
                              "must equal out_channel of conv"));
        auto bn_scale_t =
            scope->Var(bn_scale->Name())->GetMutable<phi::DenseTensor>();
        auto bn_mean_t =
            scope->Var(bn_mean->Name())->GetMutable<phi::DenseTensor>();
        auto bn_var_t =
            scope->Var(bn_var->Name())->GetMutable<phi::DenseTensor>();
        float* filter_ptr =
            filter_t->mutable_data<float>(paddle::platform::CPUPlace());
        float* bn_scale_ptr =
            bn_scale_t->mutable_data<float>(paddle::platform::CPUPlace());
        float* bn_bias_ptr =
            bn_bias_t->mutable_data<float>(paddle::platform::CPUPlace());
        float* bn_mean_ptr =
            bn_mean_t->mutable_data<float>(paddle::platform::CPUPlace());
        float* bn_var_ptr =
            bn_var_t->mutable_data<float>(paddle::platform::CPUPlace());
        auto mean_len = bn_mean_t->numel();
        auto filter_len = filter_t->numel();
        auto filter_stride = filter_len / mean_len;
        float epsilon = PADDLE_GET_CONST(float, bn->Op()->GetAttr("epsilon"));
        if (fusion_bias_node == nullptr) {  // prev node is conv
          LOG(INFO) << "------------no ew_bias_add, create new fusion_bias var";
          PrepareBias(graph, scope, block, bn_bias, &fusion_bias_node);
        } else {  // prev node is ew_bias_add
          LOG(INFO)
              << "------------deal with fusion bias var after ew_bias_add";
        }
        auto fusion_bias_t = scope->Var(fusion_bias_node->Name())
                                 ->GetMutable<phi::DenseTensor>();
        float* fusion_bias_ptr =
            fusion_bias_t->mutable_data<float>(paddle::platform::CPUPlace());
        // recompute bias and weights
        if (ew_bias_add == nullptr) {
          for (int i = 0; i < mean_len; ++i) {
            bn_scale_ptr[i] = bn_scale_ptr[i] / sqrtf(bn_var_ptr[i] + epsilon);
            fusion_bias_ptr[i] += (0.f - bn_mean_ptr[i]) * bn_scale_ptr[i];
            for (int j = 0; j < filter_stride; j++) {
              filter_ptr[i * filter_stride + j] *= bn_scale_ptr[i];
            }
          }
        } else {
          for (int i = 0; i < mean_len; ++i) {
            bn_scale_ptr[i] = bn_scale_ptr[i] / sqrtf(bn_var_ptr[i] + epsilon);
            bn_bias_ptr[i] +=
                (fusion_bias_ptr[i] - bn_mean_ptr[i]) * bn_scale_ptr[i];
            for (int j = 0; j < filter_stride; j++) {
              filter_ptr[i * filter_stride + j] *= bn_scale_ptr[i];
            }
          }
          memcpy(fusion_bias_ptr, bn_bias_ptr, mean_len * sizeof(float));
        }
      }
    }
    // filter max
    Node* filter_int16 = nullptr;
    Node* filter_max = nullptr;
    PrepareWeight<int16_t>(
        graph, scope, block, conv_filter, &filter_int16, &filter_max, false);
    // output && output max
    std::string conv_xpu_out_name;
    if (!act_type.empty()) {
      conv_xpu_out_name = act_out->Name();
    } else if (ew_branch_add) {
      conv_xpu_out_name = ew_branch_add_out->Name();
    } else if (bn) {
      conv_xpu_out_name = bn_out->Name();
    } else if (ew_bias_add) {
      conv_xpu_out_name = ew_bias_add_out->Name();
    } else {
      conv_xpu_out_name = conv_out->Name();
    }
    LOG(INFO) << "-----------output var name is: " << conv_xpu_out_name;
    std::string conv_out_max_name = conv_xpu_out_name + "_max";
    VarDesc conv_out_max_desc(conv_out_max_name);
    Node* conv_xpu_out_max = graph->CreateVarNode(&conv_out_max_desc);
    // Generate conv_xpu op
    framework::OpDesc conv_xpu_op_desc(block);
    // set input&output var
    conv_xpu_op_desc.SetType("conv_xpu");
    conv_xpu_op_desc.SetInput("Input", {input->Name()});
    conv_xpu_op_desc.SetInput("Filter", {filter_int16->Name()});
    conv_xpu_op_desc.SetInput("FilterMax", {filter_max->Name()});
    conv_xpu_op_desc.SetOutput("Output", {conv_xpu_out_name});
    conv_xpu_op_desc.SetOutput("OutputMax", {conv_out_max_name});
    // set fusion_bias input node
    if (has_bias) {
      LOG(INFO) << "----link fusion_bias node, it's name is: "
                << fusion_bias_node->Name();
      conv_xpu_op_desc.SetInput("Bias", {fusion_bias_node->Name()});
      conv_xpu_op_desc.SetAttr("has_bias", has_bias);
    }
    // set ew_branch_add input node
    if (ew_branch_add != nullptr) {
      LOG(INFO) << "----deal with branch node,it's name is: "
                << ew_branch_add_in->Name();
      auto* branch_t = scope->FindVar(ew_branch_add_in->Name())
                           ->GetMutable<phi::DenseTensor>();
      auto branch_dims = branch_t->dims();
      PADDLE_ENFORCE_EQ(branch_dims.size(),
                        4UL,
                        platform::errors::InvalidArgument(
                            "Required shape rank of Attribute(%s) == 4, "
                            "but received rank == %s",
                            ew_branch_add_in->Name(),
                            branch_dims.size()));
      conv_xpu_op_desc.SetInput("Branch", {ew_branch_add_in->Name()});
    }
    // set attrs of conv_xpu
    float act_param_ = 0.0f;
    if (!act_type.empty()) {
      LOG(INFO) << "----deal with act node";
      if (act_type == "leaky_relu") {
        act_param_ = PADDLE_GET_CONST(float, act->Op()->GetAttr("alpha"));
      } else if (act_type == "hard_sigmoid") {
        act_param_ = PADDLE_GET_CONST(float, act->Op()->GetAttr("slope"));
      }
    }
    conv_xpu_op_desc.SetAttr("act_type", ConvertActivationType(act_type));
    conv_xpu_op_desc.SetAttr("act_param", act_param_);
    std::vector<int> conv_bias;
    if (has_bias) {
      conv_bias.push_back(1);
    } else {
      conv_bias.push_back(0);
    }
    if (conv->Op()->HasAttr("padding_algorithm")) {
      conv_xpu_op_desc.SetAttr(
          "padding_algorithm",
          PADDLE_GET_CONST(std::string,
                           conv->Op()->GetAttr("padding_algorithm")));
    }
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
                      platform::errors::InvalidArgument(
                          "padding length should be 4, but received %d, ",
                          conv_paddings.size()));
    conv_xpu_op_desc.SetAttr(
        "dilations",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("dilations")));
    conv_xpu_op_desc.SetAttr(
        "groups", PADDLE_GET_CONST(int, conv->Op()->GetAttr("groups")));
    conv_xpu_op_desc.SetAttr(
        "strides",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("strides")));
    conv_xpu_op_desc.SetAttr("conv_bias", conv_bias);
    conv_xpu_op_desc.SetAttr("op_type", std::vector<int>{0});
    conv_xpu_op_desc.SetAttr("place_x", std::vector<int>{0});
    conv_xpu_op_desc.SetAttr("place_y", std::vector<int>{9});
    conv_xpu_op_desc.SetAttr("place_z", std::vector<int>{10});
    conv_xpu_op_desc.SetAttr("paddings", conv_paddings);
    conv_xpu_op_desc.SetAttr("block_lod", std::vector<int>{1});
    conv_xpu_op_desc.SetAttr("has_branch", with_branch_x || with_branch_y);

    // LOG(INFO) << "----create conv_xpu op!";
    auto* conv_xpu = graph->CreateOpNode(&conv_xpu_op_desc);
    IR_NODE_LINK_TO(input, conv_xpu);
    IR_NODE_LINK_TO(filter_int16, conv_xpu);
    IR_NODE_LINK_TO(filter_max, conv_xpu);
    if (ew_bias_add || bn) {
      SAFE_IR_NODE_LINK_TO(fusion_bias_node, conv_xpu);
      // LOG(INFO) << "----fusion_bias node link success!";
    }
    if (ew_branch_add) {
      IR_NODE_LINK_TO(ew_branch_add_in, conv_xpu);
      // LOG(INFO) << "----ew_branch_add node link success!";
    }
    if (act_out) {
      IR_NODE_LINK_TO(conv_xpu, act_out);
    } else if (ew_branch_add_out) {
      IR_NODE_LINK_TO(conv_xpu, ew_branch_add_out);
    } else if (bn_out) {
      IR_NODE_LINK_TO(conv_xpu, bn_out);
    } else if (ew_bias_add_out) {
      IR_NODE_LINK_TO(conv_xpu, ew_bias_add_out);
    } else {
      IR_NODE_LINK_TO(conv_xpu, conv_out);
    }
    IR_NODE_LINK_TO(conv_xpu, conv_xpu_out_max);
    LOG(INFO) << "----conv_out node link success!";
    LOG(INFO) << "----start delete useless node";
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {conv};
    if (act != nullptr) {
      LOG(INFO) << "----delete act node";
      delete_nodes.insert(act);
    }
    if (ew_branch_add != nullptr) {
      LOG(INFO) << "----delete ew_branch_add node";
      delete_nodes.insert(ew_branch_add);
    }
    if (bn != nullptr) {
      LOG(INFO) << "----delete bn node";
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
      LOG(INFO) << "----delete ew_bias_add node";
      delete_nodes.insert(ew_bias_add);
      delete_nodes.insert(ew_bias_add_y);
    }
    GraphSafeRemoveNodes(graph, delete_nodes);
    LOG(INFO) << "----delete nodes complete";
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_xpu_fuse_pass, paddle::framework::ir::ConvXPUFusePass);

REGISTER_PASS_CAPABILITY(conv_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "conv_xpu", 0));
