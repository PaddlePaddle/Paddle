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

struct Conv2dPoolingXPUPattern : public PatternBase {
  Conv2dPoolingXPUPattern(PDPattern* pattern,
                          const std::string& name_scope,
                          const std::string& conv_type,
                          const std::string& act_type,
                          bool with_conv_bias,
                          bool with_bn);
  // declare operator node's name
  PATTERN_DECL_NODE(conv);
  PATTERN_DECL_NODE(ew_bias_add);
  PATTERN_DECL_NODE(bn);
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(pool2d);
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
  PATTERN_DECL_NODE(act_out);
  PATTERN_DECL_NODE(pool2d_out);

 private:
  std::string conv_type_;
  std::string act_type_;
  bool with_conv_bias_{false};
  bool with_bn_{false};
};

Conv2dPoolingXPUPattern::Conv2dPoolingXPUPattern(PDPattern* pattern,
                                                 const std::string& name_scope,
                                                 const std::string& conv_type,
                                                 const std::string& act_type,
                                                 bool with_conv_bias,
                                                 bool with_bn)
    : PatternBase(pattern, name_scope, name_scope),
      conv_type_(conv_type),
      act_type_(act_type),
      with_conv_bias_(with_conv_bias),
      with_bn_(with_bn) {
  auto conv = pattern->NewNode(conv_repr())->assert_is_op(conv_type_);
  auto input = pattern->NewNode(input_repr())
                   ->assert_is_op_input(conv_type_, "Input")
                   ->AsInput()
                   ->assert_more([](Node* node) {
                     return node->Var()->GetShape().size() == 4;
                   });
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input(conv_type_, "Filter")
                         ->AsInput();
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output(conv_type_, "Output")
                      ->assert_has_n_outputs(1);
  conv->LinksFrom({input, conv_filter}).LinksTo({conv_out});
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
    if (with_bn_ || !act_type_.empty()) {
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
  PDNode* act = nullptr;
  PDNode* act_out = nullptr;
  PDNode* pool2d = nullptr;
  PDNode* pool2d_out = nullptr;
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
  // act op
  if (!act_type_.empty()) {
    bn_out->assert_is_op_input(act_type_, "X");
    act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
    act_out = pattern->NewNode(act_out_repr())
                  ->assert_is_op_output(act_type_, "Out")
                  ->assert_has_n_outputs(1);
    act->LinksFrom({bn_out}).LinksTo({act_out});
  } else {
    act_out = bn_out;
  }
  // pool2d op
  act_out->assert_is_op_input("pool2d", "X");
  pool2d = pattern->NewNode(pool2d_repr())
               ->assert_is_op("pool2d")
               ->assert_more([](Node* node) {
                 auto* op_desc = node->Op();
                 auto pooling_type =
                     op_desc->GetAttrIfExists<std::string>("pooling_type");
                 auto is_global =
                     op_desc->GetAttrIfExists<bool>("global_pooling");
                 return (pooling_type == "max" && !is_global) ||
                        (pooling_type == "avg" && !is_global);
               });
  pool2d_out = pattern->NewNode(pool2d_out_repr())
                   ->assert_is_op_output("pool2d", "Out")
                   ->AsOutput();
  pool2d->LinksFrom({act_out}).LinksTo({pool2d_out});
}

}  // namespace patterns

/*
fuse conv2d block in resnet50-like model to xpu_conv2d op
For example:
graph[1]: sub block
                    in_Input
                      |
                    conv2d----in_Filter
                      |
                 elementwise_add -----conv_Bias
                      |
                  batch_norm ------bn_Bias
                      |
                     act
                      |
                    pool2d
                      |
                     out
------------------------------------------------------
After the pass is applied:
                    in_Input
       in_Filter      |     in_FilterMax
                 \    |    /
                  \   |   /
              conv2d_pooling_xpu ------ in_Bias
                       |    \
                       |     \
                      out   outmax
*/
class Conv2dPoolingXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                const std::string& conv_type,
                const std::string& act_type,
                bool with_conv_bias,
                bool with_bn) const;

  const std::string name_scope_{"conv2d_pooling_xpu_fuse_pass"};
};

void Conv2dPoolingXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto conv_type : {"conv2d", "depthwise_conv2d"}) {
    for (auto with_conv_bias : {true, false}) {
      for (auto with_bn : {true, false}) {
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
          found_subgraph_count +=
              ApplyImpl(graph, conv_type, act_type, with_conv_bias, with_bn);
        }
      }
    }
  }
  AddStatis(found_subgraph_count);
}

int Conv2dPoolingXPUFusePass::ApplyImpl(ir::Graph* graph,
                                        const std::string& conv_type,
                                        const std::string& act_type,
                                        bool with_conv_bias,
                                        bool with_bn) const {
  GraphPatternDetector gpd;
  patterns::Conv2dPoolingXPUPattern pattern(gpd.mutable_pattern(),
                                            name_scope_,
                                            conv_type,
                                            act_type,
                                            with_conv_bias,
                                            with_bn);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle Conv2dPoolingXPUFusePass fuse";
    /* declare operator node's name */
    GET_IR_NODE(conv);
    GET_IR_NODE(ew_bias_add);
    GET_IR_NODE(bn);
    GET_IR_NODE(act);
    GET_IR_NODE(pool2d);
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
    GET_IR_NODE(pool2d_out);
    auto* block = conv->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

    // recompute bias and weight for conv2d_pooling_xpu op
    auto* filter_t =
        scope->FindVar(conv_filter->Name())->GetMutable<phi::DenseTensor>();
    // conv_filter fp16 --> fp32
    auto filter_dtype = filter_t->dtype();
    int out_dtype = proto::VarType::Type::VarType_Type_FP32;
    if (filter_dtype == phi::DataType::FLOAT16) {
      out_dtype = proto::VarType::Type::VarType_Type_FP16;
      CastToFp32(filter_t, nullptr);
    }

    auto filter_dims = filter_t->dims();
    bool has_bias = with_bn || with_conv_bias;
    // Create conv_fusion_bias (conv bias) variable
    Node* fusion_bias_node = nullptr;
    if (has_bias) {
      if (with_conv_bias) {
        auto* ew_bias_add_y_t = scope->FindVar(ew_bias_add_y->Name())
                                    ->GetMutable<phi::DenseTensor>();
        auto ew_bias_add_y_dims = ew_bias_add_y_t->dims();
        PADDLE_ENFORCE_EQ(filter_dims[0],
                          ew_bias_add_y_dims[0],
                          platform::errors::InvalidArgument(
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
                          platform::errors::InvalidArgument(
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
        if (!with_conv_bias) {  // prev node is conv
          PrepareBias(graph, scope, block, bn_bias, &fusion_bias_node);
        }
        auto fusion_bias_t = scope->Var(fusion_bias_node->Name())
                                 ->GetMutable<phi::DenseTensor>();
        float* fusion_bias_ptr =
            fusion_bias_t->mutable_data<float>(paddle::platform::CPUPlace());
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
    // filter max
    Node* filter_int16 = nullptr;
    Node* filter_max = nullptr;
    PrepareWeight<int16_t>(
        graph, scope, block, conv_filter, &filter_int16, &filter_max, false);
    // output && output max
    std::string conv2d_pooling_xpu_out_name = pool2d_out->Name();
    std::string conv2d_pooling_xpu_out_max_name =
        conv2d_pooling_xpu_out_name + "_max";
    VarDesc conv2d_pooling_xpu_out_max_desc(conv2d_pooling_xpu_out_max_name);
    Node* conv2d_pooling_xpu_out_max =
        graph->CreateVarNode(&conv2d_pooling_xpu_out_max_desc);
    // Generate conv2d_pooling_xpu op
    framework::OpDesc conv2d_pooling_xpu_op_desc(block);
    // set input&output var
    conv2d_pooling_xpu_op_desc.SetType("conv2d_pooling_xpu");
    conv2d_pooling_xpu_op_desc.SetInput("x", {input->Name()});
    conv2d_pooling_xpu_op_desc.SetInput("filter", {filter_int16->Name()});
    conv2d_pooling_xpu_op_desc.SetInput("filter_max", {filter_max->Name()});
    conv2d_pooling_xpu_op_desc.SetOutput("out", {conv2d_pooling_xpu_out_name});
    conv2d_pooling_xpu_op_desc.SetOutput("out_max",
                                         {conv2d_pooling_xpu_out_max_name});
    // set fusion_bias input node
    if (has_bias) {
      conv2d_pooling_xpu_op_desc.SetInput("bias", {fusion_bias_node->Name()});
    }
    // set attrs of conv2d_pooling_xpu
    float act_param_ = 0.0f;
    if (!act_type.empty()) {
      if (act_type == "leaky_relu") {
        act_param_ = PADDLE_GET_CONST(float, act->Op()->GetAttr("alpha"));
      } else if (act_type == "hard_sigmoid") {
        act_param_ = PADDLE_GET_CONST(float, act->Op()->GetAttr("slope"));
      }
    }
    conv2d_pooling_xpu_op_desc.SetAttr("act_type",
                                       ConvertActivationType(act_type));
    conv2d_pooling_xpu_op_desc.SetAttr("act_param", act_param_);
    conv2d_pooling_xpu_op_desc.SetAttr(
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
                      platform::errors::InvalidArgument(
                          "padding length should be 4, but received %d, ",
                          conv_paddings.size()));
    conv2d_pooling_xpu_op_desc.SetAttr("paddings", conv_paddings);
    conv2d_pooling_xpu_op_desc.SetAttr(
        "groups", PADDLE_GET_CONST(int, conv->Op()->GetAttr("groups")));
    conv2d_pooling_xpu_op_desc.SetAttr(
        "dilations",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("dilations")));
    conv2d_pooling_xpu_op_desc.SetAttr(
        "strides",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("strides")));
    conv2d_pooling_xpu_op_desc.SetAttr("out_dtype", out_dtype);
    // set pooling attr of conv2d_pooling_xpu
    auto pool2d_paddings =
        PADDLE_GET_CONST(std::vector<int>, pool2d->Op()->GetAttr("paddings"));
    if (pool2d_paddings.size() == 2) {
      for (int i = 0; i < 2; i++) {
        int copy_pad = *(pool2d_paddings.begin() + 2 * i);
        pool2d_paddings.insert(pool2d_paddings.begin() + 2 * i + 1, copy_pad);
      }
    }
    conv2d_pooling_xpu_op_desc.SetAttr("pool2d_paddings", pool2d_paddings);
    auto pool2d_strides =
        PADDLE_GET_CONST(std::vector<int>, pool2d->Op()->GetAttr("strides"));
    auto pool2d_ksize =
        PADDLE_GET_CONST(std::vector<int>, pool2d->Op()->GetAttr("ksize"));
    conv2d_pooling_xpu_op_desc.SetAttr("pool2d_strides", pool2d_strides);
    conv2d_pooling_xpu_op_desc.SetAttr("pool2d_ksize", pool2d_ksize);
    bool is_avg = true;
    auto pool_type = PADDLE_GET_CONST(std::string, pool2d->Op()->GetAttr("pooling_type"));
    if (pool_type == "max") {
      is_avg = false;
    }
    conv2d_pooling_xpu_op_desc.SetAttr("is_avg", is_avg);
    // create conv2d_pooling_xpu op
    auto* conv2d_pooling_xpu = graph->CreateOpNode(&conv2d_pooling_xpu_op_desc);
    IR_NODE_LINK_TO(input, conv2d_pooling_xpu);
    IR_NODE_LINK_TO(filter_int16, conv2d_pooling_xpu);
    IR_NODE_LINK_TO(filter_max, conv2d_pooling_xpu);
    if (ew_bias_add || bn) {
      SAFE_IR_NODE_LINK_TO(fusion_bias_node, conv2d_pooling_xpu);
    }
    IR_NODE_LINK_TO(conv2d_pooling_xpu, pool2d_out);
    IR_NODE_LINK_TO(conv2d_pooling_xpu, conv2d_pooling_xpu_out_max);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {conv, conv_out, pool2d};
    if (act != nullptr) {
      delete_nodes.insert(act);
      delete_nodes.insert(act_out);
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
      delete_nodes.insert(bn_out);
    }
    if (ew_bias_add != nullptr) {
      delete_nodes.insert(ew_bias_add);
      delete_nodes.insert(ew_bias_add_y);
      delete_nodes.insert(ew_bias_add_out);
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

REGISTER_PASS(conv2d_pooling_xpu_fuse_pass,
              paddle::framework::ir::Conv2dPoolingXPUFusePass);

REGISTER_PASS_CAPABILITY(conv2d_pooling_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "conv2d_pooling_xpu", 0));
