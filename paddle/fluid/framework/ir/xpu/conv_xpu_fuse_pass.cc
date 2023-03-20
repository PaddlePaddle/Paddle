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
      with_branch_(with_branch_x | with_branch_y),
      with_branch_x_(with_branch_x),
      with_branch_y_(with_branch_y) {
  // conv op
  auto* input =
      pattern->NewNode(input_repr())->assert_is_op_input(conv_type_, "Input");
  ->AsInput() auto* conv_filter = pattern->NewNode(conv_filter_repr())
                                      ->assert_is_op_input(conv_type_, "Filter")
                                      ->AsInput();
  auto* conv =
      pattern->NewNode(conv_repr())->assert_is_op(conv_type_)->AsIntermediate();
  auto* conv_out = pattern->NewNode(conv_out_repr())
                       ->assert_is_op_output(conv_type_, "Output");
  conv->LinksFrom({input, conv_filter}).LinksTo({conv_out});
  // ew_bias_add op
  PDNode* ew_bias_add = nullptr;
  PDNode* ew_bias_add_y = nullptr;
  PDNode* ew_bias_add_out = nullptr;
  if (with_conv_bias_) {
    conv_out->assert_is_op_input("elementwise_add", "X")->AsIntermediate();
    ew_bias_add_y = pattern->NewNode(ew_bias_add_y_repr())
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->assert_is_persistable_var()
                        ->assert_has_n_outputs(1)
                        ->AsIntermediate();
    ew_bias_add = pattern->NewNode(ew_bias_add_repr())
                      ->assert_is_op("elementwise_add")
                      ->AsIntermediate();
    ew_bias_add_out = pattern->NewNode(ew_bias_add_out_repr())
                          ->assert_is_op_output("elementwise_add", "Out");
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
    ew_bias_add_out->->assert_is_op_input("batch_norm", "X")->AsIntermediate();
    bn_bias->NewNode(bn_bias_repr())->assert_is_op_input("batch_norm", "Bias");
    ->assert_has_n_outputs(1)->AsIntermediate();
    bn_mean->NewNode(bn_mean_repr())->assert_is_op_input("batch_norm", "Mean");
    ->assert_has_n_outputs(1)->AsIntermediate();
    bn_scale->NewNode(bn_scale_repr())
        ->assert_is_op_input("batch_norm", "Scale");
    ->assert_has_n_outputs(1)->AsIntermediate();
    bn_var->NewNode(bn_var_repr())
        ->assert_is_op_input("batch_norm", "Variance");
    ->assert_has_n_outputs(1)->AsIntermediate();
    bn = pattern->NewNode(bn_repr())
             ->assert_is_op("batch_norm")
             ->AsIntermediate();
    bn_out =
        pattern->NewNode(bn_out_repr())->assert_is_op_output("batch_norm", "Y");
    bn_mean_out = pattern->NewNode(bn_mean_out_repr())
                      ->assert_is_op_output("batch_norm", "MeanOut")
                      ->AsIntermediate();
    bn_saved_mean = pattern->NewNode(bn_saved_mean_repr())
                        ->assert_is_op_output("batch_norm", "SavedMean")
                        ->AsIntermediate();
    bn_var_out = pattern->NewNode(bn_var_out_repr())
                     ->assert_is_op_output("batch_norm", "VarianceOut")
                     ->AsIntermediate();
    bn_saved_var = pattern->NewNode(bn_saved_var_repr())
                       ->assert_is_op_output("batch_norm", "SavedVariance")
                       ->AsIntermediate();
    bn->LinksFrom({ew_bias_add_out, bn_bias, bn_mean, bn_scale, bn_var})
        .LinksTo(
            {bn_out, bn_mean_out, bn_var_out_, bn_saved_mean, bn_saved_var});
  } else {
    bn_out = ew_bias_add_out;
  }
  // ew_branch_add op
  if (with_branch_) {
    if (with_branch_x_) {
      bn_out->->assert_is_op_input("elementwise_add", "Y")->AsIntermediate();
      ew_branch_add_in = pattern->NewNode(ew_branch_add_in_repr())
                             ->assert_is_op_input("elementwise_add", "X")
                             ->assert_var_not_persistable()
                             ->AsInput();
    } else if (with_branch_y_) {
      bn_out->->assert_is_op_input("elementwise_add", "X")->AsIntermediate();
      ew_branch_add_in = pattern->NewNode(ew_branch_add_in_repr())
                             ->assert_is_op_input("elementwise_add", "Y")
                             ->assert_var_not_persistable()
                             ->AsInput();
    }
    ew_branch_add = pattern->NewNode(ew_branch_add_repr())
                        ->assert_is_op("elementwise_add")
                        ->AsIntermediate();
    ew_branch_add_out = pattern->NewNode(ew_branch_add_in_repr())
                            ->assert_is_op_output("elementwise_add", "Out");
    ew_branch_add->LinksFrom({bn_out, ew_branch_add_in})
        .LinksTo({ew_branch_add_out});
  } else {
    ew_branch_add_out = bn_out;
  }
  // act op
  if (!act_type_.empty()) {
    ew_branch_add_out->->assert_is_op_input(act_type_, "X")->AsIntermediate();
    act = pattern->NewNode(act_repr())->assert_is_op(act_type_);
    act_out = pattern->NewNode(act_out_repr())
                  ->assert_is_op_output(act_type_, "Out")
                  ->AsOutput();
    act->LinksFrom({ew_branch_add_out}).LinksTo({act_out});
  } else {
    act_out = ew_branch_add_out;
    act_out->AsOutput();
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
#define GET_CONV_FUSE_NODES          \
  /* declare operator node's name */ \
  GET_IR_NODE(conv);                 \
  GET_IR_NODE(ew_bias_add);          \
  GET_IR_NODE(bn);                   \
  GET_IR_NODE(ew_branch_add);        \
  GET_IR_NODE(act);                  \
  /* declare variable node's name*/  \
  GET_IR_NODE(input);                \
  GET_IR_NODE(conv_filter);          \
  GET_IR_NODE(conv_out);             \
  GET_IR_NODE(ew_bias_add_y);        \
  GET_IR_NODE(ew_bias_add_out);      \
  GET_IR_NODE(bn_bias);              \
  GET_IR_NODE(bn_mean);              \
  GET_IR_NODE(bn_scale);             \
  GET_IR_NODE(bn_var);               \
  GET_IR_NODE(bn_out);               \
  GET_IR_NODE(bn_var_out);           \
  GET_IR_NODE(bn_mean_out);          \
  GET_IR_NODE(bn_saved_var);         \
  GET_IR_NODE(bn_saved_mean);        \
  GET_IR_NODE(ew_branch_add_in);     \
  GET_IR_NODE(ew_branch_add_out);    \
  GET_IR_NODE(act_out);

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
    for (auto with_branch_x : {true, false}) {
      for (auto with_branch_y : {true, false}) {
        for (auto with_conv_bias : {true, false}) {
          for (auto with_bn : {true, false}) {
            for (auto act_type : {"relu",
                                  "sigmoid",
                                  "tanh",
                                  "gelu",
                                  "leaky_relu",
                                  "hard_swish",
                                  "hard_sigmoid",
                                  "relu6",
                                  "swish",
                                  ""}) {
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

int ConvXPUFusePass::ApplyImpl(ir::Graph* graph,
                               const std::string& conv_type,
                               const std::string& act_type,
                               bool with_conv_bias,
                               bool with_bn,
                               bool with_branch_x,
                               bool with_branch_y) const {
  GraphPatternDetector gpd;
  patterns::ConvXPUPattern pattern(gpd.mutable_pattern(),
                                   name_scope_,
                                   conv_type,
                                   act_type,
                                   with_conv_bias,
                                   with_bn,
                                   with_branch_x,
                                   with_branch_y);
  auto with_branch = with_branch_y || with_branch_x;
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ConvXPUFusePass fuse";
    GET_CONV_FUSE_NODES
    auto* block = conv->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

    // recompute bias and weight for conv_xpu op
    auto* filter_t =
        scope->FindVar(conv_filter->Name())->GetMutable<phi::DenseTensor>();
    auto filter_dims = filter_t->dims();
    std::string fusion_bias_name = conv_filter->Name() + "_conv_fusion_bias";
    VarDesc fusion_bias_desc(
        patterns::PDNodeName(name_scope_, fusion_bias_name));
    fusion_bias_desc.SetShape(phi::vectorize(phi::make_ddim({filter_dims[0]})));
    fusion_bias_desc.SetDataType(
        framework::TransToProtoVarType(phi::DataType::FLOAT32));
    fusion_bias_desc.SetPersistable(true);
    auto* fusion_bias_node = graph->CreateVarNode(&fusion_bias_desc);
    auto* fusion_bias_t =
        scope->Var(fusion_bias_node->Name())->GetMutable<phi::DenseTensor>();
    bool has_bias = with_bn || with_conv_bias;
    if (has_bias) {
      fusion_bias_t->Resize(filter_dims[0]);
      float* fusion_bias_ptr =
          fusion_bias_t->mutable_data<float>(paddle::platform::CPUPlace());
      if (with_conv_bias_) {
        auto ew_bias_add_y_name = ew_bias_add_y->Name();
        auto* ew_bias_add_y_t =
            scope->Var(ew_bias_add_y_name)->GetMutable<phi::DenseTensor>();
        float* ew_bias_add_y_ptr =
            ew_bias_add_y_t->mutable_data<float>(paddle::platform::CPUPlace());
        auto ew_bias_add_y_numel = ew_bias_add_y_t->numel();
        if (ew_bias_add_y_numel != filter_dims[0] && ew_bias_add_y_numel == 1) {
          for (int i = 0; i < filter_dims[0]; ++i) {
            fusion_bias_ptr[i] = ew_bias_add_y_ptr[0];
          }
        } else if (ew_bias_add_y_numel == filter_dims[0]) {
          for (int i = 0; i < filter_dims[0]; ++i) {
            fusion_bias_ptr[i] = ew_bias_add_y_ptr[i];
          }
        } else {
          PADDLE_ENFORCE_EQ(ew_bias_add_y_numel,
                            filter_dims[0],
                            platform::errors::InvalidArgument(
                                "elem size of elemwise_bias and  "
                                "conv_filter_oc should be the same."
                                "but get (%d) and (%d)",
                                ew_bias_add_y_numel,
                                filter_dims[0]));
        }
      } else {
        for (int i = 0; i < filter_dims[0]; ++i) {
          fusion_bias_ptr[i] = 0.f;
        }
      }
      if (with_bn) {
        auto bn_scale_t =
            scope->Var(bn_scale->Name())->GetMutable<phi::DenseTensor>();
        auto bn_bias_t =
            scope->Var(bn_bias->Name())->GetMutable<phi::DenseTensor>();
        auto bn_mean_t =
            scope->Var(bn_mean->Name())->GetMutable<phi::DenseTensor>();
        auto bn_var_t =
            scope->Var(bn_var->Name())->GetMutable<phi::DenseTensor>();
        auto mean_len = bn_mean_t->numel();
        auto filter_len = filter_t->numel();
        auto filter_stride = filter_len / mean_len;
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
        float epsilon = PADDLE_GET_CONST(float, bn->Op()->GetAttr("epsilon"));

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
    // quant weight
    Node* filter_int16 = nullptr;
    Node* filter_max = nullptr;
    PrepareWeight<int16_t>(
        graph, scope, block, conv_filter, &filter_int16, &filter_max, false);

    // Generate fc_xpu op
    framework::OpDesc conv_xpu_op_desc(block);
    // set input&output var
    conv_xpu_op_desc.SetType("conv_xpu");
    conv_xpu_op_desc.SetInput("Input", {input->Name()});
    conv_xpu_op_desc.SetInput("Filter", {filter_int16->Name()});
    conv_xpu_op_desc.SetInput("FilterMax", {filter_max->Name()});
    Node* bias_fp32 = nullptr;
    if (has_bias) {
      conv_xpu_op_desc.SetAttr("has_bias", has_bias);
      conv_xpu_op_desc.SetInput("Bias", {fusion_bias_name});
      // PrepareBias(graph, scope, block, bias, &bias_fp32);
      // conv_xpu_op_desc.SetInput("Bias", {bias_fp32->Name()});
    }
    if (with_branch) {
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
    // act && output 
    std::string xpu_out_name;
    float act_param_ = 0.0f;
    if (!act_type.empty()) {
      xpu_out_name = act_out->Name();
      if (act_type == "leaky_relu") {
        act_param_ = act->Op()->GetAttr("alpha");
      } else if (act_type == "hard_sigmoid") {
        act_param_ = act->Op()->GetAttr("slope");
      }
    } else if (with_branch) {
      xpu_out_name = ew_branch_add_out->Name();
    } else if (with_bn) {
      xpu_out_name = bn_out->Name();
    } else if (with_conv_bias) {
      xpu_out_name = ew_bias_add_out->Name();
    } else {
      xpu_out_name = conv_out->Name();
    }
    std::string xpu_out_max_name = xpu_out_name + "_max";
    VarDesc xpu_out_max_desc(xpu_out_max_name);
    Node* xpu_out_max = graph->CreateVarNode(&xpu_out_max_desc);
    conv_xpu_op_desc.SetOutput("Output", {xpu_out_name});
    conv_xpu_op_desc.SetOutput("OutputMax", {xpu_out_max_name});
    // set attrs
    conv_xpu_op_desc.SetAttr("act_type", ConvertActivationType(act_type));
    conv_xpu_op_desc.SetAttr("act_param", act_param_);
    std::vector<int> conv_groups{
        PADDLE_GET_CONST(int, conv->Op()->GetAttr("groups"))};
    std::vector<int> conv_bias;
    if (with_bn || with_conv_bias) {
      conv_bias.push_back(1);
    } else {
      conv_bias.push_back(0);
    }
    if (conv->Op()->HasAttr("padding_algorithm")) {
      conv_xpu_op_desc.SetAttr(
          "padding_algorithm",
          PADDLE_GET_CONST(std::vector<string>,
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
    conv_xpu_op_desc.SetAttr("groups", conv_groups);
    conv_xpu_op_desc.SetAttr("conv_bias", conv_bias);
    conv_xpu_op_desc.SetAttr("filter_dims", phi::vectorize(filter_dims));
    conv_xpu_op_desc.SetAttr("op_type", std::vector<int>{0});
    conv_xpu_op_desc.SetAttr("place_x", std::vector<int>{0});
    conv_xpu_op_desc.SetAttr("place_y", std::vector<int>{9});
    conv_xpu_op_desc.SetAttr("place_z", std::vector<int>{10});
    conv_xpu_op_desc.SetAttr(
        "strides",
        PADDLE_GET_CONST(std::vector<int>, conv->Op()->GetAttr("strides")));
    conv_xpu_op_desc.SetAttr("paddings", conv_paddings);
    conv_xpu_op_desc.SetAttr("block_lod", std::vector<int>{1});
    conv_xpu_op_desc.SetAttr("has_branch", with_branch);

    auto* conv_xpu = graph->CreateOpNode(&fc_xpu_op_desc);
    IR_NODE_LINK_TO(input, conv_xpu);
    IR_NODE_LINK_TO(filter_int16, conv_xpu);
    IR_NODE_LINK_TO(filter_max, conv_xpu);
    if (with_bn || with_conv_bias) {
      SAFE_IR_NODE_LINK_TO(fusion_bias_node, conv_xpu);
    }
    if (with_branch) {
      IR_NODE_LINK_TO(ew_branch_add_in, conv_xpu);
    }
    IR_NODE_LINK_TO(conv_xpu, act_out);
    IR_NODE_LINK_TO(conv_xpu, xpu_out_max);

    // delete useless node
    GraphSafeRemoveNodes(graph,
                         {ew_bias_add_y,
                          ew_bias_add,
                          bn_scale,
                          bn_bias,
                          bn_mean,
                          bn_var,
                          bn,
                          bn_mean_out,
                          bn_var_out,
                          bn_saved_mean,
                          bn_saved_var,
                          ew_branch_add,
                          act});
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
