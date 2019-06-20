// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/mir/fusion/conv_bn_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvBNFuser::BuildPattern() {
  auto* conv_input =
      VarNode("conv_input")->assert_is_op_input(conv_type_, "Input")->AsInput();
  auto* conv_weight = VarNode("conv_weight")
                          ->assert_is_op_input(conv_type_, "Filter")
                          ->AsInput();
  auto* conv = OpNode("conv2d", conv_type_)->assert_is_op(conv_type_);
  auto* conv_out = VarNode("conv_out")
                       ->assert_is_op_output(conv_type_, "Output")
                       ->assert_is_op_input("batch_norm", "X");

  auto* bn_scale = VarNode("bn_scale")
                       ->assert_is_op_input("batch_norm", "Scale")
                       ->AsIntermediate();
  auto* bn_bias =
      VarNode("bn_bias")->assert_is_op_input("batch_norm", "Bias")->AsInput();
  auto* bn_mean = VarNode("bn_mean")
                      ->assert_is_op_input("batch_norm", "Mean")
                      ->AsIntermediate();
  auto* bn_var = VarNode("bn_variance")
                     ->assert_is_op_input("batch_norm", "Variance")
                     ->AsIntermediate();
  auto* bn =
      OpNode("bn", "batch_norm")->assert_is_op("batch_norm")->AsIntermediate();

  auto* bn_out =
      VarNode("bn_out")->assert_is_op_output("batch_norm", "Y")->AsOutput();
  auto* bn_mean_out = VarNode("bn_mean_out")
                          ->assert_is_op_output("batch_norm", "MeanOut")
                          ->AsIntermediate();
  auto* bn_var_out = VarNode("bn_var_out")
                         ->assert_is_op_output("batch_norm", "VarianceOut")
                         ->AsIntermediate();
  auto* bn_saved_mean = VarNode("bn_saved_mean")
                            ->assert_is_op_output("batch_norm", "SavedMean")
                            ->AsIntermediate();
  auto* bn_saved_var = VarNode("bn_saved_var")
                           ->assert_is_op_output("batch_norm", "SavedVariance")
                           ->AsIntermediate();

  conv->LinksFrom({conv_input, conv_weight}).LinksTo({conv_out});

  bn->LinksFrom({conv_out, bn_scale, bn_bias, bn_mean, bn_var})
      .LinksTo({bn_out, bn_mean_out, bn_saved_mean, bn_saved_var, bn_var_out});
}

void ConvBNFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto eltwise_op = LiteOpRegistry::Global().Create("elementwise_add");
  auto conv = matched.at("conv2d")->stmt()->op();
  auto* scope = conv->scope();
  auto& valid_places = conv->valid_places();

  auto conv_weight_t = scope->FindVar(matched.at("conv_weight")->arg()->name)
                           ->GetMutable<lite::Tensor>();
  auto conv_weight_d = conv_weight_t->mutable_data<float>();
  auto conv_weight_dims = conv_weight_t->dims();
  size_t weight_num = conv_weight_t->data_size();

  auto bn_scale_t = scope->FindVar(matched.at("bn_scale")->arg()->name)
                        ->GetMutable<lite::Tensor>();
  size_t bias_size = bn_scale_t->data_size();
  auto bn_scale_d = bn_scale_t->mutable_data<float>();
  CHECK_EQ(bias_size, static_cast<size_t>(conv_weight_dims[0]))
      << "The BN bias's size should be equal to the size of the first "
      << "dim size of the conv weights";

  auto bn_mean_t = scope->FindVar(matched.at("bn_mean")->arg()->name)
                       ->GetMutable<lite::Tensor>();
  auto bn_mean_d = bn_mean_t->mutable_data<float>();

  auto bn_var_t = scope->FindVar(matched.at("bn_variance")->arg()->name)
                      ->GetMutable<lite::Tensor>();
  auto bn_var_d = bn_var_t->mutable_data<float>();

  auto bn_bias_t = scope->FindVar(matched.at("bn_bias")->arg()->name)
                       ->GetMutable<lite::Tensor>();
  auto bn_bias_d = bn_bias_t->mutable_data<float>();
  auto eps = matched.at("bn")->stmt()->op_info()->GetAttr<float>("epsilon");

  ComputeFusedWeight(bn_scale_d, bn_mean_d, bn_var_d, bn_bias_d, conv_weight_d,
                     eps, bias_size, weight_num / bias_size);

  eltwise_op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(eltwise_op, valid_places);

  IR_NODE_LINK_TO(matched.at("conv_out"), new_op_node);
  IR_NODE_LINK_TO(matched.at("bn_bias"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("bn_out"));
}

cpp::OpDesc ConvBNFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc;
  op_desc.SetType("elementwise_add");
  op_desc.SetInput("X", {matched.at("conv_out")->arg()->name});
  op_desc.SetInput("Y", {matched.at("bn_bias")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("bn_out")->arg()->name});
  op_desc.SetAttr("axis", 1);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
