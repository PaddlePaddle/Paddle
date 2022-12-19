// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fused_multi_transformer_encoder_pass.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

PDNode* FusedMultiTransformerEncoderPattern::operator()() {
  auto* input0 = pattern->NewNode(input0_repr());
  input0->assert_is_op_input("layer_norm", "X");

  // pre-LayerNorm
  auto* layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto* layer_norm_scale_var = pattern->NewNode(layer_norm_scale_repr())
                                   ->AsInput()
                                   ->assert_is_persistable_var()
                                   ->assert_is_op_input("layer_norm", "Scale");
  auto* layer_norm_bias_var = pattern->NewNode(layer_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("layer_norm", "Bias");
  auto* layer_norm_mean_var = pattern->NewNode(layer_norm_mean_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("layer_norm", "Mean");
  auto* layer_norm_variance_var =
      pattern->NewNode(layer_norm_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("layer_norm", "Variance");
  auto* layer_norm_out_var = pattern->NewNode(layer_norm_out_repr())
                                 ->AsIntermediate()
                                 ->assert_is_op_output("layer_norm", "Y")
                                 ->assert_is_op_input("matmul_v2", "X")
                                 ->assert_more([](Node* x) {
                                   if (x->outputs.size() == 3) {
                                     return true;
                                   } else {
                                     return false;
                                   }
                                 });

  layer_norm->LinksFrom({input0, layer_norm_bias_var, layer_norm_scale_var})
      .LinksTo(
          {layer_norm_out_var, layer_norm_mean_var, layer_norm_variance_var});

  // Q path Nodes
  auto* matmul0 = pattern->NewNode(matmul0_repr())->assert_is_op("matmul_v2");
  auto* matmul0_w_var = pattern->NewNode(matmul0_w_repr())
                            ->AsInput()
                            ->assert_is_op_input("matmul_v2", "Y");
  auto* matmul0_out_var = pattern->NewNode(matmul0_out_repr())
                              ->assert_is_op_output("matmul_v2")
                              ->AsIntermediate()
                              ->assert_is_op_input("elementwise_add");

  auto* eltadd0 =
      pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");
  auto* eltadd0_b_var = pattern->NewNode(eltadd0_b_repr())
                            ->AsInput()
                            ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd0_out_var = pattern->NewNode(eltadd0_out_repr())
                              ->assert_is_op_output("elementwise_add")
                              ->AsIntermediate()
                              ->assert_is_op_input("reshape2");

  auto* reshape2_0 =
      pattern->NewNode(reshape2_0_repr())->assert_is_op("reshape2");
  auto* reshape2_0_out_var = pattern->NewNode(reshape2_0_out_repr())
                                 ->assert_is_op_output("reshape2")
                                 ->AsIntermediate()
                                 ->assert_is_op_input("transpose2");

  auto* transpose2_0 =
      pattern->NewNode(transpose2_0_repr())->assert_is_op("transpose2");
  auto* transpose2_0_out_var = pattern->NewNode(transpose2_0_out_repr())
                                   ->assert_is_op_output("transpose2")
                                   ->AsIntermediate()
                                   ->assert_is_op_input("matmul", "X");

  // Q path Links
  matmul0->LinksFrom({layer_norm_out_var, matmul0_w_var})
      .LinksTo({matmul0_out_var});
  eltadd0->LinksFrom({matmul0_out_var, eltadd0_b_var})
      .LinksTo({eltadd0_out_var});
  reshape2_0->LinksFrom({eltadd0_out_var}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});

  // K path Nodes
  auto* matmul1 = pattern->NewNode(matmul1_repr())->assert_is_op("matmul_v2");
  auto* matmul1_w_var = pattern->NewNode(matmul1_w_repr())
                            ->AsInput()
                            ->assert_is_op_input("matmul_v2", "Y");
  auto* matmul1_out_var = pattern->NewNode(matmul1_out_repr())
                              ->assert_is_op_output("matmul_v2")
                              ->AsIntermediate()
                              ->assert_is_op_input("elementwise_add");

  auto* eltadd1 =
      pattern->NewNode(eltadd1_repr())->assert_is_op("elementwise_add");
  auto* eltadd1_b_var = pattern->NewNode(eltadd1_b_repr())
                            ->AsInput()
                            ->assert_is_op_input("elementwise_add", "Y");

  auto* eltadd1_out_var = pattern->NewNode(eltadd1_out_repr())
                              ->assert_is_op_output("elementwise_add")
                              ->AsIntermediate()
                              ->assert_is_op_input("reshape2");

  auto* reshape2_1 =
      pattern->NewNode(reshape2_1_repr())->assert_is_op("reshape2");
  auto* reshape2_1_out_var = pattern->NewNode(reshape2_1_out_repr())
                                 ->assert_is_op_output("reshape2")
                                 ->AsIntermediate()
                                 ->assert_is_op_input("transpose2");

  auto* transpose2_1 =
      pattern->NewNode(transpose2_1_repr())->assert_is_op("transpose2");
  auto* transpose2_1_out_var = pattern->NewNode(transpose2_1_out_repr())
                                   ->assert_is_op_output("transpose2")
                                   ->AsOutput()
                                   ->assert_is_op_input("matmul", "Y")
                                   ->assert_is_op_input("while")
                                   ->assert_more([](Node* x) {
                                     if (x->outputs.size() == 2) {
                                       return true;
                                     } else {
                                       return false;
                                     }
                                   });

  // K path Links
  matmul1->LinksFrom({layer_norm_out_var, matmul1_w_var})
      .LinksTo({matmul1_out_var});
  eltadd1->LinksFrom({matmul1_out_var, eltadd1_b_var})
      .LinksTo({eltadd1_out_var});
  reshape2_1->LinksFrom({eltadd1_out_var}).LinksTo({reshape2_1_out_var});
  transpose2_1->LinksFrom({reshape2_1_out_var}).LinksTo({transpose2_1_out_var});

  // V path Nodes
  auto* matmul2 = pattern->NewNode(matmul2_repr())->assert_is_op("matmul_v2");
  auto* matmul2_w_var = pattern->NewNode(matmul2_w_repr())
                            ->AsInput()
                            ->assert_is_op_input("matmul_v2", "Y");
  auto* matmul2_out_var = pattern->NewNode(matmul2_out_repr())
                              ->assert_is_op_output("matmul_v2")
                              ->AsIntermediate()
                              ->assert_is_op_input("elementwise_add");

  auto* eltadd2 =
      pattern->NewNode(eltadd2_repr())->assert_is_op("elementwise_add");
  auto* eltadd2_b_var = pattern->NewNode(eltadd2_b_repr())
                            ->AsInput()
                            ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd2_out_var = pattern->NewNode(eltadd2_out_repr())
                              ->assert_is_op_output("elementwise_add")
                              ->AsIntermediate()
                              ->assert_is_op_input("reshape2");

  auto* reshape2_2 =
      pattern->NewNode(reshape2_2_repr())->assert_is_op("reshape2");
  auto* reshape2_2_out_var = pattern->NewNode(reshape2_2_out_repr())
                                 ->assert_is_op_output("reshape2")
                                 ->AsIntermediate()
                                 ->assert_is_op_input("transpose2");

  auto* transpose2_2 =
      pattern->NewNode(transpose2_2_repr())->assert_is_op("transpose2");
  auto* transpose2_2_out_var = pattern->NewNode(transpose2_2_out_repr())
                                   ->assert_is_op_output("transpose2")
                                   ->AsOutput()
                                   ->assert_is_op_input("matmul_v2", "Y")
                                   ->assert_is_op_input("while")
                                   ->assert_more([](Node* x) {
                                     if (x->outputs.size() == 2) {
                                       return true;
                                     } else {
                                       return false;
                                     }
                                   });

  // V path Links
  matmul2->LinksFrom({layer_norm_out_var, matmul2_w_var})
      .LinksTo({matmul2_out_var});
  eltadd2->LinksFrom({matmul2_out_var, eltadd2_b_var})
      .LinksTo({eltadd2_out_var});
  reshape2_2->LinksFrom({eltadd2_out_var}).LinksTo({reshape2_2_out_var});
  transpose2_2->LinksFrom({reshape2_2_out_var}).LinksTo({transpose2_2_out_var});

  // QK path Nodes
  auto* matmul_qk = pattern->NewNode(matmul_qk_repr())->assert_is_op("matmul");
  auto* matmul_qk_out_var =
      pattern->NewNode(matmul_qk_out_repr())->assert_is_op_output("matmul");
  matmul_qk_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  auto* eltadd_qk =
      pattern->NewNode(eltadd_qk_repr())->assert_is_op("elementwise_add");
  auto* eltadd_qk_b_var = pattern->NewNode(eltadd_qk_b_repr())
                              ->AsInput()
                              ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd_qk_out_var = pattern->NewNode(eltadd_qk_out_repr())
                                ->assert_is_op_output("elementwise_add")
                                ->AsIntermediate()
                                ->assert_is_op_input("softmax");

  auto* softmax_qk =
      pattern->NewNode(softmax_qk_repr())->assert_is_op("softmax");
  auto* softmax_qk_out_var = pattern->NewNode(softmax_qk_out_repr())
                                 ->assert_is_op_output("softmax")
                                 ->AsIntermediate()
                                 ->assert_is_op_input("matmul_v2", "X");

  // QK path Linsk
  matmul_qk->LinksFrom({transpose2_0_out_var, transpose2_1_out_var})
      .LinksTo({matmul_qk_out_var});
  eltadd_qk->LinksFrom({matmul_qk_out_var, eltadd_qk_b_var})
      .LinksTo({eltadd_qk_out_var});
  softmax_qk->LinksFrom({eltadd_qk_out_var}).LinksTo({softmax_qk_out_var});

  // QKV path Nodes
  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_op("matmul_v2");
  auto* matmul_qkv_out_var =
      pattern->NewNode(matmul_qkv_out_repr())->assert_is_op_output("matmul_v2");
  matmul_qkv_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_qkv =
      pattern->NewNode(transpose2_qkv_repr())->assert_is_op("transpose2");
  auto* transpose2_qkv_out_var = pattern->NewNode(transpose2_qkv_out_repr())
                                     ->assert_is_op_output("transpose2");
  transpose2_qkv_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_qkv =
      pattern->NewNode(reshape2_qkv_repr())->assert_is_op("reshape2");
  auto* reshape2_qkv_out_var =
      pattern->NewNode(reshape2_qkv_out_repr())
          ->assert_is_op_output("reshape2")
          ->AsIntermediate()
          ->assert_is_op_input("matmul_v2");  // -> out_linear

  auto* matmul_linear =
      pattern->NewNode(matmul_linear_repr())->assert_is_op("matmul_v2");
  auto* matmul_linear_w_var = pattern->NewNode(matmul_linear_w_repr())
                                  ->AsInput()
                                  ->assert_is_op_input("matmul_v2", "Y");
  auto* matmul_linear_out_var = pattern->NewNode(matmul_linear_out_repr())
                                    ->assert_is_op_output("matmul_v2")
                                    ->AsIntermediate()
                                    ->assert_is_op_input("elementwise_add");

  auto* eltadd_linear =
      pattern->NewNode(eltadd_linear_repr())->assert_is_op("elementwise_add");
  auto* eltadd_linear_b_var = pattern->NewNode(eltadd_linear_b_repr())
                                  ->AsInput()
                                  ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd_linear_out_var = pattern->NewNode(eltadd_linear_out_repr())
                                    ->assert_is_op_output("elementwise_add")
                                    ->AsIntermediate()
                                    ->assert_is_op_input("elementwise_add");

  auto* eltadd_out =
      pattern->NewNode(eltadd_out_repr())->assert_is_op("elementwise_add");
  auto* attention_output = pattern->NewNode(attention_output_repr())
                               ->assert_is_op_output("elementwise_add")
                               ->AsIntermediate();

  // QKV path Links
  matmul_qkv->LinksFrom({softmax_qk_out_var, transpose2_2_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});
  matmul_linear->LinksFrom({reshape2_qkv_out_var, matmul_linear_w_var})
      .LinksTo({matmul_linear_out_var});
  eltadd_linear->LinksFrom({matmul_linear_out_var, eltadd_linear_b_var})
      .LinksTo({eltadd_linear_out_var});
  eltadd_out->LinksFrom({input0, eltadd_linear_out_var})
      .LinksTo({attention_output});

  // while loop
  auto* while0 = pattern->NewNode(while0_repr())->assert_is_op("while");
  while0->LinksFrom({transpose2_1_out_var, transpose2_2_out_var});

  // Feed Forward LayerNorm Nodes
  auto* ffn_layer_norm =
      pattern->NewNode(ffn_layer_norm_repr())->assert_is_op("layer_norm");
  auto* ffn_layer_norm_scale_var =
      pattern->NewNode(ffn_layer_norm_scale_repr())
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("layer_norm", "Scale");
  auto* ffn_layer_norm_bias_var =
      pattern->NewNode(ffn_layer_norm_bias_repr())
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("layer_norm", "Bias");
  auto* ffn_layer_norm_mean_var =
      pattern->NewNode(ffn_layer_norm_mean_repr())
          ->AsOutput()
          ->assert_is_op_output("layer_norm", "Mean");
  auto* ffn_layer_norm_variance_var =
      pattern->NewNode(ffn_layer_norm_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("layer_norm", "Variance");
  auto* ffn_layer_norm_out_var = pattern->NewNode(ffn_layer_norm_out_repr())
                                     ->AsIntermediate()
                                     ->assert_is_op_output("layer_norm", "Y")
                                     ->assert_is_op_input("matmul_v2", "X");

  ffn_layer_norm
      ->LinksFrom(
          {attention_output, ffn_layer_norm_bias_var, ffn_layer_norm_scale_var})
      .LinksTo({ffn_layer_norm_out_var,
                ffn_layer_norm_mean_var,
                ffn_layer_norm_variance_var});

  // Feed Forward fc1 -> gelu -> fc2
  auto* ffn_matmul0 =
      pattern->NewNode(ffn_matmul0_repr())->assert_is_op("matmul_v2");
  auto* ffn_matmul0_w_var = pattern->NewNode(ffn_matmul0_w_repr())
                                ->AsInput()
                                ->assert_is_op_input("matmul_v2", "Y");
  auto* ffn_matmul0_out_var = pattern->NewNode(ffn_matmul0_out_repr())
                                  ->assert_is_op_output("matmul_v2")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd0 =
      pattern->NewNode(ffn_eltadd0_repr())->assert_is_op("elementwise_add");
  auto* ffn_eltadd0_b_var = pattern->NewNode(ffn_eltadd0_b_repr())
                                ->AsInput()
                                ->assert_is_op_input("elementwise_add", "Y");
  auto* ffn_eltadd0_out_var = pattern->NewNode(ffn_eltadd0_out_repr())
                                  ->assert_is_op_output("elementwise_add")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("gelu");

  auto* ffn_gelu = pattern->NewNode(ffn_gelu_repr())->assert_is_op("gelu");
  auto* ffn_gelu_out_var = pattern->NewNode(ffn_gelu_out_repr())
                               ->assert_is_op_output("gelu")
                               ->AsIntermediate()
                               ->assert_is_op_input("matmul_v2");

  auto* ffn_matmul1 =
      pattern->NewNode(ffn_matmul1_repr())->assert_is_op("matmul_v2");
  auto* ffn_matmul1_w_var = pattern->NewNode(ffn_matmul1_w_repr())
                                ->AsInput()
                                ->assert_is_op_input("matmul_v2", "Y");
  auto* ffn_matmul1_out_var = pattern->NewNode(ffn_matmul1_out_repr())
                                  ->assert_is_op_output("matmul_v2")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd1 =
      pattern->NewNode(ffn_eltadd1_repr())->assert_is_op("elementwise_add");
  auto* ffn_eltadd1_b_var = pattern->NewNode(ffn_eltadd1_b_repr())
                                ->AsInput()
                                ->assert_is_op_input("elementwise_add", "Y");
  auto* ffn_eltadd1_out_var = pattern->NewNode(ffn_eltadd1_out_repr())
                                  ->assert_is_op_output("elementwise_add")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd_out =
      pattern->NewNode(ffn_eltadd_out_repr())->assert_is_op("elementwise_add");
  auto* ffn_output = pattern->NewNode(ffn_output_repr())
                         ->assert_is_op_output("elementwise_add")
                         ->AsOutput();

  ffn_matmul0->LinksFrom({ffn_layer_norm_out_var, ffn_matmul0_w_var})
      .LinksTo({ffn_matmul0_out_var});
  ffn_eltadd0->LinksFrom({ffn_matmul0_out_var, ffn_eltadd0_b_var})
      .LinksTo({ffn_eltadd0_out_var});
  ffn_gelu->LinksFrom({ffn_eltadd0_out_var}).LinksTo({ffn_gelu_out_var});
  ffn_matmul1->LinksFrom({ffn_gelu_out_var, ffn_matmul1_w_var})
      .LinksTo({ffn_matmul1_out_var});
  ffn_eltadd1->LinksFrom({ffn_matmul1_out_var, ffn_eltadd1_b_var})
      .LinksTo({ffn_eltadd1_out_var});

  ffn_eltadd_out->LinksFrom({attention_output, ffn_eltadd1_out_var})
      .LinksTo({ffn_output});

  return ffn_output;
}

PDNode* FusedMultiTransformerEncoderFuseQKVPattern::operator()() {
  auto* input0 = pattern->NewNode(input0_repr());
  input0->assert_is_op_input("layer_norm", "X");

  // pre-LayerNorm
  auto* layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto* layer_norm_scale_var = pattern->NewNode(layer_norm_scale_repr())
                                   ->AsInput()
                                   ->assert_is_persistable_var()
                                   ->assert_is_op_input("layer_norm", "Scale");
  auto* layer_norm_bias_var = pattern->NewNode(layer_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("layer_norm", "Bias");
  auto* layer_norm_mean_var = pattern->NewNode(layer_norm_mean_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("layer_norm", "Mean");
  auto* layer_norm_variance_var =
      pattern->NewNode(layer_norm_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("layer_norm", "Variance");
  auto* layer_norm_out_var = pattern->NewNode(layer_norm_out_repr())
                                 ->AsIntermediate()
                                 ->assert_is_op_output("layer_norm", "Y")
                                 ->assert_is_op_input("matmul_v2", "X");

  layer_norm->LinksFrom({input0, layer_norm_bias_var, layer_norm_scale_var})
      .LinksTo(
          {layer_norm_out_var, layer_norm_mean_var, layer_norm_variance_var});

  // QKV fused path Nodes
  auto* matmul0 = pattern->NewNode(matmul0_repr())->assert_is_op("matmul_v2");
  auto* matmul0_w_var = pattern->NewNode(matmul0_w_repr())
                            ->AsInput()
                            ->assert_is_op_input("matmul_v2", "Y");
  auto* matmul0_out_var = pattern->NewNode(matmul0_out_repr())
                              ->assert_is_op_output("matmul_v2")
                              ->AsIntermediate()
                              ->assert_is_op_input("elementwise_add");

  auto* eltadd0 =
      pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");
  auto* eltadd0_b_var = pattern->NewNode(eltadd0_b_repr())
                            ->AsInput()
                            ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd0_out_var = pattern->NewNode(eltadd0_out_repr())
                              ->assert_is_op_output("elementwise_add")
                              ->AsIntermediate()
                              ->assert_is_op_input("reshape2");

  auto* reshape2_0 =
      pattern->NewNode(reshape2_0_repr())->assert_is_op("reshape2");
  auto* reshape2_0_out_var = pattern->NewNode(reshape2_0_out_repr())
                                 ->assert_is_op_output("reshape2")
                                 ->AsIntermediate()
                                 ->assert_is_op_input("transpose2");

  auto* transpose2_0 =
      pattern->NewNode(transpose2_0_repr())->assert_is_op("transpose2");
  auto* transpose2_0_out_var = pattern->NewNode(transpose2_0_out_repr())
                                   ->assert_is_op_output("transpose2")
                                   ->AsIntermediate()
                                   ->assert_is_op_input("split", "X");

  auto* split0 = pattern->NewNode(split0_repr())->assert_is_op("split");
  auto* split0_q_out_var = pattern->NewNode(split0_q_out_repr())
                               ->assert_is_op_output("split")
                               ->AsIntermediate()
                               ->assert_is_op_input("matmul_v2", "X");
  auto* split0_k_out_var = pattern->NewNode(split0_k_out_repr())
                               ->assert_is_op_output("split")
                               ->AsOutput()
                               ->assert_is_op_input("matmul_v2", "Y")
                               ->assert_is_op_input("while");
  auto* split0_v_out_var = pattern->NewNode(split0_v_out_repr())
                               ->assert_is_op_output("split")
                               ->AsOutput()
                               ->assert_is_op_input("matmul_v2", "Y")
                               ->assert_is_op_input("while");

  // QKV fused path Links
  matmul0->LinksFrom({layer_norm_out_var, matmul0_w_var})
      .LinksTo({matmul0_out_var});
  eltadd0->LinksFrom({matmul0_out_var, eltadd0_b_var})
      .LinksTo({eltadd0_out_var});
  reshape2_0->LinksFrom({eltadd0_out_var}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  split0->LinksFrom({transpose2_0_out_var})
      .LinksTo({split0_q_out_var, split0_k_out_var, split0_v_out_var});

  // while loop
  auto* while0 = pattern->NewNode(while0_repr())->assert_is_op("while");
  while0->LinksFrom({split0_k_out_var, split0_v_out_var});

  // QK path Nodes
  auto* matmul_qk =
      pattern->NewNode(matmul_qk_repr())->assert_is_op("matmul_v2");
  auto* matmul_qk_out_var =
      pattern->NewNode(matmul_qk_out_repr())->assert_is_op_output("matmul_v2");
  matmul_qk_out_var->AsIntermediate()->assert_is_op_input("scale");

  auto* scale_qk = pattern->NewNode(scale_qk_repr())->assert_is_op("scale");
  auto* scale_qk_out_var = pattern->NewNode(scale_qk_out_repr())
                               ->assert_is_op_output("scale")
                               ->AsIntermediate()
                               ->assert_is_op_input("elementwise_add", "X");

  auto* eltadd_qk =
      pattern->NewNode(eltadd_qk_repr())->assert_is_op("elementwise_add");
  auto* eltadd_qk_b_var = pattern->NewNode(eltadd_qk_b_repr())
                              ->AsInput()
                              ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd_qk_out_var = pattern->NewNode(eltadd_qk_out_repr())
                                ->assert_is_op_output("elementwise_add")
                                ->AsIntermediate()
                                ->assert_is_op_input("softmax");

  auto* softmax_qk =
      pattern->NewNode(softmax_qk_repr())->assert_is_op("softmax");
  auto* softmax_qk_out_var = pattern->NewNode(softmax_qk_out_repr())
                                 ->assert_is_op_output("softmax")
                                 ->AsIntermediate()
                                 ->assert_is_op_input("matmul_v2", "X");

  // QK path Linsk
  matmul_qk->LinksFrom({split0_q_out_var, split0_k_out_var})
      .LinksTo({matmul_qk_out_var});
  scale_qk->LinksFrom({matmul_qk_out_var}).LinksTo({scale_qk_out_var});
  eltadd_qk->LinksFrom({scale_qk_out_var, eltadd_qk_b_var})
      .LinksTo({eltadd_qk_out_var});
  softmax_qk->LinksFrom({eltadd_qk_out_var}).LinksTo({softmax_qk_out_var});

  // QKV path Nodes
  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_op("matmul_v2");
  auto* matmul_qkv_out_var =
      pattern->NewNode(matmul_qkv_out_repr())->assert_is_op_output("matmul_v2");
  matmul_qkv_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_qkv =
      pattern->NewNode(transpose2_qkv_repr())->assert_is_op("transpose2");
  auto* transpose2_qkv_out_var = pattern->NewNode(transpose2_qkv_out_repr())
                                     ->assert_is_op_output("transpose2");
  transpose2_qkv_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_qkv =
      pattern->NewNode(reshape2_qkv_repr())->assert_is_op("reshape2");
  auto* reshape2_qkv_out_var =
      pattern->NewNode(reshape2_qkv_out_repr())
          ->assert_is_op_output("reshape2")
          ->AsIntermediate()
          ->assert_is_op_input("matmul_v2");  // -> out_linear

  auto* matmul_linear =
      pattern->NewNode(matmul_linear_repr())->assert_is_op("matmul_v2");
  auto* matmul_linear_w_var = pattern->NewNode(matmul_linear_w_repr())
                                  ->AsInput()
                                  ->assert_is_op_input("matmul_v2", "Y");
  auto* matmul_linear_out_var = pattern->NewNode(matmul_linear_out_repr())
                                    ->assert_is_op_output("matmul_v2")
                                    ->AsIntermediate()
                                    ->assert_is_op_input("elementwise_add");

  auto* eltadd_linear =
      pattern->NewNode(eltadd_linear_repr())->assert_is_op("elementwise_add");
  auto* eltadd_linear_b_var = pattern->NewNode(eltadd_linear_b_repr())
                                  ->AsInput()
                                  ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd_linear_out_var = pattern->NewNode(eltadd_linear_out_repr())
                                    ->assert_is_op_output("elementwise_add")
                                    ->AsIntermediate()
                                    ->assert_is_op_input("elementwise_add");

  auto* eltadd_out =
      pattern->NewNode(eltadd_out_repr())->assert_is_op("elementwise_add");
  auto* attention_output = pattern->NewNode(attention_output_repr())
                               ->assert_is_op_output("elementwise_add")
                               ->AsIntermediate();

  // QKV path Links
  matmul_qkv->LinksFrom({softmax_qk_out_var, split0_v_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});
  matmul_linear->LinksFrom({reshape2_qkv_out_var, matmul_linear_w_var})
      .LinksTo({matmul_linear_out_var});
  eltadd_linear->LinksFrom({matmul_linear_out_var, eltadd_linear_b_var})
      .LinksTo({eltadd_linear_out_var});
  eltadd_out->LinksFrom({input0, eltadd_linear_out_var})
      .LinksTo({attention_output});

  // Feed Forward LayerNorm Nodes
  auto* ffn_layer_norm =
      pattern->NewNode(ffn_layer_norm_repr())->assert_is_op("layer_norm");
  auto* ffn_layer_norm_scale_var =
      pattern->NewNode(ffn_layer_norm_scale_repr())
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("layer_norm", "Scale");
  auto* ffn_layer_norm_bias_var =
      pattern->NewNode(ffn_layer_norm_bias_repr())
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("layer_norm", "Bias");
  auto* ffn_layer_norm_mean_var =
      pattern->NewNode(ffn_layer_norm_mean_repr())
          ->AsIntermediate()
          ->assert_is_op_output("layer_norm", "Mean");
  auto* ffn_layer_norm_variance_var =
      pattern->NewNode(ffn_layer_norm_variance_repr())
          ->AsIntermediate()
          ->assert_is_op_output("layer_norm", "Variance");
  auto* ffn_layer_norm_out_var = pattern->NewNode(ffn_layer_norm_out_repr())
                                     ->AsIntermediate()
                                     ->assert_is_op_output("layer_norm", "Y")
                                     ->assert_is_op_input("matmul_v2", "X");

  ffn_layer_norm
      ->LinksFrom(
          {attention_output, ffn_layer_norm_bias_var, ffn_layer_norm_scale_var})
      .LinksTo({ffn_layer_norm_out_var,
                ffn_layer_norm_mean_var,
                ffn_layer_norm_variance_var});

  // Feed Forward fc1 -> gelu -> fc2
  auto* ffn_matmul0 =
      pattern->NewNode(ffn_matmul0_repr())->assert_is_op("matmul_v2");
  auto* ffn_matmul0_w_var = pattern->NewNode(ffn_matmul0_w_repr())
                                ->AsInput()
                                ->assert_is_op_input("matmul_v2", "Y");
  auto* ffn_matmul0_out_var = pattern->NewNode(ffn_matmul0_out_repr())
                                  ->assert_is_op_output("matmul_v2")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd0 =
      pattern->NewNode(ffn_eltadd0_repr())->assert_is_op("elementwise_add");
  auto* ffn_eltadd0_b_var = pattern->NewNode(ffn_eltadd0_b_repr())
                                ->AsInput()
                                ->assert_is_op_input("elementwise_add", "Y");
  auto* ffn_eltadd0_out_var = pattern->NewNode(ffn_eltadd0_out_repr())
                                  ->assert_is_op_output("elementwise_add")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("gelu");

  auto* ffn_gelu = pattern->NewNode(ffn_gelu_repr())->assert_is_op("gelu");
  auto* ffn_gelu_out_var = pattern->NewNode(ffn_gelu_out_repr())
                               ->assert_is_op_output("gelu")
                               ->AsIntermediate()
                               ->assert_is_op_input("matmul_v2");

  auto* ffn_matmul1 =
      pattern->NewNode(ffn_matmul1_repr())->assert_is_op("matmul_v2");
  auto* ffn_matmul1_w_var = pattern->NewNode(ffn_matmul1_w_repr())
                                ->AsInput()
                                ->assert_is_op_input("matmul_v2", "Y");
  auto* ffn_matmul1_out_var = pattern->NewNode(ffn_matmul1_out_repr())
                                  ->assert_is_op_output("matmul_v2")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd1 =
      pattern->NewNode(ffn_eltadd1_repr())->assert_is_op("elementwise_add");
  auto* ffn_eltadd1_b_var = pattern->NewNode(ffn_eltadd1_b_repr())
                                ->AsInput()
                                ->assert_is_op_input("elementwise_add", "Y");
  auto* ffn_eltadd1_out_var = pattern->NewNode(ffn_eltadd1_out_repr())
                                  ->assert_is_op_output("elementwise_add")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd_out =
      pattern->NewNode(ffn_eltadd_out_repr())->assert_is_op("elementwise_add");
  auto* ffn_output = pattern->NewNode(ffn_output_repr())
                         ->assert_is_op_output("elementwise_add")
                         ->AsOutput();

  ffn_matmul0->LinksFrom({ffn_layer_norm_out_var, ffn_matmul0_w_var})
      .LinksTo({ffn_matmul0_out_var});
  ffn_eltadd0->LinksFrom({ffn_matmul0_out_var, ffn_eltadd0_b_var})
      .LinksTo({ffn_eltadd0_out_var});
  ffn_gelu->LinksFrom({ffn_eltadd0_out_var}).LinksTo({ffn_gelu_out_var});
  ffn_matmul1->LinksFrom({ffn_gelu_out_var, ffn_matmul1_w_var})
      .LinksTo({ffn_matmul1_out_var});
  ffn_eltadd1->LinksFrom({ffn_matmul1_out_var, ffn_eltadd1_b_var})
      .LinksTo({ffn_eltadd1_out_var});

  ffn_eltadd_out->LinksFrom({attention_output, ffn_eltadd1_out_var})
      .LinksTo({ffn_output});

  return ffn_output;
}

PDNode* MultiDevicesFusedMultiTransformerEncoderFuseQKVPattern::operator()() {
  auto* input0 = pattern->NewNode(input0_repr());
  input0->assert_is_op_input("layer_norm", "X");

  // pre-LayerNorm
  auto* layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto* layer_norm_scale_var = pattern->NewNode(layer_norm_scale_repr())
                                   ->AsInput()
                                   ->assert_is_persistable_var()
                                   ->assert_is_op_input("layer_norm", "Scale");
  auto* layer_norm_bias_var = pattern->NewNode(layer_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("layer_norm", "Bias");
  auto* layer_norm_mean_var = pattern->NewNode(layer_norm_mean_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("layer_norm", "Mean");
  auto* layer_norm_variance_var =
      pattern->NewNode(layer_norm_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("layer_norm", "Variance");
  auto* layer_norm_out_var = pattern->NewNode(layer_norm_out_repr())
                                 ->AsIntermediate()
                                 ->assert_is_op_output("layer_norm", "Y")
                                 ->assert_is_op_input("c_identity", "X");

  layer_norm->LinksFrom({input0, layer_norm_bias_var, layer_norm_scale_var})
      .LinksTo(
          {layer_norm_out_var, layer_norm_mean_var, layer_norm_variance_var});

  // communication c_identity
  auto* c_identity =
      pattern->NewNode(c_identity_repr())->assert_is_op("c_identity");
  auto* c_identity_out_var = pattern->NewNode(c_identity_out_repr())
                                 ->AsIntermediate()
                                 ->assert_is_op_output("c_identity", "Out")
                                 ->assert_is_op_input("matmul_v2", "X");
  c_identity->LinksFrom({layer_norm_out_var}).LinksTo({c_identity_out_var});

  // QKV fused path Nodes
  auto* matmul0 = pattern->NewNode(matmul0_repr())->assert_is_op("matmul_v2");
  auto* matmul0_w_var = pattern->NewNode(matmul0_w_repr())
                            ->AsInput()
                            ->assert_is_op_input("matmul_v2", "Y");
  auto* matmul0_out_var = pattern->NewNode(matmul0_out_repr())
                              ->assert_is_op_output("matmul_v2")
                              ->AsIntermediate()
                              ->assert_is_op_input("elementwise_add");

  auto* eltadd0 =
      pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");
  auto* eltadd0_b_var = pattern->NewNode(eltadd0_b_repr())
                            ->AsInput()
                            ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd0_out_var = pattern->NewNode(eltadd0_out_repr())
                              ->assert_is_op_output("elementwise_add")
                              ->AsIntermediate()
                              ->assert_is_op_input("reshape2");

  auto* reshape2_0 =
      pattern->NewNode(reshape2_0_repr())->assert_is_op("reshape2");
  auto* reshape2_0_out_var = pattern->NewNode(reshape2_0_out_repr())
                                 ->assert_is_op_output("reshape2")
                                 ->AsIntermediate()
                                 ->assert_is_op_input("transpose2");

  auto* transpose2_0 =
      pattern->NewNode(transpose2_0_repr())->assert_is_op("transpose2");
  auto* transpose2_0_out_var = pattern->NewNode(transpose2_0_out_repr())
                                   ->assert_is_op_output("transpose2")
                                   ->AsIntermediate()
                                   ->assert_is_op_input("split", "X");

  auto* split0 = pattern->NewNode(split0_repr())->assert_is_op("split");
  auto* split0_q_out_var = pattern->NewNode(split0_q_out_repr())
                               ->assert_is_op_output("split")
                               ->AsIntermediate()
                               ->assert_is_op_input("matmul_v2", "X");
  auto* split0_k_out_var = pattern->NewNode(split0_k_out_repr())
                               ->assert_is_op_output("split")
                               ->AsOutput()
                               ->assert_is_op_input("matmul_v2", "Y")
                               ->assert_is_op_input("while");
  auto* split0_v_out_var = pattern->NewNode(split0_v_out_repr())
                               ->assert_is_op_output("split")
                               ->AsOutput()
                               ->assert_is_op_input("matmul_v2", "Y")
                               ->assert_is_op_input("while");

  // QKV fused path Links
  matmul0->LinksFrom({c_identity_out_var, matmul0_w_var})
      .LinksTo({matmul0_out_var});
  eltadd0->LinksFrom({matmul0_out_var, eltadd0_b_var})
      .LinksTo({eltadd0_out_var});
  reshape2_0->LinksFrom({eltadd0_out_var}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  split0->LinksFrom({transpose2_0_out_var})
      .LinksTo({split0_q_out_var, split0_k_out_var, split0_v_out_var});

  // while loop
  auto* while0 = pattern->NewNode(while0_repr())->assert_is_op("while");
  while0->LinksFrom({split0_k_out_var, split0_v_out_var});

  // QK path Nodes
  auto* matmul_qk =
      pattern->NewNode(matmul_qk_repr())->assert_is_op("matmul_v2");
  auto* matmul_qk_out_var =
      pattern->NewNode(matmul_qk_out_repr())->assert_is_op_output("matmul_v2");
  matmul_qk_out_var->AsIntermediate()->assert_is_op_input("scale");

  auto* scale_qk = pattern->NewNode(scale_qk_repr())->assert_is_op("scale");
  auto* scale_qk_out_var = pattern->NewNode(scale_qk_out_repr())
                               ->assert_is_op_output("scale")
                               ->AsIntermediate()
                               ->assert_is_op_input("elementwise_add", "X");

  auto* eltadd_qk =
      pattern->NewNode(eltadd_qk_repr())->assert_is_op("elementwise_add");
  auto* eltadd_qk_b_var = pattern->NewNode(eltadd_qk_b_repr())
                              ->AsInput()
                              ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd_qk_out_var = pattern->NewNode(eltadd_qk_out_repr())
                                ->assert_is_op_output("elementwise_add")
                                ->AsIntermediate()
                                ->assert_is_op_input("softmax");

  auto* softmax_qk =
      pattern->NewNode(softmax_qk_repr())->assert_is_op("softmax");
  auto* softmax_qk_out_var = pattern->NewNode(softmax_qk_out_repr())
                                 ->assert_is_op_output("softmax")
                                 ->AsIntermediate()
                                 ->assert_is_op_input("matmul_v2", "X");

  // QK path Linsk
  matmul_qk->LinksFrom({split0_q_out_var, split0_k_out_var})
      .LinksTo({matmul_qk_out_var});
  scale_qk->LinksFrom({matmul_qk_out_var}).LinksTo({scale_qk_out_var});
  eltadd_qk->LinksFrom({scale_qk_out_var, eltadd_qk_b_var})
      .LinksTo({eltadd_qk_out_var});
  softmax_qk->LinksFrom({eltadd_qk_out_var}).LinksTo({softmax_qk_out_var});

  // QKV path Nodes
  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_op("matmul_v2");
  auto* matmul_qkv_out_var =
      pattern->NewNode(matmul_qkv_out_repr())->assert_is_op_output("matmul_v2");
  matmul_qkv_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_qkv =
      pattern->NewNode(transpose2_qkv_repr())->assert_is_op("transpose2");
  auto* transpose2_qkv_out_var = pattern->NewNode(transpose2_qkv_out_repr())
                                     ->assert_is_op_output("transpose2");
  transpose2_qkv_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_qkv =
      pattern->NewNode(reshape2_qkv_repr())->assert_is_op("reshape2");
  auto* reshape2_qkv_out_var =
      pattern->NewNode(reshape2_qkv_out_repr())
          ->assert_is_op_output("reshape2")
          ->AsIntermediate()
          ->assert_is_op_input("matmul_v2");  // -> out_linear

  auto* matmul_linear =
      pattern->NewNode(matmul_linear_repr())->assert_is_op("matmul_v2");
  auto* matmul_linear_w_var = pattern->NewNode(matmul_linear_w_repr())
                                  ->AsInput()
                                  ->assert_is_op_input("matmul_v2", "Y");
  auto* matmul_linear_out_var = pattern->NewNode(matmul_linear_out_repr())
                                    ->assert_is_op_output("matmul_v2")
                                    ->AsIntermediate()
                                    ->assert_is_op_input("c_allreduce_sum");

  // communication c_allreduce_sum
  auto* c_allreduce_sum =
      pattern->NewNode(c_allreduce_sum_repr())->assert_is_op("c_allreduce_sum");
  auto* c_allreduce_sum_out_var = pattern->NewNode(c_allreduce_sum_out_repr())
                                      ->assert_is_op_output("c_allreduce_sum")
                                      ->AsIntermediate()
                                      ->assert_is_op_input("elementwise_add");

  auto* eltadd_linear =
      pattern->NewNode(eltadd_linear_repr())->assert_is_op("elementwise_add");
  auto* eltadd_linear_b_var = pattern->NewNode(eltadd_linear_b_repr())
                                  ->AsInput()
                                  ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd_linear_out_var = pattern->NewNode(eltadd_linear_out_repr())
                                    ->assert_is_op_output("elementwise_add")
                                    ->AsIntermediate()
                                    ->assert_is_op_input("elementwise_add");

  auto* eltadd_out =
      pattern->NewNode(eltadd_out_repr())->assert_is_op("elementwise_add");
  auto* attention_output = pattern->NewNode(attention_output_repr())
                               ->assert_is_op_output("elementwise_add")
                               ->AsIntermediate();

  // QKV path Links
  matmul_qkv->LinksFrom({softmax_qk_out_var, split0_v_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});
  matmul_linear->LinksFrom({reshape2_qkv_out_var, matmul_linear_w_var})
      .LinksTo({matmul_linear_out_var});
  c_allreduce_sum->LinksFrom({matmul_linear_out_var})
      .LinksTo({c_allreduce_sum_out_var});
  eltadd_linear->LinksFrom({c_allreduce_sum_out_var, eltadd_linear_b_var})
      .LinksTo({eltadd_linear_out_var});
  eltadd_out->LinksFrom({input0, eltadd_linear_out_var})
      .LinksTo({attention_output});

  // Feed Forward LayerNorm Nodes
  auto* ffn_layer_norm =
      pattern->NewNode(ffn_layer_norm_repr())->assert_is_op("layer_norm");
  auto* ffn_layer_norm_scale_var =
      pattern->NewNode(ffn_layer_norm_scale_repr())
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("layer_norm", "Scale");
  auto* ffn_layer_norm_bias_var =
      pattern->NewNode(ffn_layer_norm_bias_repr())
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("layer_norm", "Bias");
  auto* ffn_layer_norm_mean_var =
      pattern->NewNode(ffn_layer_norm_mean_repr())
          ->AsIntermediate()
          ->assert_is_op_output("layer_norm", "Mean");
  auto* ffn_layer_norm_variance_var =
      pattern->NewNode(ffn_layer_norm_variance_repr())
          ->AsIntermediate()
          ->assert_is_op_output("layer_norm", "Variance");
  auto* ffn_layer_norm_out_var = pattern->NewNode(ffn_layer_norm_out_repr())
                                     ->AsIntermediate()
                                     ->assert_is_op_output("layer_norm", "Y")
                                     ->assert_is_op_input("c_identity", "X");

  ffn_layer_norm
      ->LinksFrom(
          {attention_output, ffn_layer_norm_bias_var, ffn_layer_norm_scale_var})
      .LinksTo({ffn_layer_norm_out_var,
                ffn_layer_norm_mean_var,
                ffn_layer_norm_variance_var});

  // communication c_identity
  auto* ffn_c_identity =
      pattern->NewNode(ffn_c_identity_repr())->assert_is_op("c_identity");
  auto* ffn_c_identity_out_var = pattern->NewNode(ffn_c_identity_out_repr())
                                     ->assert_is_op_output("c_identity", "Out")
                                     ->AsIntermediate()
                                     ->assert_is_op_input("matmul_v2", "X");
  ffn_c_identity->LinksFrom({ffn_layer_norm_out_var})
      .LinksTo({ffn_c_identity_out_var});

  // Feed Forward fc1 -> gelu -> fc2
  auto* ffn_matmul0 =
      pattern->NewNode(ffn_matmul0_repr())->assert_is_op("matmul_v2");
  auto* ffn_matmul0_w_var = pattern->NewNode(ffn_matmul0_w_repr())
                                ->AsInput()
                                ->assert_is_op_input("matmul_v2", "Y");
  auto* ffn_matmul0_out_var = pattern->NewNode(ffn_matmul0_out_repr())
                                  ->assert_is_op_output("matmul_v2")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd0 =
      pattern->NewNode(ffn_eltadd0_repr())->assert_is_op("elementwise_add");
  auto* ffn_eltadd0_b_var = pattern->NewNode(ffn_eltadd0_b_repr())
                                ->AsInput()
                                ->assert_is_op_input("elementwise_add", "Y");
  auto* ffn_eltadd0_out_var = pattern->NewNode(ffn_eltadd0_out_repr())
                                  ->assert_is_op_output("elementwise_add")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("gelu");

  auto* ffn_gelu = pattern->NewNode(ffn_gelu_repr())->assert_is_op("gelu");
  auto* ffn_gelu_out_var = pattern->NewNode(ffn_gelu_out_repr())
                               ->assert_is_op_output("gelu")
                               ->AsIntermediate()
                               ->assert_is_op_input("matmul_v2");

  auto* ffn_matmul1 =
      pattern->NewNode(ffn_matmul1_repr())->assert_is_op("matmul_v2");
  auto* ffn_matmul1_w_var = pattern->NewNode(ffn_matmul1_w_repr())
                                ->AsInput()
                                ->assert_is_op_input("matmul_v2", "Y");
  auto* ffn_matmul1_out_var = pattern->NewNode(ffn_matmul1_out_repr())
                                  ->assert_is_op_output("matmul_v2")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("c_allreduce_sum");

  // communication c_allreduce_sum
  auto* ffn_c_allreduce_sum = pattern->NewNode(ffn_c_allreduce_sum_repr())
                                  ->assert_is_op("c_allreduce_sum");
  auto* ffn_c_allreduce_sum_out_var =
      pattern->NewNode(ffn_c_allreduce_sum_out_repr())
          ->assert_is_op_output("c_allreduce_sum")
          ->AsIntermediate()
          ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd1 =
      pattern->NewNode(ffn_eltadd1_repr())->assert_is_op("elementwise_add");
  auto* ffn_eltadd1_b_var = pattern->NewNode(ffn_eltadd1_b_repr())
                                ->AsInput()
                                ->assert_is_op_input("elementwise_add", "Y");
  auto* ffn_eltadd1_out_var = pattern->NewNode(ffn_eltadd1_out_repr())
                                  ->assert_is_op_output("elementwise_add")
                                  ->AsIntermediate()
                                  ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd_out =
      pattern->NewNode(ffn_eltadd_out_repr())->assert_is_op("elementwise_add");
  auto* ffn_output = pattern->NewNode(ffn_output_repr())
                         ->assert_is_op_output("elementwise_add")
                         ->AsOutput();

  ffn_matmul0->LinksFrom({ffn_c_identity_out_var, ffn_matmul0_w_var})
      .LinksTo({ffn_matmul0_out_var});
  ffn_eltadd0->LinksFrom({ffn_matmul0_out_var, ffn_eltadd0_b_var})
      .LinksTo({ffn_eltadd0_out_var});
  ffn_gelu->LinksFrom({ffn_eltadd0_out_var}).LinksTo({ffn_gelu_out_var});
  ffn_matmul1->LinksFrom({ffn_gelu_out_var, ffn_matmul1_w_var})
      .LinksTo({ffn_matmul1_out_var});
  ffn_c_allreduce_sum->LinksFrom({ffn_matmul1_out_var})
      .LinksTo({ffn_c_allreduce_sum_out_var});
  ffn_eltadd1->LinksFrom({ffn_c_allreduce_sum_out_var, ffn_eltadd1_b_var})
      .LinksTo({ffn_eltadd1_out_var});

  ffn_eltadd_out->LinksFrom({attention_output, ffn_eltadd1_out_var})
      .LinksTo({ffn_output});

  return ffn_output;
}

}  // namespace patterns

template <typename T>
inline void QKVWeightsProcess(phi::DenseTensor* wq_tensor,
                              phi::DenseTensor* wk_tensor,
                              phi::DenseTensor* wv_tensor,
                              const int num_head,
                              const int dim_head,
                              const int dim_embed) {
  auto* wq_data = wq_tensor->mutable_data<T>(platform::CPUPlace());
  auto* wk_data = wk_tensor->mutable_data<T>(platform::CPUPlace());
  auto* wv_data = wv_tensor->mutable_data<T>(platform::CPUPlace());

  auto combined_w_dims = phi::make_ddim({3, num_head, dim_head, dim_embed});

  phi::DenseTensor tmp_combined_w_tensor;
  tmp_combined_w_tensor.Resize(combined_w_dims);
  auto* tmp_combined_w_data =
      tmp_combined_w_tensor.mutable_data<T>(platform::CPUPlace());

  std::vector<T*> w_vec = {wq_data, wk_data, wv_data};
  // Combine the three fc weights together.
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < num_head; j++) {
      for (int k = 0; k < dim_head; k++) {
        for (int l = 0; l < dim_embed; l++) {
          int out_idx = i * num_head * dim_head * dim_embed +
                        j * dim_head * dim_embed + k * dim_embed + l;
          int in_idx = l * num_head * dim_head + j * dim_head + k;
          tmp_combined_w_data[out_idx] = w_vec[i][in_idx];
        }
      }
    }
  }

  wq_tensor->Resize(combined_w_dims);
  auto* new_combined_w_data = wq_tensor->mutable_data<T>(platform::CPUPlace());
  memcpy(
      new_combined_w_data, tmp_combined_w_data, sizeof(T) * wq_tensor->numel());
}

template <typename T>
inline void QKVBiasProcess(phi::DenseTensor* bq_tensor,
                           phi::DenseTensor* bk_tensor,
                           phi::DenseTensor* bv_tensor,
                           const int num_head,
                           const int dim_head,
                           const int dim_embed) {
  auto* bq_data = bq_tensor->mutable_data<T>(platform::CPUPlace());
  auto* bk_data = bk_tensor->mutable_data<T>(platform::CPUPlace());
  auto* bv_data = bv_tensor->mutable_data<T>(platform::CPUPlace());

  auto combined_bias_dims = phi::make_ddim({3, num_head, dim_head});

  phi::DenseTensor tmp_combined_bias_tensor;
  tmp_combined_bias_tensor.Resize(combined_bias_dims);
  auto* tmp_combined_bias_data =
      tmp_combined_bias_tensor.mutable_data<T>(platform::CPUPlace());

  size_t bias_size = bq_tensor->numel();
  memcpy(tmp_combined_bias_data, bq_data, sizeof(T) * bias_size);
  memcpy(tmp_combined_bias_data + bias_size, bk_data, sizeof(T) * bias_size);
  memcpy(
      tmp_combined_bias_data + 2 * bias_size, bv_data, sizeof(T) * bias_size);

  bq_tensor->Resize(combined_bias_dims);
  auto* new_combined_bias_data =
      bq_tensor->mutable_data<T>(platform::CPUPlace());
  memcpy(new_combined_bias_data,
         tmp_combined_bias_data,
         sizeof(T) * bq_tensor->numel());
}

inline void QKVWeightsBiasProcess(phi::DenseTensor* wq_tensor,
                                  phi::DenseTensor* wk_tensor,
                                  phi::DenseTensor* wv_tensor,
                                  phi::DenseTensor* bq_tensor,
                                  phi::DenseTensor* bk_tensor,
                                  phi::DenseTensor* bv_tensor,
                                  const int num_head,
                                  const int dim_head,
                                  const int dim_embed) {
  switch (wq_tensor->dtype()) {
    case paddle::experimental::DataType::FLOAT16:
      QKVWeightsProcess<platform::float16>(
          wq_tensor, wk_tensor, wv_tensor, num_head, dim_head, dim_embed);
      break;
    case paddle::experimental::DataType::FLOAT32:
      QKVWeightsProcess<float>(
          wq_tensor, wk_tensor, wv_tensor, num_head, dim_head, dim_embed);
      break;
    case paddle::experimental::DataType::INT8:
      QKVWeightsProcess<int8_t>(
          wq_tensor, wk_tensor, wv_tensor, num_head, dim_head, dim_embed);
      break;
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "fused_multi_transformer not supported weight dtype. "
          "we now only support fp32/fp16/int8."));
      break;
  }
  switch (bq_tensor->dtype()) {
    case paddle::experimental::DataType::FLOAT16:
      QKVBiasProcess<platform::float16>(
          bq_tensor, bk_tensor, bv_tensor, num_head, dim_head, dim_embed);
      break;
    case paddle::experimental::DataType::FLOAT32:
      QKVBiasProcess<float>(
          bq_tensor, bk_tensor, bv_tensor, num_head, dim_head, dim_embed);
      break;
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "fused_multi_transformer not supported bias dtype. "
          "we now only support fp32/fp16."));
      break;
  }
}

template <typename T>
inline void QKVWeightsProcessFuseQKV(phi::DenseTensor* qkv_w_tensor,
                                     const int num_head,
                                     const int dim_head,
                                     const int dim_embed) {
  auto* qkv_w_data = qkv_w_tensor->data<T>();
  auto transpose_w_dims = phi::make_ddim({3, num_head, dim_head, dim_embed});

  phi::DenseTensor tmp_transpose_w_tensor;
  tmp_transpose_w_tensor.Resize(transpose_w_dims);
  auto* tmp_transpose_w_data =
      tmp_transpose_w_tensor.mutable_data<T>(platform::CPUPlace());

  // transpose qkv matmul Y to QKVWeights
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < num_head; j++) {
      for (int k = 0; k < dim_head; k++) {
        for (int l = 0; l < dim_embed; l++) {
          int out_idx = i * num_head * dim_head * dim_embed +
                        j * dim_head * dim_embed + k * dim_embed + l;
          int in_idx =
              l * num_head * 3 * dim_head + j * 3 * dim_head + i * dim_head + k;
          tmp_transpose_w_data[out_idx] = qkv_w_data[in_idx];
        }
      }
    }
  }

  qkv_w_tensor->Resize(transpose_w_dims);
  auto* new_transpose_w_data =
      qkv_w_tensor->mutable_data<T>(platform::CPUPlace());
  memcpy(new_transpose_w_data,
         tmp_transpose_w_data,
         sizeof(T) * qkv_w_tensor->numel());
}

template <typename T>
inline void QKVBiasProcessFuseQKV(phi::DenseTensor* qkv_b_tensor,
                                  const int num_head,
                                  const int dim_head,
                                  const int dim_embed) {
  auto* qkv_b_data = qkv_b_tensor->data<T>();
  auto transpose_b_dims = phi::make_ddim({3, num_head, dim_head});

  phi::DenseTensor tmp_transpose_b_tensor;
  tmp_transpose_b_tensor.Resize(transpose_b_dims);
  auto* tmp_transpose_b_data =
      tmp_transpose_b_tensor.mutable_data<T>(platform::CPUPlace());

  // transpose qkv elemenwise_add Y to QKVBias
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < num_head; j++) {
      for (int k = 0; k < dim_head; k++) {
        int out_idx = i * num_head * dim_head + j * dim_head + k;
        int in_idx = j * 3 * dim_head + i * dim_head + k;
        tmp_transpose_b_data[out_idx] = qkv_b_data[in_idx];
      }
    }
  }

  qkv_b_tensor->Resize({3, num_head, dim_head});
  auto* new_transpose_b_data =
      qkv_b_tensor->mutable_data<T>(platform::CPUPlace());
  memcpy(new_transpose_b_data,
         tmp_transpose_b_data,
         sizeof(T) * qkv_b_tensor->numel());
}

inline void QKVWeightsBiasProcessFuseQKV(phi::DenseTensor* qkv_w_tensor,
                                         phi::DenseTensor* qkv_b_tensor,
                                         const int num_head,
                                         const int dim_head,
                                         const int dim_embed) {
  switch (qkv_w_tensor->dtype()) {
    case paddle::experimental::DataType::FLOAT16:
      QKVWeightsProcessFuseQKV<platform::float16>(
          qkv_w_tensor, num_head, dim_head, dim_embed);
      break;
    case paddle::experimental::DataType::FLOAT32:
      QKVWeightsProcessFuseQKV<float>(
          qkv_w_tensor, num_head, dim_head, dim_embed);
      break;
    case paddle::experimental::DataType::INT8:
      QKVWeightsProcessFuseQKV<int8_t>(
          qkv_w_tensor, num_head, dim_head, dim_embed);
      break;
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "fused_multi_transformer not supported weight dtype. "
          "we now only support fp32/fp16/int8."));
      break;
  }
  switch (qkv_b_tensor->dtype()) {
    case paddle::experimental::DataType::FLOAT16:
      QKVBiasProcessFuseQKV<platform::float16>(
          qkv_b_tensor, num_head, dim_head, dim_embed);
      break;
    case paddle::experimental::DataType::FLOAT32:
      QKVBiasProcessFuseQKV<float>(qkv_b_tensor, num_head, dim_head, dim_embed);
      break;
    default:
      PADDLE_THROW(platform::errors::Unavailable(
          "fused_multi_transformer not supported bias dtype. "
          "we now only support fp32/fp16."));
      break;
  }
}

// Just use for fused_multi_transformer_int8
inline void TransposeWeights(phi::DenseTensor* weight_tensor) {
  int m = weight_tensor->dims()[0];
  int n = weight_tensor->dims()[1];
  phi::DenseTensor tmp_weight_tensor;
  auto tmp_weight_data =
      tmp_weight_tensor.mutable_data<int8_t>({n, m}, platform::CPUPlace());
  auto weight_data = weight_tensor->data<int8_t>();
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int in_idx = i * n + j;
      int out_idx = j * m + i;
      tmp_weight_data[out_idx] = weight_data[in_idx];
    }
  }
  weight_tensor->Resize({n, m});
  auto new_weight_data =
      weight_tensor->mutable_data<int8_t>(platform::CPUPlace());
  memcpy(new_weight_data, tmp_weight_data, sizeof(int8_t) * m * n);
}

inline Node* CreatePersistableVarNode(Graph* graph, const std::string& name) {
  auto var_desc = VarDesc(name);
  var_desc.SetDataType(framework::proto::VarType::FP32);
  var_desc.SetPersistable(true);
  auto node = graph->CreateVarNode(&var_desc);
  return node;
}

int FusedMultiTransformerEncoderPass::BuildFusion(Graph* graph,
                                                  const std::string& name_scope,
                                                  Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  bool enable_int8 = graph->Get<bool>("enable_int8");
  if (enable_int8) {
    VLOG(3) << "FusedMultiTransformerEncoderPass with int8";
  } else {
    VLOG(3) << "FusedMultiTransformerEncoderPass with fp";
  }

  // Create pattern.
  patterns::FusedMultiTransformerEncoderPattern fused_multi_transformer_pattern(
      pattern, name_scope);
  fused_multi_transformer_pattern();

  // Create New OpDesc
  auto fuse_creater = [&](Node* input0,
                          Node* layer_norm,
                          Node* layer_norm_scale,
                          Node* layer_norm_bias,
                          Node* layer_norm_mean,
                          Node* layer_norm_variance,
                          Node* matmul0,
                          Node* matmul0_w,
                          Node* matmul1_w,
                          Node* matmul2_w,
                          Node* eltadd0_b,
                          Node* eltadd1_b,
                          Node* eltadd2_b,
                          Node* transpose2_1_out,
                          Node* transpose2_2_out,
                          Node* eltadd_qk_b,
                          Node* reshape2_0,
                          Node* matmul_linear,
                          Node* matmul_linear_w,
                          Node* eltadd_linear_b,
                          Node* while0,
                          Node* ffn_layer_norm,
                          Node* ffn_layer_norm_scale,
                          Node* ffn_layer_norm_bias,
                          Node* ffn_layer_norm_mean,
                          Node* ffn_layer_norm_variance,
                          Node* ffn_matmul0,
                          Node* ffn_matmul0_w,
                          Node* ffn_matmul1,
                          Node* ffn_matmul1_w,
                          Node* ffn_eltadd0_b,
                          Node* ffn_eltadd1_b,
                          Node* ffn_output) {
    auto* matmul0_op = matmul0->Op();
    auto* matmul_linear_op = matmul_linear->Op();
    auto* ffn_matmul_0_op = ffn_matmul0->Op();
    auto* ffn_matmul_1_op = ffn_matmul1->Op();

    // Calc index of transformer layer by LayerNorm Scale name
    // This calculation assumes:
    //    1. no LayerNorm before all transformer layer
    //    2. each transformer layer contains 2 LayerNorm layer
    auto ln_scale_name = layer_norm_scale->Name();
    auto ln_name = ln_scale_name.substr(0, ln_scale_name.find('.'));
    auto ln_idx_str = ln_name.substr(ln_name.rfind('_') + 1);
    int layer_idx = atoi(ln_idx_str.c_str()) / 2;

    auto* wq_tensor =
        scope->FindVar(matmul0_w->Name())->GetMutable<phi::DenseTensor>();
    auto* wk_tensor =
        scope->FindVar(matmul1_w->Name())->GetMutable<phi::DenseTensor>();
    auto* wv_tensor =
        scope->FindVar(matmul2_w->Name())->GetMutable<phi::DenseTensor>();

    auto* bq_tensor =
        scope->FindVar(eltadd0_b->Name())->GetMutable<phi::DenseTensor>();
    auto* bk_tensor =
        scope->FindVar(eltadd1_b->Name())->GetMutable<phi::DenseTensor>();
    auto* bv_tensor =
        scope->FindVar(eltadd2_b->Name())->GetMutable<phi::DenseTensor>();

    // NOTE(minghaoBD): to make it compatible with strucutured pruning on
    // num_head dimension:
    // 1. get dim_head from reshape.shape[3], dim_embed from
    // layer_norm_bias.shape[0]
    // 2. calculate num_head according to wq_tensor.shape[1] and dim_head
    auto reshape_desc = reshape2_0->Op();
    int dim_head =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(3);
    auto* layer_norm_bias_tensor =
        scope->FindVar(layer_norm_bias->Name())->GetMutable<phi::DenseTensor>();
    int dim_embed = layer_norm_bias_tensor->dims()[0];
    int num_head = wq_tensor->dims()[1] / dim_head;

    QKVWeightsBiasProcess(wq_tensor,
                          wk_tensor,
                          wv_tensor,
                          bq_tensor,
                          bk_tensor,
                          bv_tensor,
                          num_head,
                          dim_head,
                          dim_embed);

    if (enable_int8) {
      auto* out_linear_w_tensor = scope->FindVar(matmul_linear_w->Name())
                                      ->GetMutable<phi::DenseTensor>();
      auto* ffn0_w_tensor =
          scope->FindVar(ffn_matmul0_w->Name())->GetMutable<phi::DenseTensor>();
      auto* ffn1_w_tensor =
          scope->FindVar(ffn_matmul1_w->Name())->GetMutable<phi::DenseTensor>();

      TransposeWeights(out_linear_w_tensor);
      TransposeWeights(ffn0_w_tensor);
      TransposeWeights(ffn1_w_tensor);
    }

    // reuse the mul0_w and eltadd_0_b nodes for the combined nodes.
    auto* combined_w_desc = matmul0_w->Var();
    combined_w_desc->SetShape({3, num_head, dim_head, dim_embed});
    combined_w_desc->SetPersistable(true);

    auto* combined_bias_desc = eltadd0_b->Var();
    combined_bias_desc->SetShape({3, num_head, dim_head});
    combined_bias_desc->SetPersistable(true);

    scope->EraseVars({matmul1_w->Name(), matmul2_w->Name()});
    scope->EraseVars({eltadd1_b->Name(), eltadd2_b->Name()});

    // create fused_multi_transformer
    OpDesc fused_multi_transformer_op_desc(layer_norm->Op()->Block());
    fused_multi_transformer_op_desc.SetType(enable_int8
                                                ? "fused_multi_transformer_int8"
                                                : "fused_multi_transformer");

    // 1. Input setting
    fused_multi_transformer_op_desc.SetInput("X", {input0->Name()});

    // pre-LayerNorm input
    fused_multi_transformer_op_desc.SetInput("LnScale",
                                             {layer_norm_scale->Name()});
    fused_multi_transformer_op_desc.SetInput("LnBias",
                                             {layer_norm_bias->Name()});

    // QKV computation input
    fused_multi_transformer_op_desc.SetInput("QKVW", {matmul0_w->Name()});
    fused_multi_transformer_op_desc.SetInput("QKVBias", {eltadd0_b->Name()});
    fused_multi_transformer_op_desc.SetInput("SrcMask", {eltadd_qk_b->Name()});

    // CacheKV input
    VarDesc cache_kv_desc("cache_kv" + std::to_string(layer_idx));
    // FIXME: only support max_seq_len <= 1024
    cache_kv_desc.SetDataType(
        framework::TransToProtoVarType(bq_tensor->dtype()));
    cache_kv_desc.SetPersistable(false);
    auto* cache_kv = graph->CreateVarNode(&cache_kv_desc);

    OpDesc fill_const_op_desc(layer_norm->Op()->Block());
    fill_const_op_desc.SetType("fill_constant_batch_size_like");
    fill_const_op_desc.SetInput("Input", {input0->Name()});
    fill_const_op_desc.SetOutput("Out", {cache_kv->Name()});
    std::vector<int> shape = {2, -1, num_head, 1024, dim_head};
    fill_const_op_desc.SetAttr("shape", shape);
    fill_const_op_desc.SetAttr("input_dim_idx", 0);
    fill_const_op_desc.SetAttr("output_dim_idx", 1);
    fill_const_op_desc.SetAttr("value", 0);
    fill_const_op_desc.SetAttr(
        "dtype",
        static_cast<int>(framework::TransToProtoVarType(bq_tensor->dtype())));
    auto* fill_const_op = graph->CreateOpNode(&fill_const_op_desc);

    fused_multi_transformer_op_desc.SetInput("CacheKV", {cache_kv->Name()});

    // Out Linear input
    fused_multi_transformer_op_desc.SetInput("OutLinearW",
                                             {matmul_linear_w->Name()});
    fused_multi_transformer_op_desc.SetInput("OutLinearBias",
                                             {eltadd_linear_b->Name()});

    // Feed Forward input
    fused_multi_transformer_op_desc.SetInput("FFNLnScale",
                                             {ffn_layer_norm_scale->Name()});
    fused_multi_transformer_op_desc.SetInput("FFNLnBias",
                                             {ffn_layer_norm_bias->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN1Weight",
                                             {ffn_matmul0_w->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN1Bias",
                                             {ffn_eltadd0_b->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN2Weight",
                                             {ffn_matmul1_w->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN2Bias",
                                             {ffn_eltadd1_b->Name()});

    // 2. Output setting
    fused_multi_transformer_op_desc.SetOutput("Out", {ffn_output->Name()});
    fused_multi_transformer_op_desc.SetOutput("CacheKVOut", {cache_kv->Name()});

    // Attribute setting
    fused_multi_transformer_op_desc.SetAttr("pre_layer_norm", true);
    fused_multi_transformer_op_desc.SetAttr(
        "epsilon", layer_norm->Op()->GetAttr("epsilon"));

    fused_multi_transformer_op_desc.SetAttr("is_test", true);
    fused_multi_transformer_op_desc.SetAttr("dropout_rate", 0.0f);

    // Quantization attribute/Input
    if (enable_int8) {
      // Set input scale
      std::string qkv_input_name = matmul0_op->Input("X")[0];
      auto qkv_in_scale = PADDLE_GET_CONST(
          float, matmul0_op->GetAttr("Input_scale_" + qkv_input_name));
      std::string out_linear_input_name = matmul_linear_op->Input("X")[0];
      auto out_linear_in_scale = PADDLE_GET_CONST(
          float,
          matmul_linear_op->GetAttr("Input_scale_" + out_linear_input_name));
      std::string ffn0_input_name = ffn_matmul_0_op->Input("X")[0];
      auto ffn0_in_scale = PADDLE_GET_CONST(
          float, ffn_matmul_0_op->GetAttr("Input_scale_" + ffn0_input_name));
      std::string ffn1_input_name = ffn_matmul_1_op->Input("X")[0];
      auto ffn1_in_scale = PADDLE_GET_CONST(
          float, ffn_matmul_1_op->GetAttr("Input_scale_" + ffn1_input_name));

      // Calc outscale and Set them
      auto qkv_weight_scale =
          PADDLE_GET_CONST(float, matmul0_op->GetAttr("weight_scale"));
      auto out_weight_scale =
          PADDLE_GET_CONST(float, matmul_linear_op->GetAttr("weight_scale"));
      auto ffn0_weight_scale =
          PADDLE_GET_CONST(float, ffn_matmul_0_op->GetAttr("weight_scale"));
      auto ffn1_weight_scale =
          PADDLE_GET_CONST(float, ffn_matmul_1_op->GetAttr("weight_scale"));

      auto qkv_out_scales = std::vector<float>(
          3 * dim_embed, (qkv_weight_scale / 127.0f) * (qkv_in_scale / 127.0f));
      auto out_out_scales = std::vector<float>(
          dim_embed,
          (out_weight_scale / 127.0f) * (out_linear_in_scale / 127.0f));
      auto ffn0_out_scales = std::vector<float>(
          4 * dim_embed,
          (ffn0_weight_scale / 127.0f) * (ffn0_in_scale / 127.0f));
      auto ffn1_out_scales = std::vector<float>(
          dim_embed, (ffn1_weight_scale / 127.0f) * (ffn1_in_scale / 127.0f));

      // Inverse input scale
      qkv_in_scale = 1.0f / qkv_in_scale;
      out_linear_in_scale = 1.0f / out_linear_in_scale;
      ffn0_in_scale = 1.0f / ffn0_in_scale;
      ffn1_in_scale = 1.0f / ffn1_in_scale;

      fused_multi_transformer_op_desc.SetAttr("qkv_in_scale",
                                              std::vector<float>{qkv_in_scale});
      fused_multi_transformer_op_desc.SetAttr(
          "out_linear_in_scale", std::vector<float>{out_linear_in_scale});
      fused_multi_transformer_op_desc.SetAttr(
          "ffn1_in_scale", std::vector<float>{ffn0_in_scale});
      fused_multi_transformer_op_desc.SetAttr(
          "ffn2_in_scale", std::vector<float>{ffn1_in_scale});

      auto qkv_out_scale_var = scope->Var(matmul0_w->Name() + "_out_scale");
      auto out_out_scale_var =
          scope->Var(matmul_linear_w->Name() + "_out_scale");
      auto ffn0_out_scale_var =
          scope->Var(ffn_matmul0_w->Name() + "_out_scale");
      auto ffn1_out_scale_var =
          scope->Var(ffn_matmul1_w->Name() + "_out_scale");

      auto qkv_out_scale_data =
          qkv_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({3 * dim_embed}, platform::CPUPlace());
      memcpy(qkv_out_scale_data,
             qkv_out_scales.data(),
             qkv_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "QKVOutScale", {matmul0_w->Name() + "_out_scale"});

      auto out_out_scale_data =
          out_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({dim_embed}, platform::CPUPlace());
      memcpy(out_out_scale_data,
             out_out_scales.data(),
             out_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "OutLinearOutScale", {matmul_linear_w->Name() + "_out_scale"});

      auto ffn0_out_scale_data =
          ffn0_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({4 * dim_embed}, platform::CPUPlace());
      memcpy(ffn0_out_scale_data,
             ffn0_out_scales.data(),
             ffn0_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "FFN1OutScale", {ffn_matmul0_w->Name() + "_out_scale"});

      auto ffn1_out_scale_data =
          ffn1_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({dim_embed}, platform::CPUPlace());
      memcpy(ffn1_out_scale_data,
             ffn1_out_scales.data(),
             ffn1_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "FFN2OutScale", {ffn_matmul1_w->Name() + "_out_scale"});
    }

    auto* fused_multi_transformer =
        graph->CreateOpNode(&fused_multi_transformer_op_desc);

    if (enable_int8) {
      auto qkv_out_scale_node =
          CreatePersistableVarNode(graph, matmul0_w->Name() + "_out_scale");
      auto out_out_scale_node = CreatePersistableVarNode(
          graph, matmul_linear_w->Name() + "_out_scale");
      auto ffn0_out_scale_node =
          CreatePersistableVarNode(graph, ffn_matmul0_w->Name() + "_out_scale");
      auto ffn1_out_scale_node =
          CreatePersistableVarNode(graph, ffn_matmul1_w->Name() + "_out_scale");

      IR_NODE_LINK_TO(qkv_out_scale_node, fused_multi_transformer);
      IR_NODE_LINK_TO(out_out_scale_node, fused_multi_transformer);
      IR_NODE_LINK_TO(ffn0_out_scale_node, fused_multi_transformer);
      IR_NODE_LINK_TO(ffn1_out_scale_node, fused_multi_transformer);
    }

    IR_NODE_LINK_TO(input0, fused_multi_transformer);
    IR_NODE_LINK_TO(layer_norm_scale, fused_multi_transformer);
    IR_NODE_LINK_TO(layer_norm_bias, fused_multi_transformer);

    IR_NODE_LINK_TO(matmul0_w, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd0_b, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd_qk_b, fused_multi_transformer);

    IR_NODE_LINK_TO(input0, fill_const_op);
    IR_NODE_LINK_TO(fill_const_op, cache_kv);
    IR_NODE_LINK_TO(cache_kv, fused_multi_transformer);

    IR_NODE_LINK_TO(matmul_linear_w, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd_linear_b, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_layer_norm_scale, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_layer_norm_bias, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_matmul0_w, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_eltadd0_b, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_matmul1_w, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_eltadd1_b, fused_multi_transformer);

    IR_NODE_LINK_TO(fused_multi_transformer, ffn_output);

    // rewrite while OP input
    //  1. delete k, v
    //  2. delete matmul1/2_w eltadd1/2_w
    //  3. add cache_kv
    auto while_Xs = while0->Op()->Input("X");
    while_Xs.erase(
        std::remove(
            std::begin(while_Xs), std::end(while_Xs), transpose2_1_out->Name()),
        std::end(while_Xs));
    while_Xs.erase(
        std::remove(
            std::begin(while_Xs), std::end(while_Xs), transpose2_2_out->Name()),
        std::end(while_Xs));
    while_Xs.erase(
        std::remove(
            std::begin(while_Xs), std::end(while_Xs), matmul1_w->Name()),
        std::end(while_Xs));
    while_Xs.erase(
        std::remove(
            std::begin(while_Xs), std::end(while_Xs), matmul2_w->Name()),
        std::end(while_Xs));
    while_Xs.erase(
        std::remove(
            std::begin(while_Xs), std::end(while_Xs), eltadd1_b->Name()),
        std::end(while_Xs));
    while_Xs.erase(
        std::remove(
            std::begin(while_Xs), std::end(while_Xs), eltadd2_b->Name()),
        std::end(while_Xs));
    while_Xs.emplace_back(cache_kv->Name());
    while0->Op()->SetInput("X", while_Xs);

    // rewrite while OP output
    //  1. delete k, v
    //  2. add cache_kv
    auto while_Outs = while0->Op()->Output("Out");
    while_Outs.erase(std::remove(std::begin(while_Outs),
                                 std::end(while_Outs),
                                 transpose2_1_out->Name()),
                     std::end(while_Outs));
    while_Outs.erase(std::remove(std::begin(while_Outs),
                                 std::end(while_Outs),
                                 transpose2_2_out->Name()),
                     std::end(while_Outs));
    while_Outs.emplace_back(cache_kv->Name());
    while0->Op()->SetOutput("Out", while_Outs);

    // link CacheKV to while
    IR_NODE_LINK_TO(cache_kv, while0)
    // unlink origin KV output to while
    IR_NODE_UNLINK(transpose2_1_out, while0);
    IR_NODE_UNLINK(transpose2_2_out, while0);
    IR_NODE_UNLINK(while0, transpose2_1_out);
    IR_NODE_UNLINK(while0, transpose2_2_out);
    // unlink KV weight/bias to while after merged into Q weight/bias
    IR_NODE_UNLINK(matmul1_w, while0);
    IR_NODE_UNLINK(matmul2_w, while0);
    IR_NODE_UNLINK(eltadd1_b, while0);
    IR_NODE_UNLINK(eltadd2_b, while0);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "fused_multi_transformer_encoder pass in "
                      "op compat failed.";
      return;
    }

    VLOG(4) << "handle MultiTransformer encoder fuse";
    GET_IR_NODE_FROM_SUBGRAPH(input0, input0, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm, layer_norm, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_bias, layer_norm_bias, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_mean, layer_norm_mean, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_variance,
                              layer_norm_variance,
                              fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_out, layer_norm_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul0, matmul0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul0_out, matmul0_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul0_w, matmul0_w, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0, reshape2_0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0_out, reshape2_0_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0, transpose2_0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0_out, transpose2_0_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul1, matmul1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul1_out, matmul1_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul1_w, matmul1_w, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_1, reshape2_1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_1_out, reshape2_1_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_1, transpose2_1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_1_out, transpose2_1_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul2, matmul2, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul2_out, matmul2_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul2_w, matmul2_w, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_2, reshape2_2, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_2_out, reshape2_2_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_2, transpose2_2, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_2_out, transpose2_2_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        attention_output, attention_output, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(while0, while0, fused_multi_transformer_pattern)

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_layer_norm, ffn_layer_norm, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_scale,
                              ffn_layer_norm_scale,
                              fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_bias,
                              ffn_layer_norm_bias,
                              fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_mean,
                              ffn_layer_norm_mean,
                              fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_variance,
                              ffn_layer_norm_variance,
                              fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_out,
                              ffn_layer_norm_out,
                              fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul0, ffn_matmul0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul0_out, ffn_matmul0_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul0_w, ffn_matmul0_w, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd0, ffn_eltadd0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd0_b, ffn_eltadd0_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd0_out, ffn_eltadd0_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_gelu, ffn_gelu, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_gelu_out, ffn_gelu_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul1, ffn_matmul1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul1_out, ffn_matmul1_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul1_w, ffn_matmul1_w, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd1, ffn_eltadd1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd1_b, ffn_eltadd1_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd1_out, ffn_eltadd1_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd_out, ffn_eltadd_out, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_output, ffn_output, fused_multi_transformer_pattern)

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd0, eltadd0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd0_b, eltadd0_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd0_out, eltadd0_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd1, eltadd1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd1_b, eltadd1_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd1_out, eltadd1_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd2, eltadd2, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd2_b, eltadd2_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd2_out, eltadd2_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qk, matmul_qk, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qk_out, matmul_qk_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_qk, eltadd_qk, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_qk_b, eltadd_qk_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_qk_out, eltadd_qk_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk, softmax_qk, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk_out, softmax_qk_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv, matmul_qkv, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv_out, matmul_qkv_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv, reshape2_qkv, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv_out, reshape2_qkv_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv, transpose2_qkv, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv_out,
                              transpose2_qkv_out,
                              fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_linear, matmul_linear, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_linear_w, matmul_linear_w, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_linear_out, matmul_linear_out, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_linear, eltadd_linear, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_linear_b, eltadd_linear_b, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_linear_out, eltadd_linear_out, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_out, eltadd_out, fused_multi_transformer_pattern)

    fuse_creater(input0,
                 layer_norm,
                 layer_norm_scale,
                 layer_norm_bias,
                 layer_norm_mean,
                 layer_norm_variance,
                 matmul0,
                 matmul0_w,
                 matmul1_w,
                 matmul2_w,
                 eltadd0_b,
                 eltadd1_b,
                 eltadd2_b,
                 transpose2_1_out,
                 transpose2_2_out,
                 eltadd_qk_b,
                 reshape2_0,
                 matmul_linear,
                 matmul_linear_w,
                 eltadd_linear_b,
                 while0,
                 ffn_layer_norm,
                 ffn_layer_norm_scale,
                 ffn_layer_norm_bias,
                 ffn_layer_norm_mean,
                 ffn_layer_norm_variance,
                 ffn_matmul0,
                 ffn_matmul0_w,
                 ffn_matmul1,
                 ffn_matmul1_w,
                 ffn_eltadd0_b,
                 ffn_eltadd1_b,
                 ffn_output);

    std::unordered_set<const Node*> marked_nodes({layer_norm,
                                                  layer_norm_mean,
                                                  layer_norm_variance,
                                                  layer_norm_out,
                                                  matmul0,
                                                  matmul1,
                                                  matmul2,
                                                  matmul0_out,
                                                  matmul1_out,
                                                  matmul2_out,
                                                  eltadd0,
                                                  eltadd1,
                                                  eltadd2,
                                                  eltadd0_out,
                                                  eltadd1_out,
                                                  eltadd2_out,
                                                  reshape2_0,
                                                  reshape2_1,
                                                  reshape2_2,
                                                  reshape2_0_out,
                                                  reshape2_1_out,
                                                  reshape2_2_out,
                                                  transpose2_0,
                                                  transpose2_1,
                                                  transpose2_2,
                                                  transpose2_0_out,
                                                  transpose2_1_out,
                                                  transpose2_2_out,
                                                  matmul_qk,
                                                  matmul_qk_out,
                                                  eltadd_qk,
                                                  eltadd_qk_out,
                                                  softmax_qk,
                                                  softmax_qk_out,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_qkv,
                                                  matmul_qkv_out,
                                                  reshape2_qkv,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_linear,
                                                  matmul_linear_out,
                                                  eltadd_linear,
                                                  eltadd_linear_out,
                                                  eltadd_out,
                                                  ffn_layer_norm,
                                                  ffn_layer_norm_mean,
                                                  ffn_layer_norm_variance,
                                                  ffn_layer_norm_out,
                                                  ffn_matmul0,
                                                  ffn_matmul1,
                                                  ffn_matmul0_out,
                                                  ffn_matmul1_out,
                                                  ffn_eltadd0,
                                                  ffn_eltadd1,
                                                  ffn_eltadd0_out,
                                                  ffn_eltadd1_out,
                                                  ffn_gelu,
                                                  ffn_gelu_out,
                                                  ffn_eltadd_out});

    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void FusedMultiTransformerEncoderPass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the multi_transformer pass, The scope should not be null."));

  int fusion_count = BuildFusion(graph, name_scope_, scope);
  if (fusion_count > 0) {
    graph->Set(kFusedMultiTransformerEncoderPass, new bool(true));
    graph->Set(kFusedMultiTransformerEncoderFusionCount, new int(fusion_count));
  }
  AddStatis(fusion_count);
}

FusedMultiTransformerEncoderPass::FusedMultiTransformerEncoderPass() {
  AddOpCompat(OpCompat("layer_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddOutput("Mean")
      .IsTensor()
      .End()
      .AddOutput("Variance")
      .IsTensor()
      .End()
      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End()
      .AddAttr("begin_norm_axis")
      .IsNumGT(0)
      .End();

  AddOpCompat(OpCompat("matmul_v2"))
      .AddInput("X")  // the shape shoule be (B, S, N*H)
      .IsTensor()
      .End()
      .AddInput("Y")  // the shape shoule be (N*H, N*H)
      .IsTensor()
      .End()
      .AddOutput("Out")  // the shape shoule be (B, S, N*H)
      .IsTensor()
      .End()
      .AddAttr("trans_x")
      .IsType<bool>()
      .End()
      .AddAttr("trans_y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({2, -1, 0})
      .End();

  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Shape")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ShapeTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("shape")  // -->(B, S, H, N)  <--(B, S, N*H)
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("axis")  // {0, 2, 1, 3}
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("matmul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsNumGE(0.0f)
      .IsNumLE(1.0f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("softmax"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({-1, 3})  // shape is (B, H, S, S), so axis is -1 or 3
      .End();

  AddOpCompat(OpCompat("gelu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("approximate")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("while"))
      .AddInput("X")  // A set of variables, unconstrained
      .End()
      .AddInput("Condition")  // An scalar
      .IsTensor()
      .End()
      .AddOutput("Out")  // A set of variables, unconstrained
      .End()
      .AddOutput("StepScopes")  // A vector of local scope, unconstrained
      .End()
      .AddAttr("sub_block")
      .IsType<framework::BlockDesc*>()
      .End();
}

int FusedMultiTransformerEncoderFuseQKVPass::BuildFusion(
    Graph* graph, const std::string& name_scope, Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  bool enable_int8 = graph->Get<bool>("enable_int8");
  if (enable_int8) {
    VLOG(3) << "FusedMultiTransformerEncoderFuseQKVPass with int8";
  } else {
    VLOG(3) << "FusedMultiTransformerEncoderFuseQKVPass with fp";
  }

  // Create pattern.
  patterns::FusedMultiTransformerEncoderFuseQKVPattern
      fused_multi_transformer_fuse_qkv_pattern(pattern, name_scope);
  fused_multi_transformer_fuse_qkv_pattern();

  // Create New OpDesc
  auto fuse_creater = [&](Node* input0,
                          Node* layer_norm,
                          Node* layer_norm_scale,
                          Node* layer_norm_bias,
                          Node* layer_norm_mean,
                          Node* layer_norm_variance,
                          Node* matmul0,
                          Node* matmul0_w,
                          Node* eltadd0_b,
                          Node* split0_k_out,
                          Node* split0_v_out,
                          Node* eltadd_qk_b,
                          Node* reshape2_0,
                          Node* matmul_linear,
                          Node* matmul_linear_w,
                          Node* eltadd_linear_b,
                          Node* while0,
                          Node* ffn_layer_norm,
                          Node* ffn_layer_norm_scale,
                          Node* ffn_layer_norm_bias,
                          Node* ffn_layer_norm_mean,
                          Node* ffn_layer_norm_variance,
                          Node* ffn_matmul0,
                          Node* ffn_matmul0_w,
                          Node* ffn_matmul1,
                          Node* ffn_matmul1_w,
                          Node* ffn_eltadd0_b,
                          Node* ffn_eltadd1_b,
                          Node* ffn_output) {
    auto* matmul0_op = matmul0->Op();
    auto* matmul_linear_op = matmul_linear->Op();
    auto* ffn_matmul_0_op = ffn_matmul0->Op();
    auto* ffn_matmul_1_op = ffn_matmul1->Op();

    // Calc index of transformer layer by LayerNorm Scale name
    // This calculation assumes:
    //    1. no LayerNorm before all transformer layer
    //    2. each transformer layer contains 2 LayerNorm layer
    auto ln_scale_name = layer_norm_scale->Name();
    auto ln_name = ln_scale_name.substr(0, ln_scale_name.find('.'));
    auto ln_idx_str = ln_name.substr(ln_name.rfind('_') + 1);
    int layer_idx = atoi(ln_idx_str.c_str()) / 2;

    auto* qkv_w_tensor =
        scope->FindVar(matmul0_w->Name())->GetMutable<phi::DenseTensor>();
    auto* qkv_b_tensor =
        scope->FindVar(eltadd0_b->Name())->GetMutable<phi::DenseTensor>();

    // NOTE(minghaoBD): to make it compatible with strucutured pruning on
    // num_head dimension:
    // 1. get dim_head from reshape.shape[3], dim_embed from
    // layer_norm_bias.shape[0]
    // 2. calculate num_head according to wqkv_tensor.shape[1]/3 and dim_head
    auto reshape_desc = reshape2_0->Op();
    int dim_head =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(3) /
        3;  // 3 for qkv
    auto* layer_norm_bias_tensor =
        scope->FindVar(layer_norm_bias->Name())->GetMutable<phi::DenseTensor>();
    int dim_embed = layer_norm_bias_tensor->dims()[0];
    int num_head = qkv_w_tensor->dims()[1] / 3 / dim_head;

    QKVWeightsBiasProcessFuseQKV(
        qkv_w_tensor, qkv_b_tensor, num_head, dim_head, dim_embed);

    if (enable_int8) {
      auto* out_linear_w_tensor = scope->FindVar(matmul_linear_w->Name())
                                      ->GetMutable<phi::DenseTensor>();
      auto* ffn0_w_tensor =
          scope->FindVar(ffn_matmul0_w->Name())->GetMutable<phi::DenseTensor>();
      auto* ffn1_w_tensor =
          scope->FindVar(ffn_matmul1_w->Name())->GetMutable<phi::DenseTensor>();

      TransposeWeights(out_linear_w_tensor);
      TransposeWeights(ffn0_w_tensor);
      TransposeWeights(ffn1_w_tensor);
    }

    // create fused_multi_transformer
    OpDesc fused_multi_transformer_op_desc(layer_norm->Op()->Block());
    fused_multi_transformer_op_desc.SetType(enable_int8
                                                ? "fused_multi_transformer_int8"
                                                : "fused_multi_transformer");

    // 1. Input setting
    fused_multi_transformer_op_desc.SetInput("X", {input0->Name()});

    // pre-LayerNorm input
    fused_multi_transformer_op_desc.SetInput("LnScale",
                                             {layer_norm_scale->Name()});
    fused_multi_transformer_op_desc.SetInput("LnBias",
                                             {layer_norm_bias->Name()});

    // QKV computation input
    fused_multi_transformer_op_desc.SetInput("QKVW", {matmul0_w->Name()});
    fused_multi_transformer_op_desc.SetInput("QKVBias", {eltadd0_b->Name()});
    fused_multi_transformer_op_desc.SetInput("SrcMask", {eltadd_qk_b->Name()});

    // CacheKV input
    VarDesc cache_kv_desc("cache_kv" + std::to_string(layer_idx));
    // FIXME: only support max_seq_len <= 1024
    cache_kv_desc.SetDataType(
        framework::TransToProtoVarType(qkv_b_tensor->dtype()));
    cache_kv_desc.SetPersistable(false);
    auto* cache_kv = graph->CreateVarNode(&cache_kv_desc);

    OpDesc fill_const_op_desc(layer_norm->Op()->Block());
    fill_const_op_desc.SetType("fill_constant_batch_size_like");
    fill_const_op_desc.SetInput("Input", {input0->Name()});
    fill_const_op_desc.SetOutput("Out", {cache_kv->Name()});
    std::vector<int> shape = {2, -1, num_head, 1024, dim_head};
    fill_const_op_desc.SetAttr("shape", shape);
    fill_const_op_desc.SetAttr("input_dim_idx", 0);
    fill_const_op_desc.SetAttr("output_dim_idx", 1);
    fill_const_op_desc.SetAttr("value", 0);
    fill_const_op_desc.SetAttr("dtype",
                               static_cast<int>(framework::TransToProtoVarType(
                                   qkv_b_tensor->dtype())));
    auto* fill_const_op = graph->CreateOpNode(&fill_const_op_desc);

    fused_multi_transformer_op_desc.SetInput("CacheKV", {cache_kv->Name()});

    // Out Linear input
    fused_multi_transformer_op_desc.SetInput("OutLinearW",
                                             {matmul_linear_w->Name()});
    fused_multi_transformer_op_desc.SetInput("OutLinearBias",
                                             {eltadd_linear_b->Name()});

    // Feed Forward input
    fused_multi_transformer_op_desc.SetInput("FFNLnScale",
                                             {ffn_layer_norm_scale->Name()});
    fused_multi_transformer_op_desc.SetInput("FFNLnBias",
                                             {ffn_layer_norm_bias->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN1Weight",
                                             {ffn_matmul0_w->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN1Bias",
                                             {ffn_eltadd0_b->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN2Weight",
                                             {ffn_matmul1_w->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN2Bias",
                                             {ffn_eltadd1_b->Name()});

    // 2. Output setting
    fused_multi_transformer_op_desc.SetOutput("Out", {ffn_output->Name()});
    fused_multi_transformer_op_desc.SetOutput("CacheKVOut", {cache_kv->Name()});

    // Attribute setting
    fused_multi_transformer_op_desc.SetAttr("pre_layer_norm", true);
    fused_multi_transformer_op_desc.SetAttr(
        "epsilon", layer_norm->Op()->GetAttr("epsilon"));

    // output dropout attribute
    fused_multi_transformer_op_desc.SetAttr("is_test", true);
    fused_multi_transformer_op_desc.SetAttr("dropout_rate", 0.0f);

    // Quantization attribute/Input
    if (enable_int8) {
      // Set input scale
      std::string qkv_input_name = matmul0_op->Input("X")[0];
      auto qkv_in_scale = PADDLE_GET_CONST(
          float, matmul0_op->GetAttr("Input_scale_" + qkv_input_name));
      std::string out_linear_input_name = matmul_linear_op->Input("X")[0];
      auto out_linear_in_scale = PADDLE_GET_CONST(
          float,
          matmul_linear_op->GetAttr("Input_scale_" + out_linear_input_name));
      std::string ffn0_input_name = ffn_matmul_0_op->Input("X")[0];
      auto ffn0_in_scale = PADDLE_GET_CONST(
          float, ffn_matmul_0_op->GetAttr("Input_scale_" + ffn0_input_name));
      std::string ffn1_input_name = ffn_matmul_1_op->Input("X")[0];
      auto ffn1_in_scale = PADDLE_GET_CONST(
          float, ffn_matmul_1_op->GetAttr("Input_scale_" + ffn1_input_name));

      // Calc outscale and Set them
      // TODO(wufeisheng): Currently just match layer-wise weight scale, where
      // channel-wise weight scale should also be surpported.
      auto qkv_weight_scale =
          PADDLE_GET_CONST(float, matmul0_op->GetAttr("weight_scale"));
      auto out_weight_scale =
          PADDLE_GET_CONST(float, matmul_linear_op->GetAttr("weight_scale"));
      auto ffn0_weight_scale =
          PADDLE_GET_CONST(float, ffn_matmul_0_op->GetAttr("weight_scale"));
      auto ffn1_weight_scale =
          PADDLE_GET_CONST(float, ffn_matmul_1_op->GetAttr("weight_scale"));

      auto qkv_out_scales = std::vector<float>(
          3 * dim_embed, (qkv_weight_scale / 127.0f) * (qkv_in_scale / 127.0f));
      auto out_out_scales = std::vector<float>(
          dim_embed,
          (out_weight_scale / 127.0f) * (out_linear_in_scale / 127.0f));
      auto ffn0_out_scales = std::vector<float>(
          4 * dim_embed,
          (ffn0_weight_scale / 127.0f) * (ffn0_in_scale / 127.0f));
      auto ffn1_out_scales = std::vector<float>(
          dim_embed, (ffn1_weight_scale / 127.0f) * (ffn1_in_scale / 127.0f));

      // Inverse input scale
      qkv_in_scale = 1.0f / qkv_in_scale;
      out_linear_in_scale = 1.0f / out_linear_in_scale;
      ffn0_in_scale = 1.0f / ffn0_in_scale;
      ffn1_in_scale = 1.0f / ffn1_in_scale;

      fused_multi_transformer_op_desc.SetAttr("qkv_in_scale",
                                              std::vector<float>{qkv_in_scale});
      fused_multi_transformer_op_desc.SetAttr(
          "out_linear_in_scale", std::vector<float>{out_linear_in_scale});
      fused_multi_transformer_op_desc.SetAttr(
          "ffn1_in_scale", std::vector<float>{ffn0_in_scale});
      fused_multi_transformer_op_desc.SetAttr(
          "ffn2_in_scale", std::vector<float>{ffn1_in_scale});

      auto qkv_out_scale_var = scope->Var(matmul0_w->Name() + "_out_scale");
      auto out_out_scale_var =
          scope->Var(matmul_linear_w->Name() + "_out_scale");
      auto ffn0_out_scale_var =
          scope->Var(ffn_matmul0_w->Name() + "_out_scale");
      auto ffn1_out_scale_var =
          scope->Var(ffn_matmul1_w->Name() + "_out_scale");

      auto qkv_out_scale_data =
          qkv_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({3 * dim_embed}, platform::CPUPlace());
      memcpy(qkv_out_scale_data,
             qkv_out_scales.data(),
             qkv_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "QKVOutScale", {matmul0_w->Name() + "_out_scale"});

      auto out_out_scale_data =
          out_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({dim_embed}, platform::CPUPlace());
      memcpy(out_out_scale_data,
             out_out_scales.data(),
             out_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "OutLinearOutScale", {matmul_linear_w->Name() + "_out_scale"});

      auto ffn0_out_scale_data =
          ffn0_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({4 * dim_embed}, platform::CPUPlace());
      memcpy(ffn0_out_scale_data,
             ffn0_out_scales.data(),
             ffn0_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "FFN1OutScale", {ffn_matmul0_w->Name() + "_out_scale"});

      auto ffn1_out_scale_data =
          ffn1_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({dim_embed}, platform::CPUPlace());
      memcpy(ffn1_out_scale_data,
             ffn1_out_scales.data(),
             ffn1_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "FFN2OutScale", {ffn_matmul1_w->Name() + "_out_scale"});
    }

    auto* fused_multi_transformer =
        graph->CreateOpNode(&fused_multi_transformer_op_desc);

    if (enable_int8) {
      auto qkv_out_scale_node =
          CreatePersistableVarNode(graph, matmul0_w->Name() + "_out_scale");
      auto out_out_scale_node = CreatePersistableVarNode(
          graph, matmul_linear_w->Name() + "_out_scale");
      auto ffn0_out_scale_node =
          CreatePersistableVarNode(graph, ffn_matmul0_w->Name() + "_out_scale");
      auto ffn1_out_scale_node =
          CreatePersistableVarNode(graph, ffn_matmul1_w->Name() + "_out_scale");

      IR_NODE_LINK_TO(qkv_out_scale_node, fused_multi_transformer);
      IR_NODE_LINK_TO(out_out_scale_node, fused_multi_transformer);
      IR_NODE_LINK_TO(ffn0_out_scale_node, fused_multi_transformer);
      IR_NODE_LINK_TO(ffn1_out_scale_node, fused_multi_transformer);
    }

    IR_NODE_LINK_TO(input0, fused_multi_transformer);
    IR_NODE_LINK_TO(layer_norm_scale, fused_multi_transformer);
    IR_NODE_LINK_TO(layer_norm_bias, fused_multi_transformer);

    IR_NODE_LINK_TO(matmul0_w, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd0_b, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd_qk_b, fused_multi_transformer);

    IR_NODE_LINK_TO(input0, fill_const_op);
    IR_NODE_LINK_TO(fill_const_op, cache_kv);
    IR_NODE_LINK_TO(cache_kv, fused_multi_transformer);

    IR_NODE_LINK_TO(matmul_linear_w, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd_linear_b, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_layer_norm_scale, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_layer_norm_bias, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_matmul0_w, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_eltadd0_b, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_matmul1_w, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_eltadd1_b, fused_multi_transformer);

    IR_NODE_LINK_TO(fused_multi_transformer, ffn_output);

    // rewrite while OP input
    //  1. delete k, v
    //  2. delete matmul1/2_w eltadd1/2_w
    //  3. add cache_kv
    auto while_Xs = while0->Op()->Input("X");
    while_Xs.erase(
        std::remove(
            std::begin(while_Xs), std::end(while_Xs), split0_k_out->Name()),
        std::end(while_Xs));
    while_Xs.erase(
        std::remove(
            std::begin(while_Xs), std::end(while_Xs), split0_v_out->Name()),
        std::end(while_Xs));
    while_Xs.emplace_back(cache_kv->Name());
    while0->Op()->SetInput("X", while_Xs);

    // rewrite while OP output
    //  1. delete k, v
    //  2. add cache_kv
    auto while_Outs = while0->Op()->Output("Out");
    while_Outs.erase(
        std::remove(
            std::begin(while_Outs), std::end(while_Outs), split0_k_out->Name()),
        std::end(while_Outs));
    while_Outs.erase(
        std::remove(
            std::begin(while_Outs), std::end(while_Outs), split0_v_out->Name()),
        std::end(while_Outs));
    while_Outs.emplace_back(cache_kv->Name());
    while0->Op()->SetOutput("Out", while_Outs);

    // link CacheKV to while
    IR_NODE_LINK_TO(cache_kv, while0)
    // unlink origin KV output to while
    IR_NODE_UNLINK(split0_k_out, while0);
    IR_NODE_UNLINK(split0_v_out, while0);
    IR_NODE_UNLINK(while0, split0_k_out);
    IR_NODE_UNLINK(while0, split0_v_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "fused_multi_transformer_encoder_fuse_qkv "
                      "pass in op compat failed.";
      return;
    }

    VLOG(4) << "handle MultiTransformer encoder(Fuse-QKV) fuse";
    GET_IR_NODE_FROM_SUBGRAPH(
        input0, input0, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm, layer_norm, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_scale,
                              layer_norm_scale,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias,
                              layer_norm_bias,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_mean,
                              layer_norm_mean,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_variance,
                              layer_norm_variance,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out,
                              layer_norm_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul0, matmul0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul0_out, matmul0_out, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul0_w, matmul0_w, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0, reshape2_0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0_out,
                              reshape2_0_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0, transpose2_0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0_out,
                              transpose2_0_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        split0, split0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        split0_q_out, split0_q_out, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        split0_k_out, split0_k_out, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        split0_v_out, split0_v_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm,
                              ffn_layer_norm,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_scale,
                              ffn_layer_norm_scale,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_bias,
                              ffn_layer_norm_bias,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_mean,
                              ffn_layer_norm_mean,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_variance,
                              ffn_layer_norm_variance,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_out,
                              ffn_layer_norm_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul0, ffn_matmul0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_matmul0_out,
                              ffn_matmul0_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul0_w, ffn_matmul0_w, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd0, ffn_eltadd0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd0_b, ffn_eltadd0_b, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd0_out,
                              ffn_eltadd0_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_gelu, ffn_gelu, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_gelu_out, ffn_gelu_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul1, ffn_matmul1, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_matmul1_out,
                              ffn_matmul1_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul1_w, ffn_matmul1_w, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd1, ffn_eltadd1, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd1_b, ffn_eltadd1_b, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd1_out,
                              ffn_eltadd1_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd_out,
                              ffn_eltadd_out,
                              fused_multi_transformer_fuse_qkv_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_output, ffn_output, fused_multi_transformer_fuse_qkv_pattern)

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd0, eltadd0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd0_b, eltadd0_b, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd0_out, eltadd0_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qk, matmul_qk, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qk_out, matmul_qk_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        scale_qk, scale_qk, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        scale_qk_out, scale_qk_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_qk, eltadd_qk, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_qk_b, eltadd_qk_b, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_qk_out, eltadd_qk_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk, softmax_qk, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk_out,
                              softmax_qk_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv, matmul_qkv, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv_out,
                              matmul_qkv_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv, reshape2_qkv, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv_out,
                              reshape2_qkv_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv,
                              transpose2_qkv,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv_out,
                              transpose2_qkv_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_linear, matmul_linear, fused_multi_transformer_fuse_qkv_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(matmul_linear_w,
                              matmul_linear_w,
                              fused_multi_transformer_fuse_qkv_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(matmul_linear_out,
                              matmul_linear_out,
                              fused_multi_transformer_fuse_qkv_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_linear, eltadd_linear, fused_multi_transformer_fuse_qkv_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_linear_b,
                              eltadd_linear_b,
                              fused_multi_transformer_fuse_qkv_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_linear_out,
                              eltadd_linear_out,
                              fused_multi_transformer_fuse_qkv_pattern)

    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_out, eltadd_out, fused_multi_transformer_fuse_qkv_pattern)

    GET_IR_NODE_FROM_SUBGRAPH(
        while0, while0, fused_multi_transformer_fuse_qkv_pattern)

    fuse_creater(input0,
                 layer_norm,
                 layer_norm_scale,
                 layer_norm_bias,
                 layer_norm_mean,
                 layer_norm_variance,
                 matmul0,
                 matmul0_w,
                 eltadd0_b,
                 split0_k_out,
                 split0_v_out,
                 eltadd_qk_b,
                 reshape2_0,
                 matmul_linear,
                 matmul_linear_w,
                 eltadd_linear_b,
                 while0,
                 ffn_layer_norm,
                 ffn_layer_norm_scale,
                 ffn_layer_norm_bias,
                 ffn_layer_norm_mean,
                 ffn_layer_norm_variance,
                 ffn_matmul0,
                 ffn_matmul0_w,
                 ffn_matmul1,
                 ffn_matmul1_w,
                 ffn_eltadd0_b,
                 ffn_eltadd1_b,
                 ffn_output);

    std::unordered_set<const Node*> marked_nodes({layer_norm,
                                                  layer_norm_mean,
                                                  layer_norm_variance,
                                                  layer_norm_out,
                                                  matmul0,
                                                  matmul0_out,
                                                  eltadd0,
                                                  eltadd0_out,
                                                  reshape2_0,
                                                  reshape2_0_out,
                                                  transpose2_0,
                                                  transpose2_0_out,
                                                  split0,
                                                  split0_q_out,
                                                  split0_k_out,
                                                  split0_v_out,
                                                  matmul_qk,
                                                  matmul_qk_out,
                                                  scale_qk,
                                                  scale_qk_out,
                                                  eltadd_qk,
                                                  eltadd_qk_out,
                                                  softmax_qk,
                                                  softmax_qk_out,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_qkv,
                                                  matmul_qkv_out,
                                                  reshape2_qkv,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_linear,
                                                  matmul_linear_out,
                                                  eltadd_linear,
                                                  eltadd_linear_out,
                                                  eltadd_out,
                                                  ffn_layer_norm,
                                                  ffn_layer_norm_mean,
                                                  ffn_layer_norm_variance,
                                                  ffn_layer_norm_out,
                                                  ffn_matmul0,
                                                  ffn_matmul1,
                                                  ffn_matmul0_out,
                                                  ffn_matmul1_out,
                                                  ffn_eltadd0,
                                                  ffn_eltadd1,
                                                  ffn_eltadd0_out,
                                                  ffn_eltadd1_out,
                                                  ffn_gelu,
                                                  ffn_gelu_out,
                                                  ffn_eltadd_out});

    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void FusedMultiTransformerEncoderFuseQKVPass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the fused_multi_transformer_encoder pass, "
          "The scope should not be null."));

  int fusion_count = BuildFusion(graph, name_scope_, scope);
  if (fusion_count > 0) {
    graph->Set(kFusedMultiTransformerEncoderFuseQKVPass, new bool(true));
    graph->Set(kFusedMultiTransformerEncoderFusionCount, new int(fusion_count));
  }
  AddStatis(fusion_count);
}

FusedMultiTransformerEncoderFuseQKVPass::
    FusedMultiTransformerEncoderFuseQKVPass() {
  AddOpCompat(OpCompat("layer_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddOutput("Mean")
      .IsTensor()
      .End()
      .AddOutput("Variance")
      .IsTensor()
      .End()
      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End()
      .AddAttr("begin_norm_axis")
      .IsNumGT(0)
      .End();

  AddOpCompat(OpCompat("scale"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("scale")
      .IsType<float>()  // copy to new op. so unconstrained.
      .End()
      .AddAttr("bias")
      .IsNumEQ(0.f)
      .End()
      .AddAttr("bias_after_scale")  // bias is 0, so unconstrained.
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("matmul_v2"))
      .AddInput("X")  // the shape shoule be (B, S, N*H)
      .IsTensor()
      .End()
      .AddInput("Y")  // the shape shoule be (N*H, N*H)
      .IsTensor()
      .End()
      .AddOutput("Out")  // the shape shoule be (B, S, N*H)
      .IsTensor()
      .End()
      .AddAttr("trans_x")
      .IsType<bool>()
      .End()
      .AddAttr("trans_y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({2, -1, 0})
      .End();

  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Shape")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ShapeTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("shape")  // -->(B, S, H, N)  <--(B, S, N*H)
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("axis")  // {0, 2, 1, 3}
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("matmul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsNumGE(0.0f)
      .IsNumLE(1.0f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("softmax"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({-1, 3})  // shape is (B, H, S, S), so axis is -1 or 3
      .End();

  AddOpCompat(OpCompat("gelu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("approximate")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("while"))
      .AddInput("X")  // A set of variables, unconstrained
      .End()
      .AddInput("Condition")  // An scalar
      .IsTensor()
      .End()
      .AddOutput("Out")  // A set of variables, unconstrained
      .End()
      .AddOutput("StepScopes")  // A vector of local scope, unconstrained
      .End()
      .AddAttr("sub_block")
      .IsType<framework::BlockDesc*>()
      .End();
}

int MultiDevicesFusedMultiTransformerEncoderFuseQKVPass::BuildFusion(
    Graph* graph, const std::string& name_scope, Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  bool enable_int8 = graph->Get<bool>("enable_int8");
  if (enable_int8) {
    VLOG(3) << "MultiDevicesFusedMultiTransformerEncoderFuseQKVPass with int8";
  } else {
    VLOG(3) << "MultiDevicesFusedMultiTransformerEncoderFuseQKVPass with fp";
  }

  // Create pattern.
  patterns::MultiDevicesFusedMultiTransformerEncoderFuseQKVPattern
      fused_multi_transformer_fuse_qkv_pattern(pattern, name_scope);
  fused_multi_transformer_fuse_qkv_pattern();

  // Create New OpDesc
  auto fuse_creater = [&](Node* input0,
                          Node* layer_norm,
                          Node* layer_norm_scale,
                          Node* layer_norm_bias,
                          Node* layer_norm_mean,
                          Node* layer_norm_variance,
                          Node* c_identity,
                          Node* matmul0,
                          Node* matmul0_w,
                          Node* eltadd0_b,
                          Node* split0_k_out,
                          Node* split0_v_out,
                          Node* eltadd_qk_b,
                          Node* reshape2_0,
                          Node* matmul_linear,
                          Node* matmul_linear_w,
                          Node* eltadd_linear_b,
                          Node* while0,
                          Node* ffn_layer_norm,
                          Node* ffn_layer_norm_scale,
                          Node* ffn_layer_norm_bias,
                          Node* ffn_layer_norm_mean,
                          Node* ffn_layer_norm_variance,
                          Node* ffn_c_identity,
                          Node* ffn_matmul0,
                          Node* ffn_matmul0_w,
                          Node* ffn_matmul1,
                          Node* ffn_matmul1_w,
                          Node* ffn_eltadd0_b,
                          Node* ffn_eltadd1_b,
                          Node* ffn_output) {
    auto* matmul0_op = matmul0->Op();
    auto* matmul_linear_op = matmul_linear->Op();
    auto* ffn_matmul_0_op = ffn_matmul0->Op();
    auto* ffn_matmul_1_op = ffn_matmul1->Op();

    // Calc index of transformer layer by LayerNorm Scale name
    // This calculation assumes:
    //    1. no LayerNorm before all transformer layer
    //    2. each transformer layer contains 2 LayerNorm layer
    auto ln_scale_name = layer_norm_scale->Name();
    auto ln_name = ln_scale_name.substr(0, ln_scale_name.find('.'));
    auto ln_idx_str = ln_name.substr(ln_name.rfind('_') + 1);
    int layer_idx = atoi(ln_idx_str.c_str()) / 2;

    auto* qkv_w_tensor =
        scope->FindVar(matmul0_w->Name())->GetMutable<phi::DenseTensor>();
    auto* qkv_b_tensor =
        scope->FindVar(eltadd0_b->Name())->GetMutable<phi::DenseTensor>();

    // NOTE(minghaoBD): to make it compatible with strucutured pruning on
    // num_head dimension:
    // 1. get dim_head from reshape.shape[3], dim_embed from
    // layer_norm_bias.shape[0]
    // 2. calculate num_head according to wqkv_tensor.shape[1]/3 and dim_head
    auto* layer_norm_bias_tensor =
        scope->FindVar(layer_norm_bias->Name())->GetMutable<phi::DenseTensor>();
    int dim_embed = layer_norm_bias_tensor->dims()[0];
    auto reshape_desc = reshape2_0->Op();
    int dim_head =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(3) /
        3;  // 3 for qkv
    int num_head = qkv_w_tensor->dims()[1] / 3 / dim_head;

    QKVWeightsBiasProcessFuseQKV(
        qkv_w_tensor, qkv_b_tensor, num_head, dim_head, dim_embed);

    if (enable_int8) {
      auto* out_linear_w_tensor = scope->FindVar(matmul_linear_w->Name())
                                      ->GetMutable<phi::DenseTensor>();
      auto* ffn0_w_tensor =
          scope->FindVar(ffn_matmul0_w->Name())->GetMutable<phi::DenseTensor>();
      auto* ffn1_w_tensor =
          scope->FindVar(ffn_matmul1_w->Name())->GetMutable<phi::DenseTensor>();

      TransposeWeights(out_linear_w_tensor);
      TransposeWeights(ffn0_w_tensor);
      TransposeWeights(ffn1_w_tensor);
    }

    // create fused_multi_transformer
    OpDesc fused_multi_transformer_op_desc(layer_norm->Op()->Block());
    fused_multi_transformer_op_desc.SetType(enable_int8
                                                ? "fused_multi_transformer_int8"
                                                : "fused_multi_transformer");

    // 1. Input setting
    fused_multi_transformer_op_desc.SetInput("X", {input0->Name()});

    // pre-LayerNorm input
    fused_multi_transformer_op_desc.SetInput("LnScale",
                                             {layer_norm_scale->Name()});
    fused_multi_transformer_op_desc.SetInput("LnBias",
                                             {layer_norm_bias->Name()});

    // QKV computation input
    fused_multi_transformer_op_desc.SetInput("QKVW", {matmul0_w->Name()});
    fused_multi_transformer_op_desc.SetInput("QKVBias", {eltadd0_b->Name()});
    fused_multi_transformer_op_desc.SetInput("SrcMask", {eltadd_qk_b->Name()});

    // CacheKV input
    VarDesc cache_kv_desc("cache_kv" + std::to_string(layer_idx));
    // FIXME: only support max_seq_len <= 1024
    cache_kv_desc.SetDataType(
        framework::TransToProtoVarType(qkv_b_tensor->dtype()));
    cache_kv_desc.SetPersistable(false);
    auto* cache_kv = graph->CreateVarNode(&cache_kv_desc);

    OpDesc fill_const_op_desc(layer_norm->Op()->Block());
    fill_const_op_desc.SetType("fill_constant_batch_size_like");
    fill_const_op_desc.SetInput("Input", {input0->Name()});
    fill_const_op_desc.SetOutput("Out", {cache_kv->Name()});
    std::vector<int> shape = {2, -1, num_head, 1024, dim_head};
    fill_const_op_desc.SetAttr("shape", shape);
    fill_const_op_desc.SetAttr("input_dim_idx", 0);
    fill_const_op_desc.SetAttr("output_dim_idx", 1);
    fill_const_op_desc.SetAttr("value", 0);
    fill_const_op_desc.SetAttr("dtype",
                               static_cast<int>(framework::TransToProtoVarType(
                                   qkv_b_tensor->dtype())));
    auto* fill_const_op = graph->CreateOpNode(&fill_const_op_desc);

    fused_multi_transformer_op_desc.SetInput("CacheKV", {cache_kv->Name()});

    // Out Linear input
    fused_multi_transformer_op_desc.SetInput("OutLinearW",
                                             {matmul_linear_w->Name()});
    fused_multi_transformer_op_desc.SetInput("OutLinearBias",
                                             {eltadd_linear_b->Name()});

    // Feed Forward input
    fused_multi_transformer_op_desc.SetInput("FFNLnScale",
                                             {ffn_layer_norm_scale->Name()});
    fused_multi_transformer_op_desc.SetInput("FFNLnBias",
                                             {ffn_layer_norm_bias->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN1Weight",
                                             {ffn_matmul0_w->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN1Bias",
                                             {ffn_eltadd0_b->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN2Weight",
                                             {ffn_matmul1_w->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN2Bias",
                                             {ffn_eltadd1_b->Name()});

    // 2. Output setting
    fused_multi_transformer_op_desc.SetOutput("Out", {ffn_output->Name()});
    fused_multi_transformer_op_desc.SetOutput("CacheKVOut", {cache_kv->Name()});

    // Attribute setting
    fused_multi_transformer_op_desc.SetAttr("pre_layer_norm", true);
    fused_multi_transformer_op_desc.SetAttr(
        "epsilon", layer_norm->Op()->GetAttr("epsilon"));

    // output dropout attribute
    fused_multi_transformer_op_desc.SetAttr("dropout_rate", 0.0f);
    fused_multi_transformer_op_desc.SetAttr("is_test", true);

    // parallel ring id
    auto* c_identity_op = c_identity->Op();
    fused_multi_transformer_op_desc.SetAttr("ring_id",
                                            c_identity_op->GetAttr("ring_id"));

    // Quantization attribute/Input
    if (enable_int8) {
      // Set input scale
      std::string matmul_input_scale_suffix = c_identity_op->Input("X")[0];
      auto qkv_in_scale = PADDLE_GET_CONST(
          float,
          c_identity_op->GetAttr("Input_scale_" + matmul_input_scale_suffix));

      std::string out_linear_input_name = matmul_linear_op->Input("X")[0];
      auto out_linear_in_scale = PADDLE_GET_CONST(
          float,
          matmul_linear_op->GetAttr("Input_scale_" + out_linear_input_name));

      auto* ffn_c_identity_op = ffn_c_identity->Op();
      std::string ffn_input_scale_suffix = ffn_c_identity_op->Input("X")[0];
      auto ffn0_in_scale = PADDLE_GET_CONST(
          float,
          ffn_c_identity_op->GetAttr("Input_scale_" + ffn_input_scale_suffix));

      std::string ffn1_input_name = ffn_matmul_1_op->Input("X")[0];
      auto ffn1_in_scale = PADDLE_GET_CONST(
          float, ffn_matmul_1_op->GetAttr("Input_scale_" + ffn1_input_name));

      // Calc outscale and Set them
      auto qkv_weight_scale =
          PADDLE_GET_CONST(float, matmul0_op->GetAttr("weight_scale"));
      auto out_weight_scale =
          PADDLE_GET_CONST(float, matmul_linear_op->GetAttr("weight_scale"));
      auto ffn0_weight_scale =
          PADDLE_GET_CONST(float, ffn_matmul_0_op->GetAttr("weight_scale"));
      auto ffn1_weight_scale =
          PADDLE_GET_CONST(float, ffn_matmul_1_op->GetAttr("weight_scale"));

      auto qkv_out_scales = std::vector<float>(
          3 * dim_embed, (qkv_weight_scale / 127.0f) * (qkv_in_scale / 127.0f));
      auto out_out_scales = std::vector<float>(
          dim_embed,
          (out_weight_scale / 127.0f) * (out_linear_in_scale / 127.0f));
      auto ffn0_out_scales = std::vector<float>(
          4 * dim_embed,
          (ffn0_weight_scale / 127.0f) * (ffn0_in_scale / 127.0f));
      auto ffn1_out_scales = std::vector<float>(
          dim_embed, (ffn1_weight_scale / 127.0f) * (ffn1_in_scale / 127.0f));

      // Inverse input scale
      qkv_in_scale = 1.0f / qkv_in_scale;
      out_linear_in_scale = 1.0f / out_linear_in_scale;
      ffn0_in_scale = 1.0f / ffn0_in_scale;
      ffn1_in_scale = 1.0f / ffn1_in_scale;

      fused_multi_transformer_op_desc.SetAttr("qkv_in_scale",
                                              std::vector<float>{qkv_in_scale});
      fused_multi_transformer_op_desc.SetAttr(
          "out_linear_in_scale", std::vector<float>{out_linear_in_scale});
      fused_multi_transformer_op_desc.SetAttr(
          "ffn1_in_scale", std::vector<float>{ffn0_in_scale});
      fused_multi_transformer_op_desc.SetAttr(
          "ffn2_in_scale", std::vector<float>{ffn1_in_scale});

      auto qkv_out_scale_var = scope->Var(matmul0_w->Name() + "_out_scale");
      auto out_out_scale_var =
          scope->Var(matmul_linear_w->Name() + "_out_scale");
      auto ffn0_out_scale_var =
          scope->Var(ffn_matmul0_w->Name() + "_out_scale");
      auto ffn1_out_scale_var =
          scope->Var(ffn_matmul1_w->Name() + "_out_scale");

      auto qkv_out_scale_data =
          qkv_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({3 * dim_embed}, platform::CPUPlace());
      memcpy(qkv_out_scale_data,
             qkv_out_scales.data(),
             qkv_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "QKVOutScale", {matmul0_w->Name() + "_out_scale"});

      auto out_out_scale_data =
          out_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({dim_embed}, platform::CPUPlace());
      memcpy(out_out_scale_data,
             out_out_scales.data(),
             out_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "OutLinearOutScale", {matmul_linear_w->Name() + "_out_scale"});

      auto ffn0_out_scale_data =
          ffn0_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({4 * dim_embed}, platform::CPUPlace());
      memcpy(ffn0_out_scale_data,
             ffn0_out_scales.data(),
             ffn0_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "FFN1OutScale", {ffn_matmul0_w->Name() + "_out_scale"});

      auto ffn1_out_scale_data =
          ffn1_out_scale_var->GetMutable<phi::DenseTensor>()
              ->mutable_data<float>({dim_embed}, platform::CPUPlace());
      memcpy(ffn1_out_scale_data,
             ffn1_out_scales.data(),
             ffn1_out_scales.size() * sizeof(float));
      fused_multi_transformer_op_desc.SetInput(
          "FFN2OutScale", {ffn_matmul1_w->Name() + "_out_scale"});
    }

    auto* fused_multi_transformer =
        graph->CreateOpNode(&fused_multi_transformer_op_desc);

    if (enable_int8) {
      auto qkv_out_scale_node =
          CreatePersistableVarNode(graph, matmul0_w->Name() + "_out_scale");
      auto out_out_scale_node = CreatePersistableVarNode(
          graph, matmul_linear_w->Name() + "_out_scale");
      auto ffn0_out_scale_node =
          CreatePersistableVarNode(graph, ffn_matmul0_w->Name() + "_out_scale");
      auto ffn1_out_scale_node =
          CreatePersistableVarNode(graph, ffn_matmul1_w->Name() + "_out_scale");

      IR_NODE_LINK_TO(qkv_out_scale_node, fused_multi_transformer);
      IR_NODE_LINK_TO(out_out_scale_node, fused_multi_transformer);
      IR_NODE_LINK_TO(ffn0_out_scale_node, fused_multi_transformer);
      IR_NODE_LINK_TO(ffn1_out_scale_node, fused_multi_transformer);
    }

    IR_NODE_LINK_TO(input0, fused_multi_transformer);
    IR_NODE_LINK_TO(layer_norm_scale, fused_multi_transformer);
    IR_NODE_LINK_TO(layer_norm_bias, fused_multi_transformer);

    IR_NODE_LINK_TO(matmul0_w, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd0_b, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd_qk_b, fused_multi_transformer);

    IR_NODE_LINK_TO(input0, fill_const_op);
    IR_NODE_LINK_TO(fill_const_op, cache_kv);
    IR_NODE_LINK_TO(cache_kv, fused_multi_transformer);

    IR_NODE_LINK_TO(matmul_linear_w, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd_linear_b, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_layer_norm_scale, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_layer_norm_bias, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_matmul0_w, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_eltadd0_b, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_matmul1_w, fused_multi_transformer);
    IR_NODE_LINK_TO(ffn_eltadd1_b, fused_multi_transformer);

    IR_NODE_LINK_TO(fused_multi_transformer, ffn_output);

    // rewrite while OP input
    //  1. delete k, v
    //  2. delete matmul1/2_w eltadd1/2_w
    //  3. add cache_kv
    auto while_Xs = while0->Op()->Input("X");
    while_Xs.erase(
        std::remove(
            std::begin(while_Xs), std::end(while_Xs), split0_k_out->Name()),
        std::end(while_Xs));
    while_Xs.erase(
        std::remove(
            std::begin(while_Xs), std::end(while_Xs), split0_v_out->Name()),
        std::end(while_Xs));
    while_Xs.emplace_back(cache_kv->Name());
    while0->Op()->SetInput("X", while_Xs);

    // rewrite while OP output
    //  1. delete k, v
    //  2. add cache_kv
    auto while_Outs = while0->Op()->Output("Out");
    while_Outs.erase(
        std::remove(
            std::begin(while_Outs), std::end(while_Outs), split0_k_out->Name()),
        std::end(while_Outs));
    while_Outs.erase(
        std::remove(
            std::begin(while_Outs), std::end(while_Outs), split0_v_out->Name()),
        std::end(while_Outs));
    while_Outs.emplace_back(cache_kv->Name());
    while0->Op()->SetOutput("Out", while_Outs);

    // link CacheKV to while
    IR_NODE_LINK_TO(cache_kv, while0)
    // unlink origin KV output to while
    IR_NODE_UNLINK(split0_k_out, while0);
    IR_NODE_UNLINK(split0_v_out, while0);
    IR_NODE_UNLINK(while0, split0_k_out);
    IR_NODE_UNLINK(while0, split0_v_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "fused_multi_transformer_encoder_fuse_qkv "
                      "pass in op compat failed.";
      return;
    }

    VLOG(4) << "handle MultiTransformer encoder(Fuse-QKV) fuse";
    GET_IR_NODE_FROM_SUBGRAPH(
        input0, input0, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm, layer_norm, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_scale,
                              layer_norm_scale,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias,
                              layer_norm_bias,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_mean,
                              layer_norm_mean,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_variance,
                              layer_norm_variance,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out,
                              layer_norm_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        c_identity, c_identity, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(c_identity_out,
                              c_identity_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul0, matmul0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul0_out, matmul0_out, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul0_w, matmul0_w, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0, reshape2_0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0_out,
                              reshape2_0_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0, transpose2_0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0_out,
                              transpose2_0_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        split0, split0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        split0_q_out, split0_q_out, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        split0_k_out, split0_k_out, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        split0_v_out, split0_v_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm,
                              ffn_layer_norm,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_scale,
                              ffn_layer_norm_scale,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_bias,
                              ffn_layer_norm_bias,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_mean,
                              ffn_layer_norm_mean,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_variance,
                              ffn_layer_norm_variance,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_out,
                              ffn_layer_norm_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(ffn_c_identity,
                              ffn_c_identity,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_c_identity_out,
                              ffn_c_identity_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul0, ffn_matmul0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_matmul0_out,
                              ffn_matmul0_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul0_w, ffn_matmul0_w, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd0, ffn_eltadd0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd0_b, ffn_eltadd0_b, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd0_out,
                              ffn_eltadd0_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_gelu, ffn_gelu, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_gelu_out, ffn_gelu_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul1, ffn_matmul1, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_matmul1_out,
                              ffn_matmul1_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_matmul1_w, ffn_matmul1_w, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_c_allreduce_sum,
                              ffn_c_allreduce_sum,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_c_allreduce_sum_out,
                              ffn_c_allreduce_sum_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd1, ffn_eltadd1, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_eltadd1_b, ffn_eltadd1_b, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd1_out,
                              ffn_eltadd1_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd_out,
                              ffn_eltadd_out,
                              fused_multi_transformer_fuse_qkv_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(
        ffn_output, ffn_output, fused_multi_transformer_fuse_qkv_pattern)

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd0, eltadd0, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd0_b, eltadd0_b, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd0_out, eltadd0_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qk, matmul_qk, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qk_out, matmul_qk_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        scale_qk, scale_qk, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        scale_qk_out, scale_qk_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_qk, eltadd_qk, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_qk_b, eltadd_qk_b, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_qk_out, eltadd_qk_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk, softmax_qk, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk_out,
                              softmax_qk_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv, matmul_qkv, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv_out,
                              matmul_qkv_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv, reshape2_qkv, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv_out,
                              reshape2_qkv_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv,
                              transpose2_qkv,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv_out,
                              transpose2_qkv_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_linear, matmul_linear, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_linear_w,
                              matmul_linear_w,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_linear_out,
                              matmul_linear_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(c_allreduce_sum,
                              c_allreduce_sum,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(c_allreduce_sum_out,
                              c_allreduce_sum_out,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_linear, eltadd_linear, fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_linear_b,
                              eltadd_linear_b,
                              fused_multi_transformer_fuse_qkv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_linear_out,
                              eltadd_linear_out,
                              fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        eltadd_out, eltadd_out, fused_multi_transformer_fuse_qkv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        while0, while0, fused_multi_transformer_fuse_qkv_pattern);

    fuse_creater(input0,
                 layer_norm,
                 layer_norm_scale,
                 layer_norm_bias,
                 layer_norm_mean,
                 layer_norm_variance,
                 c_identity,
                 matmul0,
                 matmul0_w,
                 eltadd0_b,
                 split0_k_out,
                 split0_v_out,
                 eltadd_qk_b,
                 reshape2_0,
                 matmul_linear,
                 matmul_linear_w,
                 eltadd_linear_b,
                 while0,
                 ffn_layer_norm,
                 ffn_layer_norm_scale,
                 ffn_layer_norm_bias,
                 ffn_layer_norm_mean,
                 ffn_layer_norm_variance,
                 ffn_c_identity,
                 ffn_matmul0,
                 ffn_matmul0_w,
                 ffn_matmul1,
                 ffn_matmul1_w,
                 ffn_eltadd0_b,
                 ffn_eltadd1_b,
                 ffn_output);

    std::unordered_set<const Node*> marked_nodes({layer_norm,
                                                  layer_norm_mean,
                                                  layer_norm_variance,
                                                  layer_norm_out,
                                                  c_identity,
                                                  c_identity_out,
                                                  matmul0,
                                                  matmul0_out,
                                                  eltadd0,
                                                  eltadd0_out,
                                                  reshape2_0,
                                                  reshape2_0_out,
                                                  transpose2_0,
                                                  transpose2_0_out,
                                                  split0,
                                                  split0_q_out,
                                                  split0_k_out,
                                                  split0_v_out,
                                                  matmul_qk,
                                                  matmul_qk_out,
                                                  scale_qk,
                                                  scale_qk_out,
                                                  eltadd_qk,
                                                  eltadd_qk_out,
                                                  softmax_qk,
                                                  softmax_qk_out,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_qkv,
                                                  matmul_qkv_out,
                                                  reshape2_qkv,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_linear,
                                                  matmul_linear_out,
                                                  c_allreduce_sum,
                                                  c_allreduce_sum_out,
                                                  eltadd_linear,
                                                  eltadd_linear_out,
                                                  eltadd_out,
                                                  ffn_layer_norm,
                                                  ffn_layer_norm_mean,
                                                  ffn_layer_norm_variance,
                                                  ffn_layer_norm_out,
                                                  ffn_c_identity,
                                                  ffn_c_identity_out,
                                                  ffn_matmul0,
                                                  ffn_matmul1,
                                                  ffn_matmul0_out,
                                                  ffn_matmul1_out,
                                                  ffn_c_allreduce_sum,
                                                  ffn_c_allreduce_sum_out,
                                                  ffn_eltadd0,
                                                  ffn_eltadd1,
                                                  ffn_eltadd0_out,
                                                  ffn_eltadd1_out,
                                                  ffn_gelu,
                                                  ffn_gelu_out,
                                                  ffn_eltadd_out});

    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void MultiDevicesFusedMultiTransformerEncoderFuseQKVPass::ApplyImpl(
    Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the fused_multi_transformer_encoder pass, "
          "The scope should not be null."));

  int fusion_count = BuildFusion(graph, name_scope_, scope);
  if (fusion_count > 0) {
    graph->Set(kMultiDevicesFusedMultiTransformerEncoderFuseQKVPass,
               new bool(true));
    graph->Set(kFusedMultiTransformerEncoderFusionCount, new int(fusion_count));
  }
  AddStatis(fusion_count);
}

MultiDevicesFusedMultiTransformerEncoderFuseQKVPass::
    MultiDevicesFusedMultiTransformerEncoderFuseQKVPass() {
  AddOpCompat(OpCompat("layer_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddOutput("Mean")
      .IsTensor()
      .End()
      .AddOutput("Variance")
      .IsTensor()
      .End()
      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End()
      .AddAttr("begin_norm_axis")
      .IsNumGT(0)
      .End();

  AddOpCompat(OpCompat("scale"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("scale")
      .IsType<float>()  // copy to new op. so unconstrained.
      .End()
      .AddAttr("bias")
      .IsNumEQ(0.f)
      .End()
      .AddAttr("bias_after_scale")  // bias is 0, so unconstrained.
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("matmul_v2"))
      .AddInput("X")  // the shape shoule be (B, S, N*H)
      .IsTensor()
      .End()
      .AddInput("Y")  // the shape shoule be (N*H, N*H)
      .IsTensor()
      .End()
      .AddOutput("Out")  // the shape shoule be (B, S, N*H)
      .IsTensor()
      .End()
      .AddAttr("trans_x")
      .IsType<bool>()
      .End()
      .AddAttr("trans_y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({2, -1, 0})
      .End();

  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Shape")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ShapeTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("shape")  // -->(B, S, H, N)  <--(B, S, N*H)
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("axis")  // {0, 2, 1, 3}
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("matmul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsNumGE(0.0f)
      .IsNumLE(1.0f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("softmax"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({-1, 3})  // shape is (B, H, S, S), so axis is -1 or 3
      .End();

  AddOpCompat(OpCompat("gelu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("approximate")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("while"))
      .AddInput("X")  // A set of variables, unconstrained
      .End()
      .AddInput("Condition")  // An scalar
      .IsTensor()
      .End()
      .AddOutput("Out")  // A set of variables, unconstrained
      .End()
      .AddOutput("StepScopes")  // A vector of local scope, unconstrained
      .End()
      .AddAttr("sub_block")
      .IsType<framework::BlockDesc*>()
      .End();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fused_multi_transformer_encoder_pass,
              paddle::framework::ir::FusedMultiTransformerEncoderPass);
REGISTER_PASS(fused_multi_transformer_encoder_fuse_qkv_pass,
              paddle::framework::ir::FusedMultiTransformerEncoderFuseQKVPass);
REGISTER_PASS(
    multi_devices_fused_multi_transformer_encoder_fuse_qkv_pass,
    paddle::framework::ir::MultiDevicesFusedMultiTransformerEncoderFuseQKVPass);

REGISTER_PASS_CAPABILITY(fused_multi_transformer_encoder_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .LE("matmul", 1)
            .EQ("matmul_v2", 0)
            .EQ("softmax", 0));
REGISTER_PASS_CAPABILITY(fused_multi_transformer_encoder_fuse_qkv_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .LE("matmul", 1)
            .EQ("matmul_v2", 0)
            .EQ("softmax", 0));
REGISTER_PASS_CAPABILITY(
    multi_devices_fused_multi_transformer_encoder_fuse_qkv_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .LE("matmul", 1)
            .EQ("matmul_v2", 0)
            .EQ("softmax", 0));
