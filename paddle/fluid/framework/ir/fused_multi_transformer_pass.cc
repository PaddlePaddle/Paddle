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

#include "paddle/fluid/framework/ir/fused_multi_transformer_pass.h"

#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

PDNode* FusedMultiTransformerPattern::operator()() {
  auto* input0 = pattern->NewNode(input0_repr());
  input0->assert_is_op_input("layer_norm", "X");

  // pre-LayerNorm
  auto* layer_norm = pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto* layer_norm_scale_var = pattern->NewNode(layer_norm_scale_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("layer_norm", "Scale");
  auto* layer_norm_bias_var = pattern->NewNode(layer_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("layer_norm", "Bias");
  auto* layer_norm_mean_var = pattern->NewNode(layer_norm_mean_repr())
                                  ->AsIntermediate()
                                  ->assert_is_op_output("layer_norm", "Mean");
  auto* layer_norm_variance_var = pattern->NewNode(layer_norm_variance_repr())
                                  ->AsIntermediate()
                                  ->assert_is_op_output("layer_norm", "Variance");
  auto* layer_norm_out_var = pattern->NewNode(layer_norm_out_repr())
                                  ->AsIntermediate()
                                  ->assert_is_op_output("layer_norm", "Y")
                                  ->assert_is_op_input("matmul_v2", "X")
                                  ->assert_more([](Node *x) {
                                    if (x->outputs.size() == 3) {
                                      return true;
                                    } else {
                                      return false;
                                    }
                                  });

  layer_norm->LinksFrom({input0, layer_norm_bias_var, layer_norm_scale_var})
            .LinksTo({layer_norm_out_var, layer_norm_mean_var, layer_norm_variance_var});

  // Q path Nodes
  auto* matmul0 = pattern->NewNode(matmul0_repr())->assert_is_op("matmul_v2");
  auto* matmul0_w_var = pattern->NewNode(matmul0_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("matmul_v2", "Y");
  auto* matmul0_out_var = pattern->NewNode(matmul0_out_repr())
                         ->assert_is_op_output("matmul_v2")
                         ->AsIntermediate()
                         ->assert_is_op_input("elementwise_add");

  auto* eltadd0 = pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");
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
  matmul0->LinksFrom({layer_norm_out_var, matmul0_w_var}).LinksTo({matmul0_out_var});
  eltadd0->LinksFrom({matmul0_out_var, eltadd0_b_var}).LinksTo({eltadd0_out_var});
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

  auto* eltadd1 = pattern->NewNode(eltadd1_repr())->assert_is_op("elementwise_add");
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
                                  // ->AsOutput()
                                  ->assert_is_op_input("matmul", "Y");
                                  // ->assert_is_op_input("while")
                                  // ->assert_more([](Node *x) {
                                  //   if (x->outputs.size() == 2) {
                                  //     return true;
                                  //   } else {
                                  //     return false;
                                  //   }
                                  // });

  // K path Links
  matmul1->LinksFrom({layer_norm_out_var, matmul1_w_var}).LinksTo({matmul1_out_var});
  eltadd1->LinksFrom({matmul1_out_var, eltadd1_b_var}).LinksTo({eltadd1_out_var});
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

  auto* eltadd2 = pattern->NewNode(eltadd2_repr())->assert_is_op("elementwise_add");
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
                                  // ->AsOutput()
                                  ->assert_is_op_input("matmul_v2", "Y");
                                  // ->assert_is_op_input("while")
                                  // ->assert_more([](Node *x) {
                                  //   if (x->outputs.size() == 2) {
                                  //     return true;
                                  //   } else {
                                  //     return false;
                                  //   }
                                  // });

  // V path Links
  matmul2->LinksFrom({layer_norm_out_var, matmul2_w_var}).LinksTo({matmul2_out_var});
  eltadd2->LinksFrom({matmul2_out_var, eltadd2_b_var}).LinksTo({eltadd2_out_var});
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
                                ->assert_is_op_input("dropout");

  auto* dropout_qk =
      pattern->NewNode(dropout_qk_repr())->assert_is_op("dropout");
  auto* dropout_qk_out_var = pattern->NewNode(dropout_qk_out_repr())
                                ->assert_is_op_output("dropout", "Out")
                                ->AsIntermediate()
                                ->assert_is_op_input("matmul_v2", "X"); // -> matmul_qkv

  // QK path Linsk
  matmul_qk->LinksFrom({transpose2_0_out_var, transpose2_1_out_var})
      .LinksTo({matmul_qk_out_var});
  eltadd_qk->LinksFrom({matmul_qk_out_var, eltadd_qk_b_var})
      .LinksTo({eltadd_qk_out_var});
  softmax_qk->LinksFrom({eltadd_qk_out_var}).LinksTo({softmax_qk_out_var});
  dropout_qk->LinksFrom({softmax_qk_out_var}).LinksTo({dropout_qk_out_var});

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
  auto* reshape2_qkv_out_var = pattern->NewNode(reshape2_qkv_out_repr())
                                   ->assert_is_op_output("reshape2")
                                   ->AsIntermediate()
                                   ->assert_is_op_input("matmul_v2"); // -> out_linear

  auto* matmul_linear =
      pattern->NewNode(matmul_linear_repr())->assert_is_op("matmul_v2");
  auto* matmul_linear_w_var = pattern->NewNode(matmul_linear_w_repr())
                            ->AsInput()
                            ->assert_is_op_input("matmul_v2", "Y");
  auto* matmul_linear_out_var = pattern->NewNode(matmul_linear_out_repr())
                            ->assert_is_op_output("matmul_v2")
                            ->AsIntermediate()
                            ->assert_is_op_input("elementwise_add");

  auto* eltadd_linear = pattern->NewNode(eltadd_linear_repr())
                               ->assert_is_op("elementwise_add");
  auto* eltadd_linear_b_var = pattern->NewNode(eltadd_linear_b_repr())
                            ->AsInput()
                            ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd_linear_out_var = pattern->NewNode(eltadd_linear_out_repr())
                            ->assert_is_op_output("elementwise_add")
                            ->AsIntermediate()
                            ->assert_is_op_input("dropout");

  auto* dropout_linear =
      pattern->NewNode(dropout_linear_repr())->assert_is_op("dropout");
  auto* dropout_linear_out_var = pattern->NewNode(dropout_linear_out_repr())
                                ->assert_is_op_output("dropout")
                                ->AsIntermediate()
                                ->assert_is_op_input("elementwise_add");

  auto* eltadd_out =
      pattern->NewNode(eltadd_out_repr())->assert_is_op("elementwise_add");
  auto* attention_output = pattern->NewNode(attention_output_repr())
                                ->assert_is_op_output("elementwise_add")
                                ->AsIntermediate();

  // QKV path Links
  matmul_qkv->LinksFrom({dropout_qk_out_var, transpose2_2_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});
  matmul_linear->LinksFrom({reshape2_qkv_out_var, matmul_linear_w_var})
      .LinksTo({matmul_linear_out_var});
  eltadd_linear->LinksFrom({matmul_linear_out_var, eltadd_linear_b_var})
      .LinksTo({eltadd_linear_out_var});
  dropout_linear->LinksFrom({eltadd_linear_out_var})
      .LinksTo({dropout_linear_out_var});
  eltadd_out->LinksFrom({input0, dropout_linear_out_var})
      .LinksTo({attention_output});

  // // while loop
  // auto* while0 =
  //     pattern->NewNode(while0_repr())->assert_is_op("while");
  // while0->LinksFrom({transpose2_1_out_var, transpose2_2_out_var});

  // Feed Forward LayerNorm Nodes
  auto* ffn_layer_norm = pattern->NewNode(ffn_layer_norm_repr())->assert_is_op("layer_norm");
  auto* ffn_layer_norm_scale_var = \
                          pattern->NewNode(ffn_layer_norm_scale_repr())
                            ->AsInput()
                            ->assert_is_persistable_var()
                            ->assert_is_op_input("layer_norm", "Scale");
  auto* ffn_layer_norm_bias_var = \
                          pattern->NewNode(ffn_layer_norm_bias_repr())
                            ->AsInput()
                            ->assert_is_persistable_var()
                            ->assert_is_op_input("layer_norm", "Bias");
  auto* ffn_layer_norm_mean_var = \
                          pattern->NewNode(ffn_layer_norm_mean_repr())
                            ->AsIntermediate()
                            ->assert_is_op_output("layer_norm", "Mean");
  auto* ffn_layer_norm_variance_var = \
                          pattern->NewNode(ffn_layer_norm_variance_repr())
                            ->AsIntermediate()
                            ->assert_is_op_output("layer_norm", "Variance");
  auto* ffn_layer_norm_out_var = \
                          pattern->NewNode(ffn_layer_norm_out_repr())
                            ->AsIntermediate()
                            ->assert_is_op_output("layer_norm", "Y")
                            ->assert_is_op_input("matmul_v2", "X");

  ffn_layer_norm->LinksFrom({attention_output, ffn_layer_norm_bias_var, ffn_layer_norm_scale_var})
            .LinksTo({ffn_layer_norm_out_var, ffn_layer_norm_mean_var, ffn_layer_norm_variance_var});


  // Feed Forward fc1 -> gelu -> fc2 -> dropout
  auto* ffn_matmul0 = pattern->NewNode(ffn_matmul0_repr())->assert_is_op("matmul_v2");
  auto* ffn_matmul0_w_var = pattern->NewNode(ffn_matmul0_w_repr())
                              ->AsInput()
                              ->assert_is_op_input("matmul_v2", "Y");
  auto* ffn_matmul0_out_var = pattern->NewNode(ffn_matmul0_out_repr())
                              ->assert_is_op_output("matmul_v2")
                              ->AsIntermediate()
                              ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd0 = pattern->NewNode(ffn_eltadd0_repr())->assert_is_op("elementwise_add");
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

  auto* ffn_matmul1 = pattern->NewNode(ffn_matmul1_repr())->assert_is_op("matmul_v2");
  auto* ffn_matmul1_w_var = pattern->NewNode(ffn_matmul1_w_repr())
                              ->AsInput()
                              ->assert_is_op_input("matmul_v2", "Y");
  auto* ffn_matmul1_out_var = pattern->NewNode(ffn_matmul1_out_repr())
                              ->assert_is_op_output("matmul_v2")
                              ->AsIntermediate()
                              ->assert_is_op_input("elementwise_add");

  auto* ffn_eltadd1 = pattern->NewNode(ffn_eltadd1_repr())->assert_is_op("elementwise_add");
  auto* ffn_eltadd1_b_var = pattern->NewNode(ffn_eltadd1_b_repr())
                            ->AsInput()
                            ->assert_is_op_input("elementwise_add", "Y");
  auto* ffn_eltadd1_out_var = pattern->NewNode(ffn_eltadd1_out_repr())
                            ->assert_is_op_output("elementwise_add")
                            ->AsIntermediate()
                            ->assert_is_op_input("dropout");

  auto* ffn_dropout =
      pattern->NewNode(ffn_dropout_repr())->assert_is_op("dropout");
  auto* ffn_dropout_out_var = pattern->NewNode(ffn_dropout_out_repr())
                                ->assert_is_op_output("dropout")
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
  ffn_dropout->LinksFrom({ffn_eltadd1_out_var})
      .LinksTo({ffn_dropout_out_var});

  ffn_eltadd_out->LinksFrom({attention_output, ffn_dropout_out_var})
      .LinksTo({ffn_output});

  return ffn_output;
}

} // namespace pattern

int FusedMultiTransformerPass::BuildFusion(Graph* graph, const std::string& name_scope, Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::FusedMultiTransformerPattern fused_multi_transformer_pattern(pattern, name_scope);
  fused_multi_transformer_pattern();

  // Create New OpDesc
  auto fuse_creater = [&](Node* input0,
                          Node* layer_norm,
                          Node* layer_norm_scale,
                          Node* layer_norm_bias,
                          Node* layer_norm_mean,
                          Node* layer_norm_variance,
                          Node* matmul0_w,
                          Node* matmul1_w,
                          Node* matmul2_w,
                          Node* eltadd0_b,
                          Node* eltadd1_b,
                          Node* eltadd2_b,
                          Node* transpose2_1_out,
                          Node* transpose2_2_out,
                          Node* eltadd_qk_b,
                          Node* dropout_qk,
                          Node* reshape2_0,
                          Node* matmul_linear_w,
                          Node* eltadd_linear_b,
                          Node* dropout_linear,
                          // Node* while0,
                          Node* ffn_layer_norm,
                          Node* ffn_layer_norm_scale,
                          Node* ffn_layer_norm_bias,
                          Node* ffn_layer_norm_mean,
                          Node* ffn_layer_norm_variance,
                          Node* ffn_matmul0_w,
                          Node* ffn_matmul1_w,
                          Node* ffn_eltadd0_b,
                          Node* ffn_eltadd1_b,
                          Node* ffn_dropout,
                          Node* ffn_output) {
    auto reshape_desc = reshape2_0->Op();
    int num_head =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(2);
    int dim_head =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(3);
    int dim_embed = num_head * dim_head;

    auto* wq_tensor = scope->FindVar(matmul0_w->Name())->GetMutable<LoDTensor>();
    auto* wk_tensor = scope->FindVar(matmul1_w->Name())->GetMutable<LoDTensor>();
    auto* wv_tensor = scope->FindVar(matmul2_w->Name())->GetMutable<LoDTensor>();

    auto* bq_tensor =
        scope->FindVar(eltadd0_b->Name())->GetMutable<LoDTensor>();
    auto* bk_tensor =
        scope->FindVar(eltadd1_b->Name())->GetMutable<LoDTensor>();
    auto* bv_tensor =
        scope->FindVar(eltadd2_b->Name())->GetMutable<LoDTensor>();

    auto* wq_data = wq_tensor->mutable_data<float>(platform::CPUPlace());
    auto* wk_data = wk_tensor->mutable_data<float>(platform::CPUPlace());
    auto* wv_data = wv_tensor->mutable_data<float>(platform::CPUPlace());
    auto* bq_data = bq_tensor->mutable_data<float>(platform::CPUPlace());
    auto* bk_data = bk_tensor->mutable_data<float>(platform::CPUPlace());
    auto* bv_data = bv_tensor->mutable_data<float>(platform::CPUPlace());

    auto combined_w_dims =
        phi::make_ddim({3, num_head, dim_head, dim_embed});
    auto combined_bias_dims = phi::make_ddim({3, num_head, dim_head});

    // reuse the mul0_w and eltadd_0_b nodes for the combined nodes.
    auto* combined_w_desc = matmul0_w->Var();
    combined_w_desc->SetShape({3, num_head, dim_head, dim_embed});
    combined_w_desc->SetPersistable(true);

    auto* combined_bias_desc = eltadd0_b->Var();
    combined_bias_desc->SetShape({3, num_head, dim_head});
    combined_bias_desc->SetPersistable(true);

    framework::LoDTensor tmp_combined_w_tensor;
    tmp_combined_w_tensor.Resize(combined_w_dims);
    auto* tmp_combined_w_data =
        tmp_combined_w_tensor.mutable_data<float>(platform::CPUPlace());

    std::vector<float*> w_vec = {wq_data, wk_data, wv_data};
    // Combine the three fc weights together.
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < num_head; j++) {
        for (int k = 0; k < dim_head; k++) {
          for (int l = 0; l < dim_embed; l++) {
            int out_idx = i * num_head * dim_head * dim_embed \
                          + j * dim_head * dim_embed \
                          + k * dim_embed + l;
            int in_idx = l * num_head * dim_head + j * dim_head + k;
            tmp_combined_w_data[out_idx] = w_vec[i][in_idx];
          }
        }
      }
    }

    wq_tensor->Resize(combined_w_dims);
    auto* new_combined_w_data =
        wq_tensor->mutable_data<float>(platform::CPUPlace());
    memcpy(new_combined_w_data,
           tmp_combined_w_data,
           sizeof(float) * wq_tensor->numel());

    scope->EraseVars({matmul1_w->Name(), matmul2_w->Name()});

    framework::LoDTensor tmp_combined_bias_tensor;
    tmp_combined_bias_tensor.Resize(combined_bias_dims);
    auto* tmp_combined_bias_data =
        tmp_combined_bias_tensor.mutable_data<float>(platform::CPUPlace());

    size_t bias_size = bq_tensor->numel();
    memcpy(tmp_combined_bias_data, bq_data, sizeof(float) * bias_size);
    memcpy(
        tmp_combined_bias_data + bias_size, bk_data, sizeof(float) * bias_size);
    memcpy(tmp_combined_bias_data + 2 * bias_size,
           bv_data,
           sizeof(float) * bias_size);

    bq_tensor->Resize(combined_bias_dims);
    auto* new_combined_bias_data =
        bq_tensor->mutable_data<float>(platform::CPUPlace());
    memcpy(new_combined_bias_data,
           tmp_combined_bias_data,
           sizeof(float) * bq_tensor->numel());

    scope->EraseVars({eltadd1_b->Name(), eltadd2_b->Name()});

    // create fused_multi_transformer
    OpDesc fused_multi_transformer_op_desc(layer_norm->Op()->Block());
    fused_multi_transformer_op_desc.SetType("fused_multi_transformer");

    // 1. Input setting
    fused_multi_transformer_op_desc.SetInput("X", {input0->Name()});

    // pre-LayerNorm input
    fused_multi_transformer_op_desc.SetInput("LnScale", {layer_norm_scale->Name()});
    fused_multi_transformer_op_desc.SetInput("LnBias", {layer_norm_bias->Name()});

    // QKV computation input
    fused_multi_transformer_op_desc.SetInput("QKVW", {matmul0_w->Name()});
    fused_multi_transformer_op_desc.SetInput("QKVBias", {eltadd0_b->Name()});
    fused_multi_transformer_op_desc.SetInput("SrcMask", {eltadd_qk_b->Name()});
    
    // CacheKV input
    VarDesc cache_kv_desc(
        patterns::PDNodeName("cache_kv", layer_norm->Name()));
    // FIXME: only support batch_size = 1, and max_seq_len <= 1024
    cache_kv_desc.SetShape(phi::vectorize({2, 1, num_head, 1024, dim_head}));
    cache_kv_desc.SetDataType(
        framework::TransToProtoVarType(wq_tensor->dtype()));
    cache_kv_desc.SetPersistable(true);
    auto* cache_kv = graph->CreateVarNode(&cache_kv_desc);
    auto* cache_kv_tensor = scope->Var(cache_kv->Name())->GetMutable<LoDTensor>();
    cache_kv_tensor->Resize(DDim{2, 1, num_head, 1024, dim_head});
    std::fill_n(cache_kv_tensor->mutable_data<float>(platform::CPUPlace()),
                cache_kv_tensor->numel(), 0.0f);

    fused_multi_transformer_op_desc.SetInput("CacheKV", {cache_kv->Name()});

    // Out Linear input
    fused_multi_transformer_op_desc.SetInput("OutLinearW", {matmul_linear_w->Name()});
    fused_multi_transformer_op_desc.SetInput("OutLinearBias", {eltadd_linear_b->Name()});

    // Feed Forward input
    fused_multi_transformer_op_desc.SetInput("FFNLnScale", {ffn_layer_norm_scale->Name()});
    fused_multi_transformer_op_desc.SetInput("FFNLnBias", {ffn_layer_norm_bias->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN1Weight", {ffn_matmul0_w->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN1Bias", {ffn_eltadd0_b->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN2Weight", {ffn_matmul1_w->Name()});
    fused_multi_transformer_op_desc.SetInput("FFN2Bias", {ffn_eltadd1_b->Name()});

    // 2. Output setting
    fused_multi_transformer_op_desc.SetOutput("Out", {ffn_output->Name()});
    fused_multi_transformer_op_desc.SetOutput("CacheKVOut", {cache_kv->Name()});

    // Attribute setting
    fused_multi_transformer_op_desc.SetAttr("pre_layer_norm", true);
    fused_multi_transformer_op_desc.SetAttr("epsilon", layer_norm->Op()->GetAttr("epsilon"));

    // // attention(qk) dropout attribute
    // auto* dropout_qk_op = dropout_qk->Op();
    // fused_multi_transformer_op_desc.SetAttr("attn_dropout_rate", dropout_qk_op->GetAttr("dropout_prob"));
    // fused_multi_transformer_op_desc.SetAttr("attn_dropout_fix_seed", dropout_qk_op->GetAttr("fix_seed"));
    // fused_multi_transformer_op_desc.SetAttr("attn_dropout_seed", dropout_qk_op->GetAttr("seed"));
    // fused_multi_transformer_op_desc.SetAttr("attn_dropout_implementation", dropout_qk_op->GetAttr("dropout_implementation"));

    // output dropout attribute
    auto* dropout_op = dropout_linear->Op();
    fused_multi_transformer_op_desc.SetAttr("dropout_rate", dropout_op->GetAttr("dropout_prob"));
    fused_multi_transformer_op_desc.SetAttr("is_test", dropout_op->GetAttr("is_test"));
    fused_multi_transformer_op_desc.SetAttr("dropout_implementation", dropout_op->GetAttr("dropout_implementation"));
    // fused_multi_transformer_op_desc.SetAttr("dropout_fix_seed", dropout_op->GetAttr("fix_seed"));
    // fused_multi_transformer_op_desc.SetAttr("dropout_seed", dropout_op->GetAttr("seed"));

    // fused_multi_transformer_op_desc.SetAttr("act_method", "gelu");
    // fused_multi_transformer_op_desc.SetAttr("trans_qkvw", true);

    auto* fused_multi_transformer = graph->CreateOpNode(&fused_multi_transformer_op_desc);
    IR_NODE_LINK_TO(input0, fused_multi_transformer);
    IR_NODE_LINK_TO(layer_norm_scale, fused_multi_transformer);
    IR_NODE_LINK_TO(layer_norm_bias, fused_multi_transformer);

    IR_NODE_LINK_TO(matmul0_w, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd0_b, fused_multi_transformer);
    IR_NODE_LINK_TO(eltadd_qk_b, fused_multi_transformer);

    IR_NODE_LINK_TO(fused_multi_transformer, ffn_output);
    
    // // // link CacheKV to while
    // // IR_NODE_LINK_TO(cache_kv, while0)
    // // unlink origin KV output to while
    // IR_NODE_UNLINK(transpose2_1_out, while0);
    // IR_NODE_UNLINK(transpose2_2_out, while0);
    // // unlink KV weight/bias to while after merged into Q weight/bias
    // IR_NODE_UNLINK(matmul1_w, while0);
    // IR_NODE_UNLINK(matmul2_w, while0);
    // IR_NODE_UNLINK(eltadd1_b, while0);
    // IR_NODE_UNLINK(eltadd2_b, while0);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(input0, input0, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_scale, layer_norm_scale, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias, layer_norm_bias, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_mean, layer_norm_mean, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_variance, layer_norm_variance, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul0, matmul0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul0_out, matmul0_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul0_w, matmul0_w, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0_out, reshape2_0_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0_out, transpose2_0_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul1, matmul1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul1_out, matmul1_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul1_w, matmul1_w, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_1_out, reshape2_1_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_1_out, transpose2_1_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul2, matmul2, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul2_out, matmul2_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul2_w, matmul2_w, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_2_out, reshape2_2_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_2_out, transpose2_2_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(attention_output, attention_output, fused_multi_transformer_pattern)
    // GET_IR_NODE_FROM_SUBGRAPH(while0, while0, fused_multi_transformer_pattern)

    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm, ffn_layer_norm, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_scale, ffn_layer_norm_scale, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_bias, ffn_layer_norm_bias, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_mean, ffn_layer_norm_mean, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_variance, ffn_layer_norm_variance, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_layer_norm_out, ffn_layer_norm_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(ffn_matmul0, ffn_matmul0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_matmul0_out, ffn_matmul0_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_matmul0_w, ffn_matmul0_w, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd0, ffn_eltadd0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd0_b, ffn_eltadd0_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd0_out, ffn_eltadd0_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(ffn_gelu, ffn_gelu, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_gelu_out, ffn_gelu_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(ffn_matmul1, ffn_matmul1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_matmul1_out, ffn_matmul1_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_matmul1_w, ffn_matmul1_w, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd1, ffn_eltadd1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd1_b, ffn_eltadd1_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd1_out, ffn_eltadd1_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(ffn_dropout, ffn_dropout, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(ffn_dropout_out, ffn_dropout_out, fused_multi_transformer_pattern)

    GET_IR_NODE_FROM_SUBGRAPH(ffn_eltadd_out, ffn_eltadd_out, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(ffn_output, ffn_output, fused_multi_transformer_pattern)

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0, eltadd0, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_b, eltadd0_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_out, eltadd0_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_b, eltadd1_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd2, eltadd2, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_b, eltadd2_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_out, eltadd2_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk, matmul_qk, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk_out, matmul_qk_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk, eltadd_qk, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_b, eltadd_qk_b, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_out, eltadd_qk_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk, softmax_qk, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk_out, softmax_qk_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dropout_qk, dropout_qk, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(dropout_qk_out, dropout_qk_out, fused_multi_transformer_pattern)

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv_out, matmul_qkv_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv_out, reshape2_qkv_out, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv, transpose2_qkv, fused_multi_transformer_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv_out, transpose2_qkv_out, fused_multi_transformer_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_linear, matmul_linear, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(matmul_linear_w, matmul_linear_w, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(matmul_linear_out, matmul_linear_out, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_linear, eltadd_linear, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_linear_b, eltadd_linear_b, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_linear_out, eltadd_linear_out, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(dropout_linear, dropout_linear, fused_multi_transformer_pattern)
    GET_IR_NODE_FROM_SUBGRAPH(dropout_linear_out, dropout_linear_out, fused_multi_transformer_pattern)

    GET_IR_NODE_FROM_SUBGRAPH(eltadd_out, eltadd_out, fused_multi_transformer_pattern)

  fuse_creater(input0,
               layer_norm,
               layer_norm_scale,
               layer_norm_bias,
               layer_norm_mean,
               layer_norm_variance,
               matmul0_w,
               matmul1_w,
               matmul2_w,
               eltadd0_b,
               eltadd1_b,
               eltadd2_b,
               transpose2_1_out,
               transpose2_2_out,
               eltadd_qk_b,
               dropout_qk,
               reshape2_0,
               matmul_linear_w,
               eltadd_linear_b,
               dropout_linear,
               // while0,
               ffn_layer_norm,
               ffn_layer_norm_scale,
               ffn_layer_norm_bias,
               ffn_layer_norm_mean,
               ffn_layer_norm_variance,
               ffn_matmul0_w,
               ffn_matmul1_w,
               ffn_eltadd0_b,
               ffn_eltadd1_b,
               ffn_dropout,
               ffn_output);

    std::unordered_set<const Node*> marked_nodes(
        {layer_norm,
         layer_norm_scale,
         layer_norm_bias,
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
         dropout_qk,
         dropout_qk_out,
         transpose2_qkv,
         transpose2_qkv_out,
         matmul_qkv,
         matmul_qkv_out,
         reshape2_qkv,
         transpose2_qkv,
         transpose2_qkv_out,
         matmul_linear,
         matmul_linear_w,
         matmul_linear_out,
         eltadd_linear,
         eltadd_linear_b,
         eltadd_linear_out,
         dropout_linear,
         dropout_linear_out,
         eltadd_out,
         ffn_layer_norm,
         ffn_layer_norm_scale,
         ffn_layer_norm_bias,
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
         ffn_dropout,
         ffn_dropout_out,
         ffn_eltadd_out});

    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void FusedMultiTransformerPass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the multi_transformer pass, The scope should not be null."));

  int fusion_count = BuildFusion(graph, name_scope_, scope);
  LOG(ERROR) << "FusedMultiTransformer fusion_count: " << fusion_count;
  if (fusion_count > 0) {
    graph->Set(kFusedMultiTransformerPass, new bool(true));
  }
  AddStatis(fusion_count);
}

FusedMultiTransformerPass::FusedMultiTransformerPass() {
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
      .AddAttr("x_num_col_dims")
      .IsNumEQ(2)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      // in bias, shape is (B, S, N*H),
      // in biasqk, shape is (B, H, S, S)
      .IsTensor()
      .End()
      .AddInput("Y")
      // in bias, shape is (N*H)
      // in biasqk, shape is (B, H, S, S)
      .IsTensor()
      .End()
      // in bias, shape is (B, S, N*H)
      // in biasqk, shape is (B, H, S, S)
      .AddOutput("Out")
      .IsTensor()
      .End()
      // in bias, it equal to 2
      // in biasqk, it equal to -1 or 0
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

  // -->: (B, S, H, N) -> (B, H, S, N)
  // <--: (B, H, S, N) -> (B, S, H, N)
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
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fused_multi_transformer_pass,
              paddle::framework::ir::FusedMultiTransformerPass);

// REGISTER_PASS_CAPABILITY(multihead_matmul_fuse_pass_v2)
//     .AddCombination(
//         paddle::framework::compatible::OpVersionComparatorCombination()
//             .EQ("mul", 0)
//             .LE("elementwise_add", 1)
//             .EQ("reshape2", 0)
//             .EQ("transpose2", 0)
//             .EQ("scale", 0)
//             .LE("matmul", 1)
//             .EQ("softmax", 0));
