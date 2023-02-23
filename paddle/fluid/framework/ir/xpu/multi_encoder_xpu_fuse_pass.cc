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

struct SingleEncoderXPUPattern : public PatternBase {
  SingleEncoderXPUPattern(PDPattern* pattern,
                          const std::string& name_scope,
                          const std::string& act_type,
                          const std::string& matmul_type_0,
                          const std::string& matmul_type_1,
                          const std::string& matmul_type_2,
                          bool norm_before,
                          bool with_q_scale,
                          bool with_mask);

  // declare operator node's name
  // If norm_before, use ln_0 & ln_1.
  // If not norm_before, use ln_1 & ln_2.
  PATTERN_DECL_NODE(ln_0);
  PATTERN_DECL_NODE(ln_1);
  PATTERN_DECL_NODE(ln_2);
  PATTERN_DECL_NODE(q_matmul);
  PATTERN_DECL_NODE(q_add);
  PATTERN_DECL_NODE(q_reshape);
  PATTERN_DECL_NODE(q_transpose);
  PATTERN_DECL_NODE(q_scale);
  PATTERN_DECL_NODE(k_matmul);
  PATTERN_DECL_NODE(k_add);
  PATTERN_DECL_NODE(k_reshape);
  PATTERN_DECL_NODE(k_transpose);
  PATTERN_DECL_NODE(v_matmul);
  PATTERN_DECL_NODE(v_add);
  PATTERN_DECL_NODE(v_reshape);
  PATTERN_DECL_NODE(v_transpose);
  PATTERN_DECL_NODE(qk_matmul);
  PATTERN_DECL_NODE(qk_add);
  PATTERN_DECL_NODE(qk_softmax);
  PATTERN_DECL_NODE(qkv_matmul_0);
  PATTERN_DECL_NODE(qkv_transpose);
  PATTERN_DECL_NODE(qkv_reshape);
  PATTERN_DECL_NODE(qkv_matmul_1);
  PATTERN_DECL_NODE(qkv_add_0);
  PATTERN_DECL_NODE(qkv_add_1);
  PATTERN_DECL_NODE(qkv_matmul_2);
  PATTERN_DECL_NODE(qkv_add_2);
  PATTERN_DECL_NODE(qkv_act);
  PATTERN_DECL_NODE(qkv_matmul_3);
  PATTERN_DECL_NODE(qkv_add_3);
  PATTERN_DECL_NODE(qkv_add_4);
  // declare variable node's name
  PATTERN_DECL_NODE(ln_0_x);
  PATTERN_DECL_NODE(ln_0_bias);
  PATTERN_DECL_NODE(ln_0_scale);
  PATTERN_DECL_NODE(ln_0_out);
  PATTERN_DECL_NODE(ln_0_mean);
  PATTERN_DECL_NODE(ln_0_variance);
  PATTERN_DECL_NODE(q_matmul_w);
  PATTERN_DECL_NODE(q_matmul_out);
  PATTERN_DECL_NODE(q_add_bias);
  PATTERN_DECL_NODE(q_add_out);
  PATTERN_DECL_NODE(q_reshape_out);
  PATTERN_DECL_NODE(q_transpose_out);
  PATTERN_DECL_NODE(q_scale_out);
  PATTERN_DECL_NODE(k_matmul_w);
  PATTERN_DECL_NODE(k_matmul_out);
  PATTERN_DECL_NODE(k_add_bias);
  PATTERN_DECL_NODE(k_add_out);
  PATTERN_DECL_NODE(k_reshape_out);
  PATTERN_DECL_NODE(k_transpose_out);
  PATTERN_DECL_NODE(v_matmul_w);
  PATTERN_DECL_NODE(v_matmul_out);
  PATTERN_DECL_NODE(v_add_bias);
  PATTERN_DECL_NODE(v_add_out);
  PATTERN_DECL_NODE(v_reshape_out);
  PATTERN_DECL_NODE(v_transpose_out);
  PATTERN_DECL_NODE(qk_matmul_out);
  PATTERN_DECL_NODE(qk_add_mask);
  PATTERN_DECL_NODE(qk_add_out);
  PATTERN_DECL_NODE(qk_softmax_out);
  PATTERN_DECL_NODE(qkv_matmul_0_out);
  PATTERN_DECL_NODE(qkv_transpose_out);
  PATTERN_DECL_NODE(qkv_reshape_out);
  PATTERN_DECL_NODE(qkv_matmul_1_w);
  PATTERN_DECL_NODE(qkv_matmul_1_out);
  PATTERN_DECL_NODE(qkv_add_0_bias);
  PATTERN_DECL_NODE(qkv_add_0_out);
  PATTERN_DECL_NODE(qkv_add_1_out);
  PATTERN_DECL_NODE(ln_1_bias);
  PATTERN_DECL_NODE(ln_1_scale);
  PATTERN_DECL_NODE(ln_1_out);
  PATTERN_DECL_NODE(ln_1_mean);
  PATTERN_DECL_NODE(ln_1_variance);
  PATTERN_DECL_NODE(qkv_matmul_2_w);
  PATTERN_DECL_NODE(qkv_matmul_2_out);
  PATTERN_DECL_NODE(qkv_add_2_bias);
  PATTERN_DECL_NODE(qkv_add_2_out);
  PATTERN_DECL_NODE(qkv_act_out);
  PATTERN_DECL_NODE(qkv_matmul_3_w);
  PATTERN_DECL_NODE(qkv_matmul_3_out);
  PATTERN_DECL_NODE(qkv_add_3_bias);
  PATTERN_DECL_NODE(qkv_add_3_out);
  PATTERN_DECL_NODE(qkv_add_4_out);
  PATTERN_DECL_NODE(ln_2_x);
  PATTERN_DECL_NODE(ln_2_bias);
  PATTERN_DECL_NODE(ln_2_scale);
  PATTERN_DECL_NODE(ln_2_out);
  PATTERN_DECL_NODE(ln_2_mean);
  PATTERN_DECL_NODE(ln_2_variance);

 private:
  std::string act_type_;
  std::string matmul_type_0_;
  std::string matmul_type_1_;
  std::string matmul_type_2_;
  bool norm_before_{false};
  bool with_q_scale_{false};
  bool with_mask_{true};
};

SingleEncoderXPUPattern::SingleEncoderXPUPattern(
    PDPattern* pattern,
    const std::string& name_scope,
    const std::string& act_type,
    const std::string& matmul_type_0,
    const std::string& matmul_type_1,
    const std::string& matmul_type_2,
    bool norm_before,
    bool with_q_scale,
    bool with_mask)
    : PatternBase(pattern, name_scope, name_scope),
      act_type_(act_type),
      matmul_type_0_(matmul_type_0),
      matmul_type_1_(matmul_type_1),
      matmul_type_2_(matmul_type_2),
      norm_before_(norm_before),
      with_q_scale_(with_q_scale),
      with_mask_(with_mask) {
  // layer_norm 0
  PDNode* ln_0_x = pattern->NewNode(ln_0_x_repr());
  PDNode* ln_0_bias = nullptr;
  PDNode* ln_0_scale = nullptr;
  PDNode* ln_0 = nullptr;
  PDNode* ln_0_out = nullptr;
  PDNode* ln_0_mean = nullptr;
  PDNode* ln_0_variance = nullptr;
  if (norm_before_) {
    ln_0_x->assert_is_op_input("layer_norm", "X")->assert_var_not_persistable();
    ln_0_bias = pattern->NewNode(ln_0_bias_repr())
                    ->assert_is_op_input("layer_norm", "Bias")
                    ->assert_is_persistable_var();
    ln_0_scale = pattern->NewNode(ln_0_scale_repr())
                     ->assert_is_op_input("layer_norm", "Scale")
                     ->assert_is_persistable_var();
    ln_0 = pattern->NewNode(ln_0_repr())->assert_is_op("layer_norm");
    ln_0_out = pattern->NewNode(ln_0_out_repr())
                   ->assert_is_op_output("layer_norm", "Y")
                   ->assert_var_not_persistable();
    ln_0_mean = pattern->NewNode(ln_0_mean_repr())
                    ->assert_is_op_output("layer_norm", "Mean")
                    ->assert_var_not_persistable();
    ln_0_variance = pattern->NewNode(ln_0_variance_repr())
                        ->assert_is_op_output("layer_norm", "Variance")
                        ->assert_var_not_persistable();
  }

  // q: matmul + add + reshape + transpose
  auto q_matmul_w = pattern->NewNode(q_matmul_w_repr())
                        ->assert_is_op_input(matmul_type_0_, "Y")
                        ->assert_is_persistable_var();
  auto* q_matmul =
      pattern->NewNode(q_matmul_repr())->assert_is_op(matmul_type_0_);
  auto* q_matmul_out = pattern->NewNode(q_matmul_out_repr())
                           ->assert_is_op_output(matmul_type_0_, "Out")
                           ->assert_var_not_persistable();
  auto q_add_bias = pattern->NewNode(q_add_bias_repr())
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->assert_is_persistable_var();
  auto* q_add = pattern->NewNode(q_add_repr())->assert_is_op("elementwise_add");
  auto* q_add_out = pattern->NewNode(q_add_out_repr())
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_var_not_persistable();
  auto* q_reshape =
      pattern->NewNode(q_reshape_repr())->assert_is_op("reshape2");
  auto* q_reshape_out = pattern->NewNode(q_reshape_out_repr())
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_var_not_persistable();
  auto* q_transpose =
      pattern->NewNode(q_transpose_repr())->assert_is_op("transpose2");
  auto* q_transpose_out = pattern->NewNode(q_transpose_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_var_not_persistable();
  PDNode* q_scale = nullptr;
  PDNode* q_scale_out = nullptr;
  if (with_q_scale_) {
    q_scale = pattern->NewNode(q_scale_repr())->assert_is_op("scale");
    q_scale_out = pattern->NewNode(q_scale_out_repr())
                      ->assert_is_op_output("scale", "Out")
                      ->assert_is_op_input(matmul_type_1_, "X")
                      ->assert_var_not_persistable();
  } else {
    q_transpose_out->assert_is_op_input(matmul_type_1_, "X");
  }

  // k: matmul + add + reshape + transpose
  auto k_matmul_w = pattern->NewNode(k_matmul_w_repr())
                        ->assert_is_op_input(matmul_type_0_, "Y")
                        ->assert_is_persistable_var();
  auto* k_matmul =
      pattern->NewNode(k_matmul_repr())->assert_is_op(matmul_type_0_);
  auto* k_matmul_out = pattern->NewNode(k_matmul_out_repr())
                           ->assert_is_op_output(matmul_type_0_, "Out")
                           ->assert_var_not_persistable();
  auto k_add_bias = pattern->NewNode(k_add_bias_repr())
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->assert_is_persistable_var();
  auto* k_add = pattern->NewNode(k_add_repr())->assert_is_op("elementwise_add");
  auto* k_add_out = pattern->NewNode(k_add_out_repr())
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_var_not_persistable();
  auto* k_reshape =
      pattern->NewNode(k_reshape_repr())->assert_is_op("reshape2");
  auto* k_reshape_out = pattern->NewNode(k_reshape_out_repr())
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_var_not_persistable();
  auto* k_transpose =
      pattern->NewNode(k_transpose_repr())->assert_is_op("transpose2");
  auto* k_transpose_out = pattern->NewNode(k_transpose_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_is_op_input(matmul_type_1_, "Y")
                              ->assert_var_not_persistable();

  // qk: matmul + add + softmax
  auto* qk_matmul =
      pattern->NewNode(qk_matmul_repr())->assert_is_op(matmul_type_1_);
  auto* qk_matmul_out = pattern->NewNode(qk_matmul_out_repr())
                            ->assert_is_op_output(matmul_type_1_, "Out")
                            ->assert_var_not_persistable();
  PDNode* qk_add_mask = nullptr;
  PDNode* qk_add = nullptr;
  PDNode* qk_add_out = nullptr;
  if (with_mask_) {
    qk_add_mask = pattern->NewNode(qk_add_mask_repr())
                      ->assert_is_op_input("elementwise_add", "Y")
                      ->assert_var_not_persistable();
    qk_add = pattern->NewNode(qk_add_repr())->assert_is_op("elementwise_add");
    qk_add_out = pattern->NewNode(qk_add_out_repr())
                     ->assert_is_op_output("elementwise_add", "Out")
                     ->assert_var_not_persistable();
  }
  auto* qk_softmax =
      pattern->NewNode(qk_softmax_repr())->assert_is_op("softmax");
  auto* qk_softmax_out = pattern->NewNode(qk_softmax_out_repr())
                             ->assert_is_op_output("softmax", "Out")
                             ->assert_is_op_input(matmul_type_2_, "X")
                             ->assert_var_not_persistable();

  // v: matmul + add + reshape + transpose
  auto v_matmul_w = pattern->NewNode(v_matmul_w_repr())
                        ->assert_is_op_input(matmul_type_0_, "Y")
                        ->assert_is_persistable_var();
  auto* v_matmul =
      pattern->NewNode(v_matmul_repr())->assert_is_op(matmul_type_0_);
  auto* v_matmul_out = pattern->NewNode(v_matmul_out_repr())
                           ->assert_is_op_output(matmul_type_0_, "Out")
                           ->assert_var_not_persistable();
  auto v_add_bias = pattern->NewNode(v_add_bias_repr())
                        ->assert_is_op_input("elementwise_add", "Y")
                        ->assert_is_persistable_var();
  auto* v_add = pattern->NewNode(v_add_repr())->assert_is_op("elementwise_add");
  auto* v_add_out = pattern->NewNode(v_add_out_repr())
                        ->assert_is_op_output("elementwise_add", "Out")
                        ->assert_var_not_persistable();
  auto* v_reshape =
      pattern->NewNode(v_reshape_repr())->assert_is_op("reshape2");
  auto* v_reshape_out = pattern->NewNode(v_reshape_out_repr())
                            ->assert_is_op_output("reshape2", "Out")
                            ->assert_var_not_persistable();
  auto* v_transpose =
      pattern->NewNode(v_transpose_repr())->assert_is_op("transpose2");
  auto* v_transpose_out = pattern->NewNode(v_transpose_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_is_op_input(matmul_type_2_, "Y")
                              ->assert_var_not_persistable();

  // qkv
  auto* qkv_matmul_0 =
      pattern->NewNode(qkv_matmul_0_repr())->assert_is_op(matmul_type_2_);
  auto* qkv_matmul_0_out = pattern->NewNode(qkv_matmul_0_out_repr())
                               ->assert_is_op_output(matmul_type_2_, "Out")
                               ->assert_var_not_persistable();
  auto* qkv_transpose =
      pattern->NewNode(qkv_transpose_repr())->assert_is_op("transpose2");
  auto* qkv_transpose_out = pattern->NewNode(qkv_transpose_out_repr())
                                ->assert_is_op_output("transpose2", "Out")
                                ->assert_var_not_persistable();
  auto* qkv_reshape =
      pattern->NewNode(qkv_reshape_repr())->assert_is_op("reshape2");
  auto* qkv_reshape_out = pattern->NewNode(qkv_reshape_out_repr())
                              ->assert_is_op_output("reshape2", "Out")
                              ->assert_var_not_persistable();
  auto qkv_matmul_1_w = pattern->NewNode(qkv_matmul_1_w_repr())
                            ->assert_is_op_input(matmul_type_0_, "Y")
                            ->assert_is_persistable_var();
  auto* qkv_matmul_1 =
      pattern->NewNode(qkv_matmul_1_repr())->assert_is_op(matmul_type_0_);
  auto* qkv_matmul_1_out = pattern->NewNode(qkv_matmul_1_out_repr())
                               ->assert_is_op_output(matmul_type_0_, "Out")
                               ->assert_var_not_persistable();
  auto qkv_add_0_bias = pattern->NewNode(qkv_add_0_bias_repr())
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->assert_is_persistable_var();
  auto* qkv_add_0 =
      pattern->NewNode(qkv_add_0_repr())->assert_is_op("elementwise_add");
  auto* qkv_add_0_out = pattern->NewNode(qkv_add_0_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_var_not_persistable();
  auto* qkv_add_1 =
      pattern->NewNode(qkv_add_1_repr())->assert_is_op("elementwise_add");
  auto* qkv_add_1_out = pattern->NewNode(qkv_add_1_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_is_op_input("layer_norm", "X")
                            ->assert_var_not_persistable();
  auto* ln_1_bias = pattern->NewNode(ln_1_bias_repr())
                        ->assert_is_op_input("layer_norm", "Bias")
                        ->assert_is_persistable_var();
  auto* ln_1_scale = pattern->NewNode(ln_1_scale_repr())
                         ->assert_is_op_input("layer_norm", "Scale")
                         ->assert_is_persistable_var();
  auto* ln_1 = pattern->NewNode(ln_1_repr())->assert_is_op("layer_norm");
  auto* ln_1_out = pattern->NewNode(ln_1_out_repr())
                       ->assert_is_op_output("layer_norm", "Y")
                       ->assert_var_not_persistable();
  auto* ln_1_mean = pattern->NewNode(ln_1_mean_repr())
                        ->assert_is_op_output("layer_norm", "Mean")
                        ->assert_var_not_persistable();
  auto* ln_1_variance = pattern->NewNode(ln_1_variance_repr())
                            ->assert_is_op_output("layer_norm", "Variance")
                            ->assert_var_not_persistable();
  auto qkv_matmul_2_w = pattern->NewNode(qkv_matmul_2_w_repr())
                            ->assert_is_op_input(matmul_type_0_, "Y")
                            ->assert_is_persistable_var();
  auto* qkv_matmul_2 =
      pattern->NewNode(qkv_matmul_2_repr())->assert_is_op(matmul_type_0_);
  auto* qkv_matmul_2_out = pattern->NewNode(qkv_matmul_2_out_repr())
                               ->assert_is_op_output(matmul_type_0_, "Out")
                               ->assert_var_not_persistable();
  auto qkv_add_2_bias = pattern->NewNode(qkv_add_2_bias_repr())
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->assert_is_persistable_var();
  auto* qkv_add_2 =
      pattern->NewNode(qkv_add_2_repr())->assert_is_op("elementwise_add");
  auto* qkv_add_2_out = pattern->NewNode(qkv_add_2_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_var_not_persistable();
  auto* qkv_act = pattern->NewNode(qkv_act_repr())->assert_is_op(act_type_);
  auto* qkv_act_out = pattern->NewNode(qkv_act_out_repr())
                          ->assert_is_op_output(act_type_, "Out")
                          ->assert_var_not_persistable();
  auto qkv_matmul_3_w = pattern->NewNode(qkv_matmul_3_w_repr())
                            ->assert_is_op_input(matmul_type_0_, "Y")
                            ->assert_is_persistable_var();
  auto* qkv_matmul_3 =
      pattern->NewNode(qkv_matmul_3_repr())->assert_is_op(matmul_type_0_);
  auto* qkv_matmul_3_out = pattern->NewNode(qkv_matmul_3_out_repr())
                               ->assert_is_op_output(matmul_type_0_, "Out")
                               ->assert_var_not_persistable();
  auto qkv_add_3_bias = pattern->NewNode(qkv_add_3_bias_repr())
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->assert_is_persistable_var();
  auto* qkv_add_3 =
      pattern->NewNode(qkv_add_3_repr())->assert_is_op("elementwise_add");
  auto* qkv_add_3_out = pattern->NewNode(qkv_add_3_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_var_not_persistable();
  auto* qkv_add_4 =
      pattern->NewNode(qkv_add_4_repr())->assert_is_op("elementwise_add");
  auto* qkv_add_4_out = pattern->NewNode(qkv_add_4_out_repr())
                            ->assert_is_op_output("elementwise_add", "Out")
                            ->assert_var_not_persistable();
  PDNode* ln_2_bias = nullptr;
  PDNode* ln_2_scale = nullptr;
  PDNode* ln_2 = nullptr;
  PDNode* ln_2_out = nullptr;
  PDNode* ln_2_mean = nullptr;
  PDNode* ln_2_variance = nullptr;
  if (!norm_before_) {
    ln_2_bias = pattern->NewNode(ln_2_bias_repr())
                    ->assert_is_op_input("layer_norm", "Bias")
                    ->assert_is_persistable_var();
    ln_2_scale = pattern->NewNode(ln_2_scale_repr())
                     ->assert_is_op_input("layer_norm", "Scale")
                     ->assert_is_persistable_var();
    ln_2 = pattern->NewNode(ln_2_repr())->assert_is_op("layer_norm");
    ln_2_out = pattern->NewNode(ln_2_out_repr())
                   ->assert_is_op_output("layer_norm", "Y")
                   ->assert_var_not_persistable();
    ln_2_mean = pattern->NewNode(ln_2_mean_repr())
                    ->assert_is_op_output("layer_norm", "Mean")
                    ->assert_var_not_persistable();
    ln_2_variance = pattern->NewNode(ln_2_variance_repr())
                        ->assert_is_op_output("layer_norm", "Variance")
                        ->assert_var_not_persistable();
  }

  // link nodes
  PDNode* q_matmul_x = ln_0_x;
  if (norm_before_) {
    ln_0->LinksFrom({ln_0_x, ln_0_bias, ln_0_scale})
        .LinksTo({ln_0_out, ln_0_mean, ln_0_variance});
    q_matmul_x = ln_0_out;
  }
  q_matmul->LinksFrom({q_matmul_x, q_matmul_w}).LinksTo({q_matmul_out});
  q_add->LinksFrom({q_matmul_out, q_add_bias}).LinksTo({q_add_out});
  q_reshape->LinksFrom({q_add_out}).LinksTo({q_reshape_out});
  q_transpose->LinksFrom({q_reshape_out}).LinksTo({q_transpose_out});
  PDNode* qk_matmul_x = q_transpose_out;
  if (with_q_scale_) {
    q_scale->LinksFrom({q_transpose_out}).LinksTo({q_scale_out});
    qk_matmul_x = q_scale_out;
  }

  k_matmul->LinksFrom({q_matmul_x, k_matmul_w}).LinksTo({k_matmul_out});
  k_add->LinksFrom({k_matmul_out, k_add_bias}).LinksTo({k_add_out});
  k_reshape->LinksFrom({k_add_out}).LinksTo({k_reshape_out});
  k_transpose->LinksFrom({k_reshape_out}).LinksTo({k_transpose_out});

  qk_matmul->LinksFrom({qk_matmul_x, k_transpose_out}).LinksTo({qk_matmul_out});
  PDNode* qk_softmax_x = qk_matmul_out;
  if (with_mask_) {
    qk_add->LinksFrom({qk_matmul_out, qk_add_mask}).LinksTo({qk_add_out});
    qk_softmax_x = qk_add_out;
  }
  qk_softmax->LinksFrom({qk_softmax_x}).LinksTo({qk_softmax_out});

  v_matmul->LinksFrom({q_matmul_x, v_matmul_w}).LinksTo({v_matmul_out});
  v_add->LinksFrom({v_matmul_out, v_add_bias}).LinksTo({v_add_out});
  v_reshape->LinksFrom({v_add_out}).LinksTo({v_reshape_out});
  v_transpose->LinksFrom({v_reshape_out}).LinksTo({v_transpose_out});

  qkv_matmul_0->LinksFrom({qk_softmax_out, v_transpose_out})
      .LinksTo({qkv_matmul_0_out});
  qkv_transpose->LinksFrom({qkv_matmul_0_out}).LinksTo({qkv_transpose_out});
  qkv_reshape->LinksFrom({qkv_transpose_out}).LinksTo({qkv_reshape_out});
  qkv_matmul_1->LinksFrom({qkv_reshape_out, qkv_matmul_1_w})
      .LinksTo({qkv_matmul_1_out});
  qkv_add_0->LinksFrom({qkv_matmul_1_out, qkv_add_0_bias})
      .LinksTo({qkv_add_0_out});
  qkv_add_1->LinksFrom({qkv_add_0_out, q_matmul_x}).LinksTo({qkv_add_1_out});
  ln_1->LinksFrom({qkv_add_1_out, ln_1_bias, ln_1_scale})
      .LinksTo({ln_1_out, ln_1_mean, ln_1_variance});
  qkv_matmul_2->LinksFrom({ln_1_out, qkv_matmul_2_w})
      .LinksTo({qkv_matmul_2_out});
  qkv_add_2->LinksFrom({qkv_matmul_2_out, qkv_add_2_bias})
      .LinksTo({qkv_add_2_out});
  qkv_act->LinksFrom({qkv_add_2_out}).LinksTo({qkv_act_out});
  qkv_matmul_3->LinksFrom({qkv_act_out, qkv_matmul_3_w})
      .LinksTo({qkv_matmul_3_out});
  qkv_add_3->LinksFrom({qkv_matmul_3_out, qkv_add_3_bias})
      .LinksTo({qkv_add_3_out});
  if (norm_before_) {
    qkv_add_4->LinksFrom({qkv_add_3_out, qkv_add_1_out})
        .LinksTo({qkv_add_4_out});
  } else {
    qkv_add_4->LinksFrom({qkv_add_3_out, ln_1_out}).LinksTo({qkv_add_4_out});
    ln_2->LinksFrom({qkv_add_4_out, ln_2_bias, ln_2_scale})
        .LinksTo({ln_2_out, ln_2_mean, ln_2_variance});
  }
}

}  // namespace patterns

/*
step1: fuse single ops to single_encoder_xpu
step2: fuse mutitl single_encoder_xpu to multi_encoder_xpu

1. step1
Origin subgraph:
        ------------ input_variable*
       |             /      |     \
       |            /       |      \
       |      v_matmul  q_matmul  k_matmul
       |           |        |         |
       |           |        |         |
       |        v_add    q_add      add
       |           |        |         |
       |           |        |         |
       |    v_reshape  q_reshape  k_reshape
       |           |        |         |
       |           |        |         |
       |  v_transpose q_transpose k_transpose
       |          |         |         |
       |          |         \         /
       |          |          qk_matmul
       |          |              |
       |          |              |
       |          |           qk_add
       |          |              |
       |          |              |
       |          |         qk_softmax
       |          |              |
       |          |              |
       |          ---------qkv_matmul_0
       |                         |
       |                         |
       |                  qkv_transpose
       |                         |
       |                         |
       |                    qkv_reshape
       |                         |
       |                         |
       |                    qkv_matmul_1
       |                         |
       |                         |
       |                     qkv_add_0
       |                         |
       |                         |
       ----------------------qkv_add_1
                                |
                                |
                            layer_norm_1
                            /       \
                            |       |
                            |  qkv_matmul_2
                            |       |
                            |       |
                            |   qkv_add_2
                            |       |
                            |       |
                            |    qkv_act
                            |       |
                            |       |
                            |  qkv_matmul_3
                            |       |
                            |       |
                            |   qkv_add_3
                            |       |
                            \       /
                            qkv_add_4
                                |
                            layer_norm

Fused subgraph:
                single_encoder_xpu

2. step2
Origin subgraph:
                       ...
                        |
                single_encoder_xpu
                        |
                (single_encoder_xpu)
                        |
                (single_encoder_xpu)
                        |
                       ...
Fused subgraph:
                multi_encoder_xpu
*/
class MultiEncoderXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplySingleEncoderXPUFuse(ir::Graph* graph,
                                const std::string& act_type,
                                const std::string& matmul_type_0,
                                const std::string& matmul_type_1,
                                const std::string& matmul_type_2,
                                bool norm_before,
                                bool with_q_scale,
                                bool with_mask) const;

  bool ApplyMultiEncoderXPUFuse(ir::Graph* graph) const;

  // 1. Transpose q_w, k_w, v_w
  // 2. Concat q_w, k_w, v_w
  // 3. Generate qkv_w_max tensor
  // 4. Quant qkv_w to int16
  void PrepareQKVWeight(const phi::DenseTensor& q_w,
                        const phi::DenseTensor& k_w,
                        const phi::DenseTensor& v_w,
                        phi::DenseTensor* qkv_w,
                        phi::DenseTensor* qkv_w_max) const;

  void ConcatQKVBias(const phi::DenseTensor& q_bias,
                     const phi::DenseTensor& k_bias,
                     const phi::DenseTensor& v_bias,
                     phi::DenseTensor* qkv_bias) const;

  const std::string name_scope_{"multi_encoder_xpu_fuse_pass"};
};

void MultiEncoderXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  std::vector<std::string> act_types{"gelu", "relu"};
  std::vector<std::string> matmul_types_0{"matmul_v2", "matmul", "mul"};
  std::vector<std::string> matmul_types_1{"matmul_v2", "matmul"};
  std::vector<std::string> matmul_types_2{"matmul_v2", "matmul"};
  std::vector<bool> norm_befores{true, false};
  std::vector<bool> with_q_scales{true, false};
  std::vector<bool> with_masks{true, false};
  int single_encoder_fused_counts = 0;
  int multi_encoder_fused_counts = 0;
  for (auto act_type : act_types) {
    for (auto matmul_type_0 : matmul_types_0) {
      for (auto matmul_type_1 : matmul_types_1) {
        for (auto matmul_type_2 : matmul_types_2) {
          for (auto norm_before : norm_befores) {
            for (auto with_q_scale : with_q_scales) {
              for (auto with_mask : with_masks) {
                single_encoder_fused_counts +=
                    ApplySingleEncoderXPUFuse(graph,
                                              act_type,
                                              matmul_type_0,
                                              matmul_type_1,
                                              matmul_type_2,
                                              norm_before,
                                              with_q_scale,
                                              with_mask);
                while (ApplyMultiEncoderXPUFuse(graph)) {
                  multi_encoder_fused_counts++;
                }
              }
            }
          }
        }
      }
    }
  }
  AddStatis(single_encoder_fused_counts);
  AddStatis(multi_encoder_fused_counts);
}

void MultiEncoderXPUFusePass::PrepareQKVWeight(
    const phi::DenseTensor& q_w,
    const phi::DenseTensor& k_w,
    const phi::DenseTensor& v_w,
    phi::DenseTensor* qkv_w,
    phi::DenseTensor* qkv_w_max) const {
  // Transpose
  phi::DenseTensor q_w_trans;
  phi::DenseTensor k_w_trans;
  phi::DenseTensor v_w_trans;
  Transpose2D<float>(q_w, &q_w_trans);
  Transpose2D<float>(k_w, &k_w_trans);
  Transpose2D<float>(v_w, &v_w_trans);

  // Concat
  auto q_w_trans_dims = q_w_trans.dims();
  auto k_w_trans_dims = k_w_trans.dims();
  auto v_w_trans_dims = v_w_trans.dims();
  qkv_w->Resize(DDim({q_w_trans_dims[0] + k_w_trans_dims[0] + v_w_trans_dims[0],
                      q_w_trans_dims[1]}));
  qkv_w->set_type(q_w.type());
  auto* dev_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  int size = q_w.numel();
  auto* qkv_w_data = dev_ctx->Alloc<float>(qkv_w);
  memcpy(qkv_w_data, q_w_trans.data(), size * sizeof(float));
  qkv_w_data += size;
  memcpy(qkv_w_data, k_w_trans.data(), size * sizeof(float));
  qkv_w_data += size;
  memcpy(qkv_w_data, v_w_trans.data(), size * sizeof(float));

  // Quant to int16
  QuantWeight<int16_t>(qkv_w, qkv_w_max, false);
}

void MultiEncoderXPUFusePass::ConcatQKVBias(const phi::DenseTensor& q_bias,
                                            const phi::DenseTensor& k_bias,
                                            const phi::DenseTensor& v_bias,
                                            phi::DenseTensor* qkv_bias) const {
  int q_bias_size = q_bias.numel();
  qkv_bias->Resize(DDim({q_bias_size * 3}));
  qkv_bias->set_type(q_bias.type());
  auto* dev_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  auto* qkv_bias_data = dev_ctx->Alloc<float>(qkv_bias);
  memcpy(qkv_bias_data, q_bias.data(), q_bias_size * sizeof(float));
  qkv_bias_data += q_bias_size;
  memcpy(qkv_bias_data, k_bias.data(), q_bias_size * sizeof(float));
  qkv_bias_data += q_bias_size;
  memcpy(qkv_bias_data, v_bias.data(), q_bias_size * sizeof(float));
}

int MultiEncoderXPUFusePass::ApplySingleEncoderXPUFuse(
    ir::Graph* graph,
    const std::string& act_type,
    const std::string& matmul_type_0,
    const std::string& matmul_type_1,
    const std::string& matmul_type_2,
    bool norm_before,
    bool with_q_scale,
    bool with_mask) const {
  GraphPatternDetector gpd;
  patterns::SingleEncoderXPUPattern pattern(gpd.mutable_pattern(),
                                            name_scope_,
                                            act_type,
                                            matmul_type_0,
                                            matmul_type_1,
                                            matmul_type_2,
                                            norm_before,
                                            with_q_scale,
                                            with_mask);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    // VLOG(4) << "handle MultiEncoderXPUFusePass fuse, step1";
    GET_IR_NODE(ln_0);
    GET_IR_NODE(ln_1);
    GET_IR_NODE(ln_2);
    GET_IR_NODE(q_matmul);
    GET_IR_NODE(q_add);
    GET_IR_NODE(q_reshape);
    GET_IR_NODE(q_transpose);
    GET_IR_NODE(q_scale);
    GET_IR_NODE(k_matmul);
    GET_IR_NODE(k_add);
    GET_IR_NODE(k_reshape);
    GET_IR_NODE(k_transpose);
    GET_IR_NODE(v_matmul);
    GET_IR_NODE(v_add);
    GET_IR_NODE(v_reshape);
    GET_IR_NODE(v_transpose);
    GET_IR_NODE(qk_matmul);
    GET_IR_NODE(qk_add);
    GET_IR_NODE(qk_softmax);
    GET_IR_NODE(qkv_matmul_0);
    GET_IR_NODE(qkv_transpose);
    GET_IR_NODE(qkv_reshape);
    GET_IR_NODE(qkv_matmul_1);
    GET_IR_NODE(qkv_add_0);
    GET_IR_NODE(qkv_add_1);
    GET_IR_NODE(qkv_matmul_2);
    GET_IR_NODE(qkv_add_2);
    GET_IR_NODE(qkv_act);
    GET_IR_NODE(qkv_matmul_3);
    GET_IR_NODE(qkv_add_3);
    GET_IR_NODE(qkv_add_4);
    GET_IR_NODE(ln_0_x);
    GET_IR_NODE(ln_0_bias);
    GET_IR_NODE(ln_0_scale);
    GET_IR_NODE(ln_0_out);
    GET_IR_NODE(ln_0_mean);
    GET_IR_NODE(ln_0_variance);
    GET_IR_NODE(q_matmul_w);
    GET_IR_NODE(q_matmul_out);
    GET_IR_NODE(q_add_bias);
    GET_IR_NODE(q_add_out);
    GET_IR_NODE(q_reshape_out);
    GET_IR_NODE(q_transpose_out);
    GET_IR_NODE(q_scale_out);
    GET_IR_NODE(k_matmul_w);
    GET_IR_NODE(k_matmul_out);
    GET_IR_NODE(k_add_bias);
    GET_IR_NODE(k_add_out);
    GET_IR_NODE(k_reshape_out);
    GET_IR_NODE(k_transpose_out);
    GET_IR_NODE(v_matmul_w);
    GET_IR_NODE(v_matmul_out);
    GET_IR_NODE(v_add_bias);
    GET_IR_NODE(v_add_out);
    GET_IR_NODE(v_reshape_out);
    GET_IR_NODE(v_transpose_out);
    GET_IR_NODE(qk_matmul_out);
    GET_IR_NODE(qk_add_mask);
    GET_IR_NODE(qk_add_out);
    GET_IR_NODE(qk_softmax_out);
    GET_IR_NODE(qkv_matmul_0_out);
    GET_IR_NODE(qkv_transpose_out);
    GET_IR_NODE(qkv_reshape_out);
    GET_IR_NODE(qkv_matmul_1_w);
    GET_IR_NODE(qkv_matmul_1_out);
    GET_IR_NODE(qkv_add_0_bias);
    GET_IR_NODE(qkv_add_0_out);
    GET_IR_NODE(qkv_add_1_out);
    GET_IR_NODE(ln_1_bias);
    GET_IR_NODE(ln_1_scale);
    GET_IR_NODE(ln_1_out);
    GET_IR_NODE(ln_1_mean);
    GET_IR_NODE(ln_1_variance);
    GET_IR_NODE(qkv_matmul_2_w);
    GET_IR_NODE(qkv_matmul_2_out);
    GET_IR_NODE(qkv_add_2_bias);
    GET_IR_NODE(qkv_add_2_out);
    GET_IR_NODE(qkv_act_out);
    GET_IR_NODE(qkv_matmul_3_w);
    GET_IR_NODE(qkv_matmul_3_out);
    GET_IR_NODE(qkv_add_3_bias);
    GET_IR_NODE(qkv_add_3_out);
    GET_IR_NODE(qkv_add_4_out);
    GET_IR_NODE(ln_2_x);
    GET_IR_NODE(ln_2_bias);
    GET_IR_NODE(ln_2_scale);
    GET_IR_NODE(ln_2_out);
    GET_IR_NODE(ln_2_mean);
    GET_IR_NODE(ln_2_variance);

    auto* block = q_matmul->Op()->Block();
    auto* scope = param_scope();

    // Prepare q,k,v weight
    std::string q_w_name = q_matmul_w->Name();
    std::string k_w_name = k_matmul_w->Name();
    std::string v_w_name = v_matmul_w->Name();
    std::string qkv_w_name = q_w_name + "_" + k_w_name + "_" + v_w_name;
    VarDesc qkv_w_desc(qkv_w_name);
    qkv_w_desc.SetPersistable(true);
    auto* qkv_w = graph->CreateVarNode(&qkv_w_desc);
    auto* qkv_w_var = block->Var(qkv_w_name);
    qkv_w_var->SetPersistable(true);
    std::string qkv_w_max_name = qkv_w_name + "_max";
    VarDesc qkv_w_max_desc(qkv_w_max_name);
    qkv_w_max_desc.SetPersistable(true);
    auto* qkv_w_max = graph->CreateVarNode(&qkv_w_max_desc);
    auto* qkv_w_max_var = block->Var(qkv_w_max_name);
    qkv_w_max_var->SetPersistable(true);
    PrepareQKVWeight(
        scope->FindVar(q_w_name)->Get<phi::DenseTensor>(),
        scope->FindVar(k_w_name)->Get<phi::DenseTensor>(),
        scope->FindVar(v_w_name)->Get<phi::DenseTensor>(),
        scope->Var(qkv_w_name)->GetMutable<phi::DenseTensor>(),
        scope->Var(qkv_w_max_name)->GetMutable<phi::DenseTensor>());

    // Prepare qkv_matmul_1_w, qkv_matmul_2_w, qkv_matmul_3_w
#define PREPARE_QKV_MATMUL_W(idx_)                                            \
  std::string qkv_matmul_##idx_##_w_name = qkv_matmul_##idx_##_w->Name();     \
  std::string qkv_matmul_##idx_##_w_max_name =                                \
      qkv_matmul_##idx_##_w_name + "_max";                                    \
  VarDesc qkv_matmul_##idx_##_w_max_desc(qkv_matmul_##idx_##_w_max_name);     \
  qkv_matmul_##idx_##_w_max_desc.SetPersistable(true);                        \
  auto qkv_matmul_##idx_##_w_max =                                            \
      graph->CreateVarNode(&qkv_matmul_##idx_##_w_max_desc);                  \
  auto qkv_matmul_##idx_##_w_max_var =                                        \
      block->Var(qkv_matmul_##idx_##_w_max_name);                             \
  qkv_matmul_##idx_##_w_max_var->SetPersistable(true);                        \
  auto qkv_matmul_##idx_##_w_max_tensor =                                     \
      scope->Var(qkv_matmul_##idx_##_w_max_name)                              \
          ->GetMutable<phi::DenseTensor>();                                   \
  auto qkv_matmul_##idx_##_w_tensor =                                         \
      scope->Var(qkv_matmul_##idx_##_w_name)->GetMutable<phi::DenseTensor>(); \
  QuantWeight<int16_t>(                                                       \
      qkv_matmul_##idx_##_w_tensor, qkv_matmul_##idx_##_w_max_tensor, true);
    PREPARE_QKV_MATMUL_W(1);
    PREPARE_QKV_MATMUL_W(2);
    PREPARE_QKV_MATMUL_W(3);
#undef PREPARE_QKV_MATMUL_W

    // Concat q_add_bias, k_add_bias, v_add_bias
    std::string q_add_bias_name = q_add_bias->Name();
    std::string k_add_bias_name = k_add_bias->Name();
    std::string v_add_bias_name = v_add_bias->Name();
    std::string qkv_add_bias_name =
        q_add_bias_name + "_" + k_add_bias_name + "_" + v_add_bias_name;
    VarDesc qkv_add_bias_desc(qkv_add_bias_name);
    qkv_add_bias_desc.SetPersistable(true);
    auto* qkv_add_bias = graph->CreateVarNode(&qkv_add_bias_desc);
    auto* qkv_add_bias_var = block->Var(qkv_add_bias_name);
    qkv_add_bias_var->SetPersistable(true);
    ConcatQKVBias(
        scope->FindVar(q_add_bias_name)->Get<phi::DenseTensor>(),
        scope->FindVar(k_add_bias_name)->Get<phi::DenseTensor>(),
        scope->FindVar(v_add_bias_name)->Get<phi::DenseTensor>(),
        scope->Var(qkv_add_bias_name)->GetMutable<phi::DenseTensor>());

    // Generate single_encoder_xpu op
    framework::OpDesc op_desc(block);
    op_desc.SetType("single_encoder_xpu");
    op_desc.SetInput("x", {ln_0_x->Name()});
    op_desc.SetInput("fc_weight",
                     {qkv_w_name,
                      qkv_matmul_1_w_name,
                      qkv_matmul_2_w_name,
                      qkv_matmul_3_w_name});
    op_desc.SetInput("fc_weight_max",
                     {qkv_w_max_name,
                      qkv_matmul_1_w_max_name,
                      qkv_matmul_2_w_max_name,
                      qkv_matmul_3_w_max_name});
    op_desc.SetInput("fc_bias",
                     {qkv_add_bias_name,
                      qkv_add_0_bias->Name(),
                      qkv_add_2_bias->Name(),
                      qkv_add_3_bias->Name()});
    if (norm_before) {
      op_desc.SetInput("ln_scale", {ln_0_scale->Name(), ln_1_scale->Name()});
      op_desc.SetInput("ln_bias", {ln_0_bias->Name(), ln_1_bias->Name()});
    } else {
      op_desc.SetInput("ln_scale", {ln_1_scale->Name(), ln_2_scale->Name()});
      op_desc.SetInput("ln_bias", {ln_1_bias->Name(), ln_2_bias->Name()});
    }
    if (with_mask) {
      op_desc.SetInput("mask", {qk_add_mask->Name()});
    }
    op_desc.SetAttr("norm_before", norm_before);
    op_desc.SetAttr("hidden_dim",
                    static_cast<int>(q_matmul_w->Var()->GetShape()[0]));
    auto q_reshape_shape =
        PADDLE_GET_CONST(std::vector<int>, q_reshape->Op()->GetAttr("shape"));
    op_desc.SetAttr("head_num", q_reshape_shape[2]);
    op_desc.SetAttr("size_per_head", q_reshape_shape[3]);
    auto qkv_matmul_2_w_shape = qkv_matmul_2_w->Var()->GetShape();
    op_desc.SetAttr(
        "ffn_hidden_dim_scale",
        static_cast<int>(qkv_matmul_2_w_shape[1] / qkv_matmul_2_w_shape[0]));
    op_desc.SetAttr("act_type", ConvertActivationType(act_type));
    op_desc.SetAttr("relative_type", static_cast<int>(0));
    if (norm_before) {
      op_desc.SetOutput("out", {qkv_add_4_out->Name()});
    } else {
      op_desc.SetOutput("out", {ln_2_out->Name()});
    }
    auto* single_encoder_xpu = graph->CreateOpNode(&op_desc);
    // Link nodes
    SAFE_IR_NODE_LINK_TO(ln_0_x, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_w, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_w_max, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_matmul_1_w, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_matmul_1_w_max, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_matmul_2_w, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_matmul_2_w_max, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_matmul_3_w, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_matmul_3_w_max, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_add_bias, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_add_0_bias, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_add_2_bias, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qkv_add_3_bias, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(ln_0_scale, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(ln_0_bias, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(ln_1_scale, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(ln_1_bias, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(ln_2_scale, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(ln_2_bias, single_encoder_xpu);
    SAFE_IR_NODE_LINK_TO(qk_add_mask, single_encoder_xpu);
    if (norm_before) {
      SAFE_IR_NODE_LINK_TO(single_encoder_xpu, qkv_add_4_out);
    } else {
      SAFE_IR_NODE_LINK_TO(single_encoder_xpu, ln_2_out);
    }

    // Delete nodes
    std::unordered_set<const Node*> delete_nodes{ln_1,
                                                 q_matmul,
                                                 q_add,
                                                 q_reshape,
                                                 q_transpose,
                                                 k_matmul,
                                                 k_add,
                                                 k_reshape,
                                                 k_transpose,
                                                 v_matmul,
                                                 v_add,
                                                 v_reshape,
                                                 v_transpose,
                                                 qk_matmul,
                                                 qk_softmax,
                                                 qkv_matmul_0,
                                                 qkv_transpose,
                                                 qkv_reshape,
                                                 qkv_matmul_1,
                                                 qkv_add_0,
                                                 qkv_add_1,
                                                 qkv_matmul_2,
                                                 qkv_add_2,
                                                 qkv_act,
                                                 qkv_matmul_3,
                                                 qkv_add_3,
                                                 qkv_add_4,
                                                 q_matmul_w,
                                                 q_matmul_out,
                                                 q_add_out,
                                                 q_reshape_out,
                                                 q_transpose_out,
                                                 k_matmul_w,
                                                 k_matmul_out,
                                                 k_add_out,
                                                 k_reshape_out,
                                                 k_transpose_out,
                                                 v_matmul_w,
                                                 v_matmul_out,
                                                 v_add_out,
                                                 v_reshape_out,
                                                 v_transpose_out,
                                                 qk_matmul_out,
                                                 qk_softmax_out,
                                                 qkv_matmul_0_out,
                                                 qkv_transpose_out,
                                                 qkv_reshape_out,
                                                 qkv_matmul_1_out,
                                                 qkv_add_0_out,
                                                 qkv_add_1_out,
                                                 ln_1_out,
                                                 ln_1_mean,
                                                 ln_1_variance,
                                                 qkv_matmul_2_out,
                                                 qkv_add_2_out,
                                                 qkv_act_out,
                                                 qkv_matmul_3_out,
                                                 qkv_add_3_out};
    if (norm_before) {
      delete_nodes.insert(ln_0);
      delete_nodes.insert(ln_0_mean);
      delete_nodes.insert(ln_0_variance);
      delete_nodes.insert(ln_0_out);
    } else {
      delete_nodes.insert(qkv_add_4_out);
      delete_nodes.insert(ln_2);
      delete_nodes.insert(ln_2_mean);
      delete_nodes.insert(ln_2_variance);
    }
    if (with_q_scale) {
      delete_nodes.insert(q_scale);
      delete_nodes.insert(q_scale_out);
    }
    if (with_mask) {
      delete_nodes.insert(qk_add);
      delete_nodes.insert(qk_add_out);
    }
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

static std::vector<Node*> GetSingleEncoders(ir::Graph* graph) {
  std::vector<Node*> single_encoders;
  for (auto* node : graph->Nodes()) {
    // Find first singld_encoder_xpu
    if (node->IsVar() || node->Op()->Type() != "single_encoder_xpu") continue;
    bool is_first_encoder = true;
    for (auto* in_node : node->inputs) {
      if (in_node->Var()->Persistable()) continue;
      if (in_node->inputs[0]->Op()->Type() == "single_encoder_xpu") {
        is_first_encoder = false;
        break;
      }
    }
    if (!is_first_encoder) continue;
    // Add continuous single_encoder_xpu
    single_encoders.push_back(node);
    while (true) {
      auto next_ops = single_encoders.back()->outputs[0]->outputs;
      if (next_ops.empty()) break;
      auto next_op_type = next_ops[0]->Op()->Type();
      if (next_op_type != "single_encoder_xpu") break;
      single_encoders.push_back(next_ops[0]);
    }
    break;
  }
  return single_encoders;
}

bool MultiEncoderXPUFusePass::ApplyMultiEncoderXPUFuse(ir::Graph* graph) const {
  auto single_encoders = GetSingleEncoders(graph);
  if (single_encoders.empty()) return false;

  // Prepare inputs/outputs names/nodes
  std::string x_name = single_encoders[0]->Op()->Inputs().at("x")[0];
  std::vector<std::string> arg_names{
      "fc_weight", "fc_weight_max", "fc_bias", "ln_scale", "ln_bias"};
  std::map<std::string, std::vector<std::string>> arg_names_map;
  std::string mask_name = single_encoders[0]->Op()->Inputs().count("mask") > 0
                              ? single_encoders[0]->Op()->Inputs().at("mask")[0]
                              : "";
  std::string out_name = single_encoders.back()->Op()->Outputs().at("out")[0];

  std::vector<Node*> in_nodes;
  for (auto* in_node : single_encoders[0]->inputs) {
    if (in_node->Var()->Name() == x_name ||
        in_node->Var()->Name() == mask_name) {
      in_nodes.push_back(in_node);
    }
  }
  for (auto* single_encoder : single_encoders) {
    auto single_encoder_in_nodes = single_encoder->inputs;
    for (auto arg_name : arg_names) {
      auto var_names = single_encoder->Op()->Inputs().at(arg_name);
      for (auto var_name : var_names) {
        arg_names_map[arg_name].push_back(var_name);
        for (auto in_node : single_encoder_in_nodes) {
          if (in_node->Var()->Name() == var_name) {
            in_nodes.push_back(in_node);
          }
        }
      }
    }
  }

  std::vector<Node*> out_nodes;
  for (auto* out_node : single_encoders.back()->outputs) {
    if (out_node->Var()->Name() == out_name) {
      out_nodes.push_back(out_node);
      break;
    }
  }

  auto* block = single_encoders[0]->Op()->Block();
  auto* scope = param_scope();
  // Create x_fp16 variable/mode/tensor
  std::string x_fp16_name = x_name + "_fp16";
  VarDesc x_fp16_desc(x_fp16_name);
  auto* x_fp16 = graph->CreateVarNode(&x_fp16_desc);
  block->Var(x_fp16_name);
  scope->Var(x_fp16_name)->GetMutable<phi::DenseTensor>();
  out_nodes.push_back(x_fp16);
  // Create out_fp16 variable/mode/tensor
  std::string out_fp16_name = out_name + "_fp16";
  VarDesc out_fp16_desc(out_fp16_name);
  auto* out_fp16 = graph->CreateVarNode(&out_fp16_desc);
  block->Var(out_fp16_name);
  scope->Var(out_fp16_name)->GetMutable<phi::DenseTensor>();
  out_nodes.push_back(out_fp16);

  // Generate multi_encoder_xpu op
  framework::OpDesc op_desc(block);
  op_desc.SetType("multi_encoder_xpu");
  op_desc.SetInput("x", {x_name});
  for (auto arg_name : arg_names) {
    op_desc.SetInput(arg_name, arg_names_map[arg_name]);
  }
  if (!mask_name.empty()) {
    op_desc.SetInput("mask", {mask_name});
  }
  op_desc.SetAttr("layer_num", static_cast<int>(single_encoders.size()));
  op_desc.SetAttr(
      "norm_before",
      PADDLE_GET_CONST(bool, single_encoders[0]->Op()->GetAttr("norm_before")));
  for (auto attr_name : {"hidden_dim",
                         "head_num",
                         "size_per_head",
                         "ffn_hidden_dim_scale",
                         "act_type",
                         "relative_type"}) {
    op_desc.SetAttr(
        attr_name,
        PADDLE_GET_CONST(int, single_encoders[0]->Op()->GetAttr(attr_name)));
  }
  op_desc.SetAttr("slice_idx", static_cast<int>(-1));
  op_desc.SetOutput("out", {out_name});
  op_desc.SetOutput("x_fp16", {x_fp16_name});
  op_desc.SetOutput("out_fp16", {out_fp16_name});
  auto* multi_encoder_xpu = graph->CreateOpNode(&op_desc);
  for (auto* in_node : in_nodes) {
    IR_NODE_LINK_TO(in_node, multi_encoder_xpu);
  }
  for (auto* out_node : out_nodes) {
    IR_NODE_LINK_TO(multi_encoder_xpu, out_node);
  }

  // delete useless node
  std::unordered_set<const Node*> delete_nodes(single_encoders.begin(),
                                               single_encoders.end());
  for (int i = 0; i < static_cast<int>(single_encoders.size()) - 1; i++) {
    std::string out_name = single_encoders[i]->Op()->Outputs().at("out")[0];
    for (auto* out_node : single_encoders[i]->outputs) {
      if (out_node->Var()->Name() != out_name) {
        delete_nodes.insert(out_node);
      }
    }
  }
  GraphSafeRemoveNodes(graph, delete_nodes);

  return true;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_encoder_xpu_fuse_pass,
              paddle::framework::ir::MultiEncoderXPUFusePass);

REGISTER_PASS_CAPABILITY(multi_encoder_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "multi_encoder_xpu", 0));
