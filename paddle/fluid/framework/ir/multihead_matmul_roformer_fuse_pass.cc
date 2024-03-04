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

#include "paddle/fluid/framework/ir/multihead_matmul_roformer_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

static void ReplaceOutputVar(Node* op, Node* old_var, Node* new_var) {
  if (op->IsOp() && op->Op()) {
    new_var->inputs.push_back(op);
    for (size_t i = 0; i < op->outputs.size(); ++i) {
      if (op->outputs[i] == old_var) {
        op->outputs[i] = new_var;
        op->Op()->RenameOutput(old_var->Name(), new_var->Name());
      }
    }
  }
}

PDNode* MultiHeadMatmulRoformerPattern::operator()() {
  std::unordered_set<std::string> matmul_ops{"matrix_multiply"};
  auto* input0 = pattern->NewNode(input0_repr());
  input0->assert_is_ops_input(matmul_ops);

  auto* input_cos = pattern->NewNode(input_cos_repr());
  input_cos->assert_is_op_input("elementwise_mul", "Y");
  auto* input_sin = pattern->NewNode(input_sin_repr());
  input_sin->assert_is_op_input("elementwise_mul", "Y");
  // First path with scale
  auto* mul0 = pattern->NewNode(mul0_repr())->assert_is_ops(matmul_ops);
  auto* mul0_w_var = pattern->NewNode(mul0_w_repr())
                         ->AsInput()
                         ->assert_is_ops_input(matmul_ops, "Y");
  auto* mul0_out_var =
      pattern->NewNode(mul0_out_repr())->assert_is_ops_output(matmul_ops);

  decltype(mul0) eltadd0 = nullptr;
  decltype(mul0) eltadd0_b_var = nullptr;
  decltype(mul0) eltadd0_out_var = nullptr;

  mul0_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  eltadd0 = pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");
  eltadd0_b_var = pattern->NewNode(eltadd0_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd0_out_var = pattern->NewNode(eltadd0_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd0_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_0 =
      pattern->NewNode(reshape2_0_repr())->assert_is_op("reshape2");

  auto* reshape2_0_out_var =
      pattern->NewNode(reshape2_0_out_repr())->assert_is_op_output("reshape2");
  reshape2_0_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_0 =
      pattern->NewNode(transpose2_0_repr())->assert_is_op("transpose2");
  auto* transpose2_0_out_var = pattern->NewNode(transpose2_0_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_0_out_var->AsIntermediate()
      ->assert_is_op_input("elementwise_mul", "X")
      ->assert_is_op_input("split", "X");

  auto* eltmul_cos_q =
      pattern->NewNode(eltmul_cos_q_repr())->assert_is_op("elementwise_mul");
  auto* eltmul_cos_q_out_var = pattern->NewNode(eltmul_cos_q_out_repr())
                                   ->assert_is_op_output("elementwise_mul");
  eltmul_cos_q_out_var->AsIntermediate()->assert_is_op_input("elementwise_add",
                                                             "X");

  auto* split_q = pattern->NewNode(split_q_repr())->assert_is_op("split");
  auto* split_q_out_var =
      pattern->NewNode(split_q_out_repr())->assert_is_op_output("split");
  split_q_out_var->AsIntermediate()->assert_is_op_input("concat", "X");
  auto* concat_q = pattern->NewNode(concat_q_repr())->assert_is_op("concat");
  auto* concat_q_out_var =
      pattern->NewNode(concat_q_out_repr())->assert_is_op_output("concat");
  concat_q_out_var->AsIntermediate()->assert_is_op_input("elementwise_mul",
                                                         "X");

  auto* eltmul_sin_q =
      pattern->NewNode(eltmul_sin_q_repr())->assert_is_op("elementwise_mul");
  auto* eltmul_sin_q_out_var = pattern->NewNode(eltmul_sin_q_out_repr())
                                   ->assert_is_op_output("elementwise_mul");
  eltmul_sin_q_out_var->AsIntermediate()->assert_is_op_input("elementwise_add",
                                                             "Y");

  auto* eltadd_q =
      pattern->NewNode(eltadd_q_repr())->assert_is_op("elementwise_add");
  auto* eltadd_q_out_var = pattern->NewNode(eltadd_q_out_repr())
                               ->assert_is_op_output("elementwise_add");
  eltadd_q_out_var->AsIntermediate()->assert_is_op_input("scale");

  auto* matmul_qk =
      pattern->NewNode(matmul_qk_repr())->assert_is_ops(matmul_ops);
  auto* matmul_qk_out_var =
      pattern->NewNode(matmul_qk_out_repr())->assert_is_ops_output(matmul_ops);
  matmul_qk_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  auto* eltadd_qk =
      pattern->NewNode(eltadd_qk_repr())->assert_is_op("elementwise_add");
  auto* eltadd_qk_b_var = pattern->NewNode(eltadd_qk_b_repr())
                              ->AsInput()
                              ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd_qk_out_var = pattern->NewNode(eltadd_qk_out_repr())
                                ->assert_is_op_output("elementwise_add");
  eltadd_qk_out_var->AsIntermediate()->assert_is_op_input("softmax");

  auto* softmax_qk =
      pattern->NewNode(softmax_qk_repr())->assert_is_op("softmax");
  auto* softmax_qk_out_var =
      pattern->NewNode(softmax_qk_out_repr())->assert_is_op_output("softmax");
  softmax_qk_out_var->AsIntermediate()->assert_is_ops_input(matmul_ops);

  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_ops(matmul_ops);
  auto* matmul_qkv_out_var =
      pattern->NewNode(matmul_qkv_out_repr())->assert_is_ops_output(matmul_ops);
  matmul_qkv_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_qkv =
      pattern->NewNode(transpose2_qkv_repr())->assert_is_op("transpose2");
  auto* transpose2_qkv_out_var = pattern->NewNode(transpose2_qkv_out_repr())
                                     ->assert_is_op_output("transpose2");
  transpose2_qkv_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_qkv =
      pattern->NewNode(reshape2_qkv_repr())->assert_is_op("reshape2");
  auto* reshape2_qkv_out_var = pattern->NewNode(reshape2_qkv_out_repr())
                                   ->assert_is_op_output("reshape2");
  reshape2_qkv_out_var->assert_is_ops_input(matmul_ops);

  auto* scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto* scale_out_var =
      pattern->NewNode(scale_out_repr())->assert_is_op_output("scale");
  scale_out_var->AsIntermediate()->assert_is_ops_input(matmul_ops);

  // Second path to matmul
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_ops(matmul_ops);
  auto* mul1_w_var = pattern->NewNode(mul1_w_repr())
                         ->AsInput()
                         ->assert_is_ops_input(matmul_ops, "Y");
  auto* mul1_out_var =
      pattern->NewNode(mul1_out_repr())->assert_is_ops_output(matmul_ops);

  decltype(mul1) eltadd1 = nullptr;
  decltype(mul1) eltadd1_b_var = nullptr;
  decltype(mul1) eltadd1_out_var = nullptr;

  mul1_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd1 = pattern->NewNode(eltadd1_repr())->assert_is_op("elementwise_add");
  eltadd1_b_var = pattern->NewNode(eltadd1_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd1_out_var = pattern->NewNode(eltadd1_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd1_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_1 =
      pattern->NewNode(reshape2_1_repr())->assert_is_op("reshape2");

  auto* reshape2_1_out_var =
      pattern->NewNode(reshape2_1_out_repr())->assert_is_op_output("reshape2");
  reshape2_1_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_1 =
      pattern->NewNode(transpose2_1_repr())->assert_is_op("transpose2");
  auto* transpose2_1_out_var = pattern->NewNode(transpose2_1_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_1_out_var->AsIntermediate()
      ->assert_is_op_input("elementwise_mul", "X")
      ->assert_is_op_input("split", "X");  // link to matmul qk

  auto* eltmul_cos_k =
      pattern->NewNode(eltmul_cos_k_repr())->assert_is_op("elementwise_mul");
  auto* eltmul_cos_k_out_var = pattern->NewNode(eltmul_cos_k_out_repr())
                                   ->assert_is_op_output("elementwise_mul");
  eltmul_cos_k_out_var->AsIntermediate()->assert_is_op_input("elementwise_add",
                                                             "X");

  auto* split_k = pattern->NewNode(split_k_repr())->assert_is_op("split");
  auto* split_k_out_var =
      pattern->NewNode(split_k_out_repr())->assert_is_op_output("split");
  split_k_out_var->AsIntermediate()->assert_is_op_input("concat", "X");
  auto* concat_k = pattern->NewNode(concat_k_repr())->assert_is_op("concat");
  auto* concat_k_out_var =
      pattern->NewNode(concat_k_out_repr())->assert_is_op_output("concat");
  concat_k_out_var->AsIntermediate()->assert_is_op_input("elementwise_mul",
                                                         "X");

  auto* eltmul_sin_k =
      pattern->NewNode(eltmul_sin_k_repr())->assert_is_op("elementwise_mul");
  auto* eltmul_sin_k_out_var = pattern->NewNode(eltmul_sin_k_out_repr())
                                   ->assert_is_op_output("elementwise_mul");
  eltmul_sin_k_out_var->AsIntermediate()->assert_is_op_input("elementwise_add",
                                                             "Y");

  auto* eltadd_k =
      pattern->NewNode(eltadd_k_repr())->assert_is_op("elementwise_add");
  auto* eltadd_k_out_var = pattern->NewNode(eltadd_k_out_repr())
                               ->assert_is_op_output("elementwise_add");
  eltadd_k_out_var->AsIntermediate()->assert_is_ops_input(matmul_ops);

  // Third path to matmul
  auto* mul2 = pattern->NewNode(mul2_repr())->assert_is_ops(matmul_ops);
  auto* mul2_w_var = pattern->NewNode(mul2_w_repr())
                         ->AsInput()
                         ->assert_is_ops_input(matmul_ops, "Y");
  auto* mul2_out_var =
      pattern->NewNode(mul2_out_repr())->assert_is_ops_output(matmul_ops);

  decltype(mul2) eltadd2 = nullptr;
  decltype(mul2) eltadd2_b_var = nullptr;
  decltype(mul2) eltadd2_out_var = nullptr;

  mul2_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd2 = pattern->NewNode(eltadd2_repr())->assert_is_op("elementwise_add");
  eltadd2_b_var = pattern->NewNode(eltadd2_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd2_out_var = pattern->NewNode(eltadd2_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd2_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_2 =
      pattern->NewNode(reshape2_2_repr())->assert_is_op("reshape2");

  auto* reshape2_2_out_var =
      pattern->NewNode(reshape2_2_out_repr())->assert_is_op_output("reshape2");
  reshape2_2_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_2 =
      pattern->NewNode(transpose2_2_repr())->assert_is_op("transpose2");
  auto* transpose2_2_out_var = pattern->NewNode(transpose2_2_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_2_out_var->AsIntermediate()->assert_is_ops_input(
      matmul_ops);  // link to matmul qkv

  // Q path
  mul0->LinksFrom({input0, mul0_w_var}).LinksTo({mul0_out_var});
  eltadd0->LinksFrom({mul0_out_var, eltadd0_b_var}).LinksTo({eltadd0_out_var});

  reshape2_0->LinksFrom({eltadd0_out_var}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  split_q->LinksFrom({transpose2_0_out_var}).LinksTo({split_q_out_var});
  concat_q->LinksFrom({split_q_out_var}).LinksTo({concat_q_out_var});
  eltmul_sin_q->LinksFrom({concat_q_out_var, input_sin})
      .LinksTo({eltmul_sin_q_out_var});
  eltmul_cos_q->LinksFrom({transpose2_0_out_var, input_cos})
      .LinksTo({eltmul_cos_q_out_var});
  eltadd_q->LinksFrom({eltmul_cos_q_out_var, eltmul_sin_q_out_var})
      .LinksTo({eltadd_q_out_var});
  scale->LinksFrom({eltadd_q_out_var}).LinksTo({scale_out_var});
  // K path
  mul1->LinksFrom({input0, mul1_w_var}).LinksTo({mul1_out_var});
  eltadd1->LinksFrom({mul1_out_var, eltadd1_b_var}).LinksTo({eltadd1_out_var});
  reshape2_1->LinksFrom({eltadd1_out_var}).LinksTo({reshape2_1_out_var});
  transpose2_1->LinksFrom({reshape2_1_out_var}).LinksTo({transpose2_1_out_var});
  split_k->LinksFrom({transpose2_1_out_var}).LinksTo({split_k_out_var});
  concat_k->LinksFrom({split_k_out_var}).LinksTo({concat_k_out_var});
  eltmul_sin_k->LinksFrom({concat_k_out_var, input_sin})
      .LinksTo({eltmul_sin_k_out_var});
  eltmul_cos_k->LinksFrom({transpose2_1_out_var, input_cos})
      .LinksTo({eltmul_cos_k_out_var});
  eltadd_k->LinksFrom({eltmul_cos_k_out_var, eltmul_sin_k_out_var})
      .LinksTo({eltadd_k_out_var});

  // compute q*k
  matmul_qk->LinksFrom({scale_out_var, eltadd_k_out_var})
      .LinksTo({matmul_qk_out_var});
  eltadd_qk->LinksFrom({matmul_qk_out_var, eltadd_qk_b_var})
      .LinksTo({eltadd_qk_out_var});
  softmax_qk->LinksFrom({eltadd_qk_out_var}).LinksTo({softmax_qk_out_var});
  // V  path
  mul2->LinksFrom({input0, mul2_w_var}).LinksTo({mul2_out_var});
  eltadd2->LinksFrom({mul2_out_var, eltadd2_b_var}).LinksTo({eltadd2_out_var});
  reshape2_2->LinksFrom({eltadd2_out_var}).LinksTo({reshape2_2_out_var});
  transpose2_2->LinksFrom({reshape2_2_out_var}).LinksTo({transpose2_2_out_var});
  // compute q*k*v
  matmul_qkv->LinksFrom({softmax_qk_out_var, transpose2_2_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});

  return transpose2_2_out_var;
}
}  // namespace patterns

MultiHeadMatmulRoformerFusePass::MultiHeadMatmulRoformerFusePass() {
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

  // QK (B, H, S, N)*(B, H, S, N) -> (B, H, S, S)
  // QKV (B, H, S, S)*(B, H, S, N) -> (B, H, S, N)
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

int MultiHeadMatmulRoformerFusePass::BuildFusion(Graph* graph,
                                                 const std::string& name_scope,
                                                 Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::MultiHeadMatmulRoformerPattern multihead_pattern(pattern,
                                                             name_scope);

  multihead_pattern();
  // Create New OpDesc
  auto fuse_creater = [&](Node* input0,
                          Node* input_cos,
                          Node* input_sin,
                          Node* mul0,
                          Node* mul1,
                          Node* mul2,
                          Node* mul0_out,
                          Node* mul1_out,
                          Node* mul2_out,
                          Node* mul0_w,
                          Node* mul1_w,
                          Node* mul2_w,
                          Node* eltadd0_b,
                          Node* eltadd1_b,
                          Node* eltadd2_b,
                          Node* eltadd_qk_b,
                          Node* reshape2,
                          Node* reshape2_qkv_out,
                          Node* scale,
                          Node* scale_out,
                          Node* matmul_qk) {
    auto scale_attr = PADDLE_GET_CONST(float, scale->Op()->GetAttr("scale"));

    // mul (B * S * Hidden) x (Hidden * 3 * N * H) = (B * S * 3 * N * H)
    // bias (B * S * 3 * N * H) + bias (3 * N * H)
    // Transpose (B * S * 3 * N * H) -> (3 * B * N * S * H)
    auto* wq_tensor =
        scope->FindVar(mul0_w->Name())->GetMutable<phi::DenseTensor>();
    auto* wk_tensor =
        scope->FindVar(mul1_w->Name())->GetMutable<phi::DenseTensor>();
    auto* wv_tensor =
        scope->FindVar(mul2_w->Name())->GetMutable<phi::DenseTensor>();

    auto* bq_tensor =
        scope->FindVar(eltadd0_b->Name())->GetMutable<phi::DenseTensor>();
    auto* bk_tensor =
        scope->FindVar(eltadd1_b->Name())->GetMutable<phi::DenseTensor>();
    auto* bv_tensor =
        scope->FindVar(eltadd2_b->Name())->GetMutable<phi::DenseTensor>();

    auto* wq_data = wq_tensor->mutable_data<float>(platform::CPUPlace());
    auto* wk_data = wk_tensor->mutable_data<float>(platform::CPUPlace());
    auto* wv_data = wv_tensor->mutable_data<float>(platform::CPUPlace());
    auto* bq_data = bq_tensor->mutable_data<float>(platform::CPUPlace());
    auto* bk_data = bk_tensor->mutable_data<float>(platform::CPUPlace());
    auto* bv_data = bv_tensor->mutable_data<float>(platform::CPUPlace());

    auto combined_w_dims =
        common::make_ddim({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
    auto combined_bias_dims = common::make_ddim({3, bq_tensor->dims()[0]});

    // reuse the mul0_w and eltadd_0_b nodes for the combined nodes.
    auto* combined_w_desc = mul0_w->Var();
    combined_w_desc->SetShape({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
    combined_w_desc->SetPersistable(true);

    auto* combined_bias_desc = eltadd0_b->Var();
    combined_bias_desc->SetShape({3, bq_tensor->dims()[0]});
    combined_bias_desc->SetPersistable(true);

    phi::DenseTensor tmp_combined_w_tensor;
    tmp_combined_w_tensor.Resize(combined_w_dims);
    auto* tmp_combined_w_data =
        tmp_combined_w_tensor.mutable_data<float>(platform::CPUPlace());

    std::vector<float*> w_vec = {wq_data, wk_data, wv_data};
    int dims_h = static_cast<int>(combined_w_dims[0]),
        dims_w = static_cast<int>(combined_w_dims[2]);
    // Combine the three fc weights together.
    for (int i = 0; i < dims_h; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < dims_w; k++) {
          int out_index = i * (3 * dims_w) + j * dims_w + k;
          int in_index = i * dims_w + k;
          tmp_combined_w_data[out_index] = w_vec[j][in_index];
        }
      }
    }

    wq_tensor->Resize(combined_w_dims);
    auto* new_combined_w_data =
        wq_tensor->mutable_data<float>(platform::CPUPlace());
    memcpy(new_combined_w_data,
           tmp_combined_w_data,
           sizeof(float) * wq_tensor->numel());

    scope->EraseVars({mul1_w->Name(), mul2_w->Name()});

    phi::DenseTensor tmp_combined_bias_tensor;
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

    auto reshape_desc = reshape2->Op();
    int head_number =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(2);

    OpDesc multihead_op_desc(mul0->Op()->Block());
    multihead_op_desc.SetType("multihead_matmul_roformer");

    multihead_op_desc.SetInput("Input", {input0->Name()});
    multihead_op_desc.SetInput("Input_cos", {input_cos->Name()});
    multihead_op_desc.SetInput("Input_sin", {input_sin->Name()});
    multihead_op_desc.SetInput("W", {mul0_w->Name()});
    multihead_op_desc.SetInput("Bias", {eltadd0_b->Name()});
    multihead_op_desc.SetInput("BiasQK", {eltadd_qk_b->Name()});

    multihead_op_desc.SetOutput("Out", {reshape2_qkv_out->Name()});
    multihead_op_desc.SetAttr("alpha", scale_attr);
    multihead_op_desc.SetAttr("head_number", head_number);

    auto* multihead = graph->CreateOpNode(&multihead_op_desc);

    IR_NODE_LINK_TO(input0, multihead);
    IR_NODE_LINK_TO(input_cos, multihead);
    IR_NODE_LINK_TO(input_sin, multihead);
    IR_NODE_LINK_TO(mul0_w, multihead);
    IR_NODE_LINK_TO(eltadd0_b, multihead);
    IR_NODE_LINK_TO(eltadd_qk_b, multihead);

    IR_NODE_LINK_TO(multihead, reshape2_qkv_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(input0, input0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input_cos, input_cos, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input_sin, input_sin, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0_out, reshape2_0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0_out, transpose2_0_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltmul_cos_q, eltmul_cos_q, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltmul_cos_q_out, eltmul_cos_q_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltmul_sin_q, eltmul_sin_q, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltmul_sin_q_out, eltmul_sin_q_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(split_q, split_q, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(split_q_out, split_q_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat_q, concat_q, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat_q_out, concat_q_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_q, eltadd_q, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_q_out, eltadd_q_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_1_out, reshape2_1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_1_out, transpose2_1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltmul_cos_k, eltmul_cos_k, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltmul_cos_k_out, eltmul_cos_k_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltmul_sin_k, eltmul_sin_k, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        eltmul_sin_k_out, eltmul_sin_k_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(split_k, split_k, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(split_k_out, split_k_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat_k, concat_k, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat_k_out, concat_k_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_k, eltadd_k, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_k_out, eltadd_k_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_2_out, reshape2_2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_2_out, transpose2_2_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, multihead_pattern);
    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0, eltadd0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_b, eltadd0_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_out, eltadd0_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_b, eltadd1_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd2, eltadd2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_b, eltadd2_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_out, eltadd2_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk, matmul_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk_out, matmul_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk, eltadd_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_b, eltadd_qk_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_out, eltadd_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk, softmax_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk_out, softmax_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv_out, matmul_qkv_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv_out, reshape2_qkv_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv, transpose2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv_out, transpose2_qkv_out, multihead_pattern);

    // If weights or biases in qkv's fc are shared by multiple multihead_matmul
    // patterns, we do not support this kind of fusion, this pass will not take
    // effect.
    bool is_fc_params_shared =
        mul0_w->outputs.size() > 1 || mul1_w->outputs.size() > 1 ||
        mul2_w->outputs.size() > 1 || eltadd0_b->outputs.size() > 1 ||
        eltadd1_b->outputs.size() > 1 || eltadd2_b->outputs.size() > 1;
    if (is_fc_params_shared) {
      return;
    }
    fuse_creater(input0,
                 input_cos,
                 input_sin,
                 mul0,
                 mul1,
                 mul2,
                 mul0_out,
                 mul1_out,
                 mul2_out,
                 mul0_w,
                 mul1_w,
                 mul2_w,
                 eltadd0_b,
                 eltadd1_b,
                 eltadd2_b,
                 eltadd_qk_b,
                 reshape2_0,
                 reshape2_qkv_out,
                 scale,
                 scale_out,
                 matmul_qk);

    std::unordered_set<const Node*> marked_nodes({eltadd0,
                                                  eltadd1,
                                                  eltadd2,
                                                  eltadd1_b,
                                                  eltadd2_b,
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
                                                  eltmul_cos_q,
                                                  eltmul_cos_q_out,
                                                  eltmul_sin_q,
                                                  eltmul_sin_q_out,
                                                  eltmul_cos_k,
                                                  eltmul_cos_k_out,
                                                  eltmul_sin_k,
                                                  eltmul_sin_k_out,
                                                  split_q,
                                                  split_q_out,
                                                  concat_q,
                                                  concat_q_out,
                                                  split_k,
                                                  split_k_out,
                                                  concat_k,
                                                  concat_k_out,
                                                  eltadd_q,
                                                  eltadd_q_out,
                                                  eltadd_k,
                                                  eltadd_k_out,
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
                                                  mul0,
                                                  mul1,
                                                  mul2,
                                                  mul0_out,
                                                  mul1_out,
                                                  mul2_out,
                                                  mul1_w,
                                                  mul2_w,
                                                  reshape2_qkv,
                                                  scale});
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void MultiHeadMatmulRoformerFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the multiheadMatmul pass, The scope should not be null."));

  int fusion_count = BuildFusion(graph, name_scope_, scope);
  if (fusion_count > 0) {
    graph->Set(kMultiheadMatmulPass, new bool(true));
  }
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multihead_matmul_roformer_fuse_pass,
              paddle::framework::ir::MultiHeadMatmulRoformerFusePass);

REGISTER_PASS_CAPABILITY(multihead_matmul_roformer_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .EQ("softmax", 0));
