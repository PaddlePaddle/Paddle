// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/kvcache_decoder_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

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



// debuggggg
PDNode* KVCacheDecoderPattern::operator()() {
  auto* input0 = pattern->NewNode(input0_repr());
  // First path with scale
  auto* mul0 = pattern->NewNode(mul0_repr())->assert_is_op("mul");
  auto* mul0_w_var = pattern->NewNode(mul0_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul0_out_var =
      pattern->NewNode(mul0_out_repr())->assert_is_op_output("mul");

  decltype(mul0) eltadd0;
  decltype(mul0) eltadd0_b_var;
  decltype(mul0) eltadd0_out_var;

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
  transpose2_0_out_var->AsIntermediate()->assert_is_op_input("scale");

  auto* scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto* scale_out_var =
      pattern->NewNode(scale_out_repr())->assert_is_op_output("scale");
  scale_out_var->AsIntermediate()->assert_is_op_input("matmul");

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
                                ->assert_is_op_output("elementwise_add");
  eltadd_qk_out_var->AsIntermediate()->assert_is_op_input("softmax");

  auto* softmax_qk =
      pattern->NewNode(softmax_qk_repr())->assert_is_op("softmax");
  auto* softmax_qk_out_var =
      pattern->NewNode(softmax_qk_out_repr())->assert_is_op_output("softmax");
  softmax_qk_out_var->AsIntermediate()->assert_is_op_input("matmul");

  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_op("matmul");
  auto* matmul_qkv_out_var =
      pattern->NewNode(matmul_qkv_out_repr())->assert_is_op_output("matmul");
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
  reshape2_qkv_out_var->assert_is_op_input("mul");

  // Second path to matmul
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_op("mul");
  auto* mul1_w_var = pattern->NewNode(mul1_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul1_out_var =
      pattern->NewNode(mul1_out_repr())->assert_is_op_output("mul");

  decltype(mul1) eltadd1;
  decltype(mul1) eltadd1_b_var;
  decltype(mul1) eltadd1_out_var;

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
  transpose2_1_out_var->AsIntermediate()->assert_is_op_input(
      "concat");  // link to matmul qk

  // Third path to matmul
  auto* mul2 = pattern->NewNode(mul2_repr())->assert_is_op("mul");
  auto* mul2_w_var = pattern->NewNode(mul2_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul2_out_var =
      pattern->NewNode(mul2_out_repr())->assert_is_op_output("mul");

  decltype(mul2) eltadd2;
  decltype(mul2) eltadd2_b_var;
  decltype(mul2) eltadd2_out_var;

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
  transpose2_2_out_var->AsIntermediate()->assert_is_op_input(
      "concat");  // link to matmul qkv

  
  auto* concat1 =
      pattern->NewNode(concat1_repr())->assert_is_op("concat"); 
  auto* concat1_out_var = pattern->NewNode(concat1_out_repr())
                                   ->assert_is_op_output("concat")
                                   ->assert_is_op_input("matmul")
                                   ->assert_is_op_input("assign");
  auto* assign1 = pattern->NewNode(assign1_repr())->assert_is_op("assign");
  auto* k_cache_w_var = pattern->NewNode(k_cache_w_repr())  
                             ->assert_is_op_output_debug("assign");
  auto* k_cache_r_var = pattern->NewNode(k_cache_r_repr())  
                             ->assert_is_op_input_debug("concat");
  

  auto* concat2 =
      pattern->NewNode(concat2_repr())->assert_is_op("concat"); 
  auto* concat2_out_var = pattern->NewNode(concat2_out_repr())
                                   ->assert_is_op_output("concat")
                                   ->assert_is_op_input("matmul")
                                   ->assert_is_op_input("assign");
  auto* assign2 = pattern->NewNode(assign2_repr())->assert_is_op("assign");
  auto* v_cache_w_var = pattern->NewNode(v_cache_w_repr())  
                             ->assert_is_op_output("assign");
  auto* v_cache_r_var = pattern->NewNode(v_cache_r_repr())                           
                             ->assert_is_op_input("concat");

  // Q path
  mul0->LinksFrom({input0, mul0_w_var}).LinksTo({mul0_out_var});
  eltadd0->LinksFrom({mul0_out_var, eltadd0_b_var}).LinksTo({eltadd0_out_var});

  reshape2_0->LinksFrom({eltadd0_out_var}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  scale->LinksFrom({transpose2_0_out_var}).LinksTo({scale_out_var});
  // K path
  mul1->LinksFrom({input0, mul1_w_var}).LinksTo({mul1_out_var});
  eltadd1->LinksFrom({mul1_out_var, eltadd1_b_var}).LinksTo({eltadd1_out_var});
  reshape2_1->LinksFrom({eltadd1_out_var}).LinksTo({reshape2_1_out_var});
  transpose2_1->LinksFrom({reshape2_1_out_var}).LinksTo({transpose2_1_out_var});
  // compute q*k
  matmul_qk->LinksFrom({scale_out_var, concat1_out_var})
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
  matmul_qkv->LinksFrom({softmax_qk_out_var, concat2_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});

  
  concat1->LinksFrom({transpose2_1_out_var, k_cache_r_var})
      .LinksTo({concat1_out_var});
  assign1->LinksFrom({concat1_out_var}).LinksTo({k_cache_w_var});

  concat2->LinksFrom({transpose2_2_out_var, v_cache_r_var})
      .LinksTo({concat2_out_var});
  assign2->LinksFrom({concat2_out_var}).LinksTo({v_cache_w_var});

  return transpose2_2_out_var;
}

}  // namespace patterns


KVCacheDecoderFusePass::KVCacheDecoderFusePass() {
  AddOpCompat(OpCompat("mul"))
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

  // QK (B, H, S, N)*(B, H, S, N) -> (B, H, S, S)
  // QKV (B, H, S, S)*(B, H, S, N) -> (B, H, S, N)
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
      .IsNumEQ(1.0f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")  // QK(true) QKV(false)
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

  AddOpCompat(OpCompat("concat"))
      .AddInput("X")  // Input("X"): vector<tensors>
      .End()
      .AddInput("AxisTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .End();
//   AddOpCompat(OpCompat("fill_constant_batch_size_like"))
//       .AddInput("Input")  // Input("X"): vector<tensors>
//       .End()
//       .AddOutput("Out")
//       .IsTensor()
//       .End()
//       .AddAttr("value")
//       .IsNumEQ(0)
//       .End()
//       .AddAttr("shape")
//       .End()
//       .AddAttr("force_cpu")
//       .End()
//       .AddAttr("str_value")
//       .End()
//       .AddAttr("input_dim_idx")
//       .End();
}

int KVCacheDecoderFusePass::BuildFusion(Graph* graph,
                                             const std::string& name_scope,
                                             Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::KVCacheDecoderPattern multihead_pattern(pattern, name_scope);

  multihead_pattern();
  // Create New OpDesc
  auto fuse_creater = [&](
      Node* input0, Node* mul0, Node* mul1, Node* mul2, Node* mul0_out,
      Node* mul1_out, Node* mul2_out, Node* mul0_w, Node* mul1_w, Node* mul2_w,
      Node* eltadd0_b, Node* eltadd1_b, Node* eltadd2_b, Node* eltadd_qk_b,
      Node* reshape2, Node* reshape2_qkv_out, Node* scale, Node* scale_out,
      Node* softmax_qk, Node* eltadd0, Node* eltadd1, Node* eltadd2,
      Node* matmul_qk, Node* reshape2_qkv,
      Node* k_cache, Node* v_cache, Node* time_step) {
    auto scale_attr = BOOST_GET_CONST(float, scale->Op()->GetAttr("scale"));

    // mul (B * S * Hidden) x (Hidden * 3 * N * H) = (B * S * 3 * N * H)
    // bias (B * S * 3 * N * H) + bias (3 * N * H)
    // Transpose (B * S * 3 * N * H) -> (3 * B * N * S * H)
    auto* wq_tensor = scope->FindVar(mul0_w->Name())->GetMutable<LoDTensor>();
    auto* bq_tensor =
        scope->FindVar(eltadd0_b->Name())->GetMutable<LoDTensor>();
    if (scope->FindVar(mul1_w->Name()) == nullptr) { // sharing weights
      // reuse the mul0_w and eltadd_0_b nodes for the combined nodes.
      auto* combined_w_desc = mul0_w->Var();
      combined_w_desc->SetShape({wq_tensor->dims()[0], wq_tensor->dims()[2], wq_tensor->dims()[2]});
      combined_w_desc->SetPersistable(true);
  
      auto* combined_bias_desc = eltadd0_b->Var();
      combined_bias_desc->SetShape({bq_tensor->dims()[0]});
      combined_bias_desc->SetPersistable(true);
      auto* mul0_out_desc = mul0_out->Var();
      auto input_shape = input0->Var()->GetShape();
      mul0_out_desc->SetShape(
          {input_shape[0], input_shape[1], wq_tensor->dims()[1]});  // [batch_size, seq_length, 3, hidden_out]
    } else {
        auto* wk_tensor = scope->FindVar(mul1_w->Name())->GetMutable<LoDTensor>();
        auto* wv_tensor = scope->FindVar(mul2_w->Name())->GetMutable<LoDTensor>();

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
            phi::make_ddim({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
        auto combined_bias_dims = phi::make_ddim({3, bq_tensor->dims()[0]});

        // reuse the mul0_w and eltadd_0_b nodes for the combined nodes.
        auto* combined_w_desc = mul0_w->Var();
        combined_w_desc->SetShape({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
        combined_w_desc->SetPersistable(true);

        auto* combined_bias_desc = eltadd0_b->Var();
        combined_bias_desc->SetShape({3, bq_tensor->dims()[0]});
        combined_bias_desc->SetPersistable(true);

        framework::LoDTensor tmp_combined_w_tensor;
        tmp_combined_w_tensor.Resize(combined_w_dims);
        auto* tmp_combined_w_data =
            tmp_combined_w_tensor.mutable_data<float>(platform::CPUPlace());

        std::vector<float*> w_vec = {wq_data, wk_data, wv_data};
        int dims_h = combined_w_dims[0], dims_w = combined_w_dims[2];
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
        memcpy(new_combined_w_data, tmp_combined_w_data,
            sizeof(float) * wq_tensor->numel());

        // scope->EraseVars({mul1_w->Name(), mul2_w->Name()});

        framework::LoDTensor tmp_combined_bias_tensor;
        tmp_combined_bias_tensor.Resize(combined_bias_dims);
        auto* tmp_combined_bias_data =
            tmp_combined_bias_tensor.mutable_data<float>(platform::CPUPlace());

        size_t bias_size = bq_tensor->numel();
        memcpy(tmp_combined_bias_data, bq_data, sizeof(float) * bias_size);
        memcpy(tmp_combined_bias_data + bias_size, bk_data,
            sizeof(float) * bias_size);
        memcpy(tmp_combined_bias_data + 2 * bias_size, bv_data,
            sizeof(float) * bias_size);

        bq_tensor->Resize(combined_bias_dims);
        auto* new_combined_bias_data =
            bq_tensor->mutable_data<float>(platform::CPUPlace());
        memcpy(new_combined_bias_data, tmp_combined_bias_data,
            sizeof(float) * bq_tensor->numel());

        // scope->EraseVars({eltadd1_b->Name(), eltadd2_b->Name()});

    }
    auto reshape_desc = reshape2->Op();
    int head_number =
        BOOST_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape")).at(2);

    auto* block = mul0->Op()->Block();
    for (auto _node : {k_cache, v_cache}) {
        auto* var_ = block->FindVar(_node->Name());
        if (var_ != nullptr) {
          var_->SetDataType(framework::proto::VarType::FP16);
        }
    }

    OpDesc multihead_op_desc(mul0->Op()->Block());
    multihead_op_desc.SetType("transformer_decoder");

    multihead_op_desc.SetInput("Input", {input0->Name()});
    multihead_op_desc.SetInput("W", {mul0_w->Name()});
    multihead_op_desc.SetInput("Bias", {eltadd0_b->Name()});
    multihead_op_desc.SetInput("BiasQK", {eltadd_qk_b->Name()});

    multihead_op_desc.SetInput("KCache", {k_cache->Name()});
    multihead_op_desc.SetInput("VCache", {v_cache->Name()});
    multihead_op_desc.SetInput("TimeStep", {time_step->Name()});
    VLOG(3) << "set timestep input: " << time_step->Name();


    multihead_op_desc.SetOutput("Out", {reshape2_qkv_out->Name()});
    multihead_op_desc.SetAttr("alpha", scale_attr);
    multihead_op_desc.SetAttr("head_number", head_number);

    auto* mul0_op_desc = mul0->Op();

    // all mul op has same input.
    if (multihead_op_desc.HasAttr("Input_scale")) {
      multihead_op_desc.SetAttr("Input_scale",
                                mul0_op_desc->GetAttr("Input_scale"));
    }
    auto* add0_op_desc = eltadd0->Op();
    auto* add1_op_desc = eltadd1->Op();
    auto* add2_op_desc = eltadd2->Op();
    if (add0_op_desc->HasAttr("out_threshold")) {
      auto out_scale0 =
          BOOST_GET_CONST(float, add0_op_desc->GetAttr("out_threshold"));
      auto out_scale1 =
          BOOST_GET_CONST(float, add1_op_desc->GetAttr("out_threshold"));
      auto out_scale2 =
          BOOST_GET_CONST(float, add2_op_desc->GetAttr("out_threshold"));
      auto out_scale_max = std::max(out_scale0, out_scale1);
      out_scale_max = std::max(out_scale_max, out_scale2);
      multihead_op_desc.SetAttr("fc_out_threshold", out_scale_max);
    }

    auto* softmax_qk_op_desc = softmax_qk->Op();
    auto* matmul_qk_op_desc = matmul_qk->Op();
    if (matmul_qk_op_desc->HasAttr("Input_scale")) {
      multihead_op_desc.SetAttr("qkv2context_plugin_int8", true);
      if (softmax_qk_op_desc->HasAttr("out_threshold")) {
        auto qkv_plugin_scale = BOOST_GET_CONST(
            float, softmax_qk_op_desc->GetAttr("out_threshold"));
        multihead_op_desc.SetAttr("dp_probs", qkv_plugin_scale);
      }
    }
    if (reshape2_qkv->Op()->HasAttr("out_threshold")) {
      multihead_op_desc.SetAttr("out_threshold",
                                reshape2_qkv->Op()->GetAttr("out_threshold"));
    }
    auto* multihead = graph->CreateOpNode(&multihead_op_desc);

    IR_NODE_LINK_TO(input0, multihead);
    IR_NODE_LINK_TO(mul0_w, multihead);
    IR_NODE_LINK_TO(eltadd0_b, multihead);
    IR_NODE_LINK_TO(eltadd_qk_b, multihead);

    IR_NODE_LINK_TO(k_cache, multihead);
    IR_NODE_LINK_TO(v_cache, multihead);

    IR_NODE_LINK_TO(multihead, reshape2_qkv_out);
    IR_NODE_LINK_TO(time_step, multihead);
    
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "Op compat check in kvcache_decoder_fuse_pass failed.";
      return;
    }
    // GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input0, input0, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0_out, reshape2_0_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0_out, transpose2_0_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1_out, reshape2_1_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1_out, transpose2_1_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2_out, reshape2_2_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2_out, transpose2_2_out,
                              multihead_pattern);

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
    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk_out, softmax_qk_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv_out, matmul_qkv_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv_out, reshape2_qkv_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv, transpose2_qkv,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv_out, transpose2_qkv_out,
                              multihead_pattern);
    // KVCache
    GET_IR_NODE_FROM_SUBGRAPH(concat1, concat1, multihead_pattern); 
    GET_IR_NODE_FROM_SUBGRAPH(concat2, concat2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat1_out, concat1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat2_out, concat2_out, multihead_pattern);
    
    GET_IR_NODE_FROM_SUBGRAPH(assign1, assign1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(assign2, assign2, multihead_pattern); 
    GET_IR_NODE_FROM_SUBGRAPH(k_cache_w, k_cache_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(k_cache_r, k_cache_r, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(v_cache_w, v_cache_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(v_cache_r, v_cache_r, multihead_pattern);

    Node* time_step = nullptr;
    for (auto* node: g->Nodes()) {
      if (node->IsOp() && node->Name()=="increment") {
           time_step = node->outputs[0];
           VLOG(3) << "find output of increment as time_step";
      }
    }
    
    fuse_creater(input0, mul0, mul1, mul2, mul0_out, mul1_out, mul2_out, mul0_w,
                 mul1_w, mul2_w, eltadd0_b, eltadd1_b, eltadd2_b, eltadd_qk_b,
                 reshape2_0, reshape2_qkv_out, scale, scale_out, softmax_qk,
                 eltadd0, eltadd1, eltadd2, matmul_qk, reshape2_qkv,
                 k_cache_r, v_cache_r, time_step);

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
                                                  scale,
                                                  concat1, concat2, concat1_out, concat2_out, 
                                                  assign1, assign2, 
                                                  k_cache_w, v_cache_w
                                                  });
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void KVCacheDecoderFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the KVCacheDecoder pass, The scope should not be null."));

  int fusion_count = BuildFusion(graph, name_scope_, scope);
  if (fusion_count > 0) {
    graph->Set(kKVCacheDecoderPass, new bool(true));
  }
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(kvcache_decoder_fuse_pass,
              paddle::framework::ir::KVCacheDecoderFusePass);

REGISTER_PASS_CAPABILITY(kvcache_decoder_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("mul", 0)
            .LE("elementwise_add", 1)
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .LE("matmul", 1)
            .EQ("softmax", 0));
