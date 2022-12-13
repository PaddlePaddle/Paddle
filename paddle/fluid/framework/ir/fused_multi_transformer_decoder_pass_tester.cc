/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/fused_multi_transformer_encoder_pass.h"  // NOLINT
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

void AddVarToScope(Scope* param_scope,
                   const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<phi::DenseTensor>();
  tensor->Resize(dims);
  tensor->mutable_data<float>(platform::CPUPlace());
}

Scope* CreateParamScope() {
  auto param_scope = new Scope();

  // MHA: pre Layer Norm
  AddVarToScope(param_scope, "ln_scale", {1024});
  AddVarToScope(param_scope, "ln_bias", {1024});

  // MHA: QKV fc
  AddVarToScope(param_scope, "weights0", {1024, 1024});
  AddVarToScope(param_scope, "weights1", {1024, 1024});
  AddVarToScope(param_scope, "weights2", {1024, 1024});
  AddVarToScope(param_scope, "bias_0", {1024});
  AddVarToScope(param_scope, "bias_1", {1024});
  AddVarToScope(param_scope, "bias_2", {1024});

  // MHA: QK bias
  AddVarToScope(param_scope, "biasqk", {1024});

  // MHA: out Linear
  AddVarToScope(param_scope, "weights_l", {1024, 1024});
  AddVarToScope(param_scope, "bias_l", {1024});

  // MHA: pre Layer Norm
  AddVarToScope(param_scope, "ffn_ln_scale", {1024});
  AddVarToScope(param_scope, "ffn_ln_bias", {1024});

  // FFN: fc1 -> (gelu) -> fc2
  AddVarToScope(param_scope, "ffn_weights0", {1024, 4096});
  AddVarToScope(param_scope, "ffn_weights1", {4096, 1024});
  AddVarToScope(param_scope, "ffn_bias_0", {4096});
  AddVarToScope(param_scope, "ffn_bias_1", {1024});

  return param_scope;
}

TEST(FusedMultiTransformerDecoderPass, basic) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (x, ln_scale, ln_bias)           layer_norm       -> layer_norm_out
  // (layer_norm_out, weights_0)      matmul_v2        -> matmul_out0
  // (layer_norm_out, weights_1)      matmul_v2        -> matmul_out1
  // (layer_norm_out, weights_2)      matmul_v2        -> matmul_out2
  // (matmul_out0, bias_0)            elementwise_add  -> eltadd_0
  // (matmul_out1, bias_1)            elementwise_add  -> eltadd_1
  // (matmul_out2, bias_2)            elementwise_add  -> eltadd_2
  // (eltadd_0)                       reshape2         -> reshape_0
  // (eltadd_1)                       reshape2         -> reshape_1
  // (eltadd_2)                       reshape2         -> reshape_2
  // (reshape_0)                      transpose2       -> transpose_0
  // (reshape_1)                      transpose2       -> transpose_1
  // (reshape_2)                      transpose2       -> transpose_2
  // (transpose_1)                    concat           -> concat_0
  // (transpose_2)                    concat           -> concat_2
  // (concat_0)                       assign           -> assign_0
  // (concat_1)                       assign           -> assign_2
  // (transpose_0, transpose_1)       matmul           -> matmul_qk
  // (matmul_qk, bias_qk)             elementwise_add  -> eltadd_qk
  // (eltadd_qk)                      softmax          -> softmax_qk
  // (softmax_qk, transpose_2)        matmul_v2        -> matmul_qkv
  // (matmul_qkv)                     transpose        -> transpose_qkv
  // (transpose_qkv)                  reshape          -> reshape_qkv
  // (reshape_qkv)                    matmul_v2        -> matmul_linear
  // (matmul_linear)                  elementwise_add  -> eltadd_linear
  // (eltadd_out)                     elementwise_add  -> attention_out
  //
  // (attention_out, scale, bias)     layer_norm       -> ffn_layer_norm_out
  // (layer_norm_out, ffn_matmul0_w)  matmul_v2        -> ffn_matmul0
  // (ffn_matmul0, ffn_bias0)         elementwise_add  -> ffn_eltadd0
  // (ffn_eltadd0)                    gelu             -> ffn_gelu
  // (ffn_gelu)                       matmul_v2        -> ffn_matmul1
  // (ffn_matmul1, ffn_bias1)         elementwise_add  -> ffn_eltadd1
  // (attention_out, ffn_eltadd1)     elementwise_add  -> ffn_output

  Layers layers;
  // MHA: pre LayerNorm
  auto* x = layers.data("x", {1, 128, 1024});
  auto* ln_scale = layers.data("ln_scale", {1024}, true);
  auto* ln_bias = layers.data("ln_bias", {1024}, true);
  auto* ln_out = layers.layer_norm(x, ln_scale, ln_bias)[0];

  // MHA: QKV fc
  auto* weights_0 = layers.data("weights0", {1024, 1024}, true);
  auto* weights_1 = layers.data("weights1", {1024, 1024}, true);
  auto* weights_2 = layers.data("weights2", {1024, 1024}, true);
  auto* matmul_out_0 =
      layers.matmul_v2(ln_out, weights_0, nullptr, false, true);
  auto* matmul_out_1 =
      layers.matmul_v2(ln_out, weights_1, nullptr, false, true);
  auto* matmul_out_2 =
      layers.matmul_v2(ln_out, weights_2, nullptr, false, true);

  auto* b0 = layers.data("bias_0", {1024}, true);
  auto* b1 = layers.data("bias_1", {1024}, true);
  auto* b2 = layers.data("bias_2", {1024}, true);
  auto* elementwise_out_0 =
      layers.elementwise_add(matmul_out_0, b0, nullptr, 2);
  auto* elementwise_out_1 =
      layers.elementwise_add(matmul_out_1, b1, nullptr, 2);
  auto* elementwise_out_2 =
      layers.elementwise_add(matmul_out_2, b2, nullptr, 2);

  std::vector<int> shape = {1, 128, 16, 64};
  auto* reshape_0 = layers.reshape2(elementwise_out_0, shape, true);
  auto* reshape_1 = layers.reshape2(elementwise_out_1, shape, true);
  auto* reshape_2 = layers.reshape2(elementwise_out_2, shape, true);

  std::vector<int> axis = {0, 2, 1, 3};
  auto* transpose_0 = layers.transpose2(reshape_0, axis, true);
  auto* transpose_1 = layers.transpose2(reshape_1, axis, true);
  auto* transpose_2 = layers.transpose2(reshape_2, axis, true);

  auto* cache_k = layers.data("cache_k", {1, 16, 128, 64});
  auto* cache_v = layers.data("cache_v", {1, 16, 128, 64});
  auto* concat_k = layers.concat({cache_k, transpose_1}, 2);
  auto* concat_v = layers.concat({cache_v, transpose_2}, 2);
  layers.assign(concat_k);
  layers.assign(concat_v);

  // MHA: QK matmul
  auto* matmul_qk = layers.matmul(transpose_0, concat_k, nullptr, false, true);

  auto* bqk = layers.data("biasqk", {1, 12, 128, 128}, true);
  auto* elementwise_qk = layers.elementwise_add(matmul_qk, bqk);
  auto* softmax_qk = layers.softmax(elementwise_qk, -1);

  // MHA: QKV matmul
  auto* matmul_qkv = layers.matmul_v2(softmax_qk, concat_v);

  auto* transpose_qkv = layers.transpose2(matmul_qkv, {0, 2, 1, 3}, true);
  auto* reshape_qkv_out = layers.reshape2(transpose_qkv, {1, 128, 1024}, true);

  // MHA: out Linear
  auto* weights_l = layers.data("weights_l", {1024, 1024}, true);
  auto* bias_l = layers.data("weightsl", {1024, 1024}, true);
  auto* linear_matmut_out =
      layers.matmul_v2(reshape_qkv_out, weights_l, nullptr, false, true);
  auto* linear_eltadd_out =
      layers.elementwise_add(linear_matmut_out, bias_l, nullptr, 2);

  auto* attention_out = layers.elementwise_add(x, linear_eltadd_out);

  // FFN: pre LayerNorm
  auto* ffn_ln_scale = layers.data("ffn_ln_scale", {1024}, true);
  auto* ffn_ln_bias = layers.data("ffn_ln_bias", {1024}, true);
  auto* ffn_ln_out =
      layers.layer_norm(attention_out, ffn_ln_scale, ffn_ln_bias)[0];

  // FFN: fc1 -> gelu -> fc2
  auto* ffn_weights0 = layers.data("ffn_weights0", {1024, 4096}, true);
  auto* ffn_weights1 = layers.data("ffn_weights1", {4096, 1024}, true);
  auto* ffn_bias0 = layers.data("ffn_bias0", {4096}, true);
  auto* ffn_bias1 = layers.data("ffn_bias1", {1024}, true);
  auto* ffn_matmul0_out =
      layers.matmul_v2(ffn_ln_out, ffn_weights0, nullptr, false, true);
  auto* ffn_eltadd0_out =
      layers.elementwise_add(ffn_matmul0_out, ffn_bias0, nullptr, 2);
  auto* ffn_gelu_out = layers.gelu(ffn_eltadd0_out);
  auto* ffn_matmul1_out =
      layers.matmul_v2(ffn_gelu_out, ffn_weights1, nullptr, false, true);
  auto* ffn_eltadd1_out =
      layers.elementwise_add(ffn_matmul1_out, ffn_bias1, nullptr, 2);

  layers.elementwise_add(attention_out, ffn_eltadd1_out);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());
  graph->Set("enable_int8", new bool(false));

  auto pass =
      PassRegistry::Instance().Get("fused_multi_transformer_decoder_pass");
  if (pass.get() == nullptr)
    LOG(INFO) << "get fused_multi_transformer_decoder_pass failed";
  int num_nodes_before = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  VLOG(3) << DebugString(graph);
  int num_fused_nodes_after = GetNumOpNodes(graph, "fused_multi_transformer");

  PADDLE_ENFORCE_EQ(num_nodes_before,
                    num_nodes_after + 60,
                    platform::errors::InvalidArgument(
                        "After the fused_multi_transformer_decoder_pass, The "
                        "node num in graph "
                        "should be %d, but the result is %d",
                        num_nodes_before - 60,
                        num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fused_nodes_after,
                    1,
                    platform::errors::InvalidArgument(
                        "After the fused_multi_transformer_decoder pass, "
                        "there should be one fused_multi_transformer op, "
                        "but the result is %d",
                        num_fused_nodes_after));
}

TEST(FusedMultiTransformerDecoderPass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("fused_multi_transformer_decoder_pass"));
}

TEST(FusedMultiTransformerDecoderFuseQKVPass, basic) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (x, ln_scale, ln_bias)           layer_norm       -> layer_norm_out
  // (layer_norm_out, weights_0)      matmul_v2        -> matmul_out0
  // (matmul_out0, bias_0)            elementwise_add  -> eltadd_0
  // (eltadd_0)                       reshape2         -> reshape_0
  // (reshape_0)                      transpose2       -> transpose_0
  // (transpose_0)                    split            -> split_q, split_k,
  // split_v (split_k)                concat           -> concat_k
  // (split_v)                        concat           -> concat_v
  // (concat_k)                       assign           -> assign_k
  // (concat_v)                       assign           -> assign_v
  // (split_q, split_k)               matmul_v2        -> matmul_qk
  // (matmul_qk)                      scale            -> scale_qk
  // (scale_qk, bias_qk)              elementwise_add  -> eltadd_qk
  // (eltadd_qk)                      softmax          -> softmax_qk
  // (softmax_qk, transpose_2)        matmul_v2        -> matmul_qkv
  // (matmul_qkv)                     transpose        -> transpose_qkv
  // (transpose_qkv)                  reshape          -> reshape_qkv
  // (reshape_qkv)                    matmul_v2        -> matmul_linear
  // (matmul_linear)                  elementwise_add  -> eltadd_linear
  // (eltadd_out)                     elementwise_add  -> attention_out
  //
  // (attention_out, scale, bias)     layer_norm       -> ffn_layer_norm_out
  // (layer_norm_out, ffn_matmul0_w)  matmul_v2        -> ffn_matmul0
  // (ffn_matmul0, ffn_bias0)         elementwise_add  -> ffn_eltadd0
  // (ffn_eltadd0)                    gelu             -> ffn_gelu
  // (ffn_gelu)                       matmul_v2        -> ffn_matmul1
  // (ffn_matmul1, ffn_bias1)         elementwise_add  -> ffn_eltadd1
  // (attention_out, ffn_eltadd1)     elementwise_add  -> ffn_output
  //
  // (transpose_1, transpose_2)       while            -> decoder block

  Layers layers;
  // MHA: pre LayerNorm
  auto* x = layers.data("x", {1, 128, 1024});
  auto* ln_scale = layers.data("ln_scale", {1024}, true);
  auto* ln_bias = layers.data("ln_bias", {1024}, true);
  auto* ln_out = layers.layer_norm(x, ln_scale, ln_bias)[0];

  // MHA: QKV fc
  auto* weights_0 = layers.data("weights0", {1024, 3072}, true);
  auto* matmul_out_0 =
      layers.matmul_v2(ln_out, weights_0, nullptr, false, true);

  auto* b0 = layers.data("bias_0", {3072}, true);
  auto* elementwise_out_0 =
      layers.elementwise_add(matmul_out_0, b0, nullptr, 2);

  std::vector<int> shape = {1, 128, 16, 64};
  auto* reshape_0 = layers.reshape2(elementwise_out_0, shape, true);

  std::vector<int> axis = {0, 2, 1, 3};
  auto* transpose_0 = layers.transpose2(reshape_0, axis, true);

  auto split_outs = layers.split(transpose_0, 3, 3);
  auto* split_q = split_outs[0];
  auto* split_k = split_outs[1];
  auto* split_v = split_outs[2];

  auto* cache_k = layers.data("cache_k", {1, 16, 128, 64});
  auto* cache_v = layers.data("cache_v", {1, 16, 128, 64});
  auto* concat_k = layers.concat({cache_k, split_k}, 2);
  auto* concat_v = layers.concat({cache_v, split_v}, 2);
  layers.assign(concat_k);
  layers.assign(concat_v);

  // MHA: QK matmul
  auto* matmul_qk = layers.matmul_v2(split_q, concat_k, nullptr, false, true);
  auto* scale_qk = layers.scale(matmul_qk, 0.125, 0, false);

  auto* bqk = layers.data("biasqk", {1, 12, 128, 128}, true);
  auto* elementwise_qk = layers.elementwise_add(scale_qk, bqk);
  auto* softmax_qk = layers.softmax(elementwise_qk, -1);

  // MHA: QKV matmul
  auto* matmul_qkv = layers.matmul_v2(softmax_qk, concat_v);

  auto* transpose_qkv = layers.transpose2(matmul_qkv, {0, 2, 1, 3}, true);
  auto* reshape_qkv_out = layers.reshape2(transpose_qkv, {1, 128, 1024}, true);

  // MHA: out Linear
  auto* weights_l = layers.data("weights_l", {1024, 1024}, true);
  auto* bias_l = layers.data("weightsl", {1024, 1024}, true);
  auto* linear_matmut_out =
      layers.matmul_v2(reshape_qkv_out, weights_l, nullptr, false, true);
  auto* linear_eltadd_out =
      layers.elementwise_add(linear_matmut_out, bias_l, nullptr, 2);

  auto* attention_out = layers.elementwise_add(x, linear_eltadd_out);

  // FFN: pre LayerNorm
  auto* ffn_ln_scale = layers.data("ffn_ln_scale", {1024}, true);
  auto* ffn_ln_bias = layers.data("ffn_ln_bias", {1024}, true);
  auto* ffn_ln_out =
      layers.layer_norm(attention_out, ffn_ln_scale, ffn_ln_bias)[0];

  // FFN: fc1 -> gelu -> fc2
  auto* ffn_weights0 = layers.data("ffn_weights0", {1024, 4096}, true);
  auto* ffn_weights1 = layers.data("ffn_weights1", {4096, 1024}, true);
  auto* ffn_bias0 = layers.data("ffn_bias0", {4096}, true);
  auto* ffn_bias1 = layers.data("ffn_bias1", {1024}, true);
  auto* ffn_matmul0_out =
      layers.matmul_v2(ffn_ln_out, ffn_weights0, nullptr, false, true);
  auto* ffn_eltadd0_out =
      layers.elementwise_add(ffn_matmul0_out, ffn_bias0, nullptr, 2);
  auto* ffn_gelu_out = layers.gelu(ffn_eltadd0_out);
  auto* ffn_matmul1_out =
      layers.matmul_v2(ffn_gelu_out, ffn_weights1, nullptr, false, true);
  auto* ffn_eltadd1_out =
      layers.elementwise_add(ffn_matmul1_out, ffn_bias1, nullptr, 2);

  layers.elementwise_add(attention_out, ffn_eltadd1_out);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());
  graph->Set("enable_int8", new bool(false));

  auto pass = PassRegistry::Instance().Get(
      "fused_multi_transformer_decoder_fuse_qkv_pass");
  if (pass.get() == nullptr)
    LOG(INFO) << "get fused_multi_transformer_decoder_fuse_qkv_pass failed";
  int num_nodes_before = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  VLOG(3) << DebugString(graph);
  int num_fused_nodes_after = GetNumOpNodes(graph, "fused_multi_transformer");

  PADDLE_ENFORCE_EQ(
      num_nodes_before,
      num_nodes_after + 52,
      platform::errors::InvalidArgument(
          "After the fused_multi_transformer_decoder_fuse_qkv_pass, "
          "The node num in graph should be %d, but the result is %d",
          num_nodes_before - 52,
          num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fused_nodes_after,
                    1,
                    platform::errors::InvalidArgument(
                        "After the fused_multi_transformer_decoder_fuse_qkv "
                        "pass, there should be one fused_multi_transformer "
                        "op, but the result is %d",
                        num_fused_nodes_after));
}

TEST(FusedMultiTransformerDecoderFuseQKVPass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("fused_multi_transformer_decoder_fuse_qkv_pass"));
}

TEST(MultiDevicesFusedMultiTransformerDecoderFuseQKVPass, basic) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (x, ln_scale, ln_bias)           layer_norm       -> layer_norm_out
  // (layer_norm_out)                 c_identity       -> c_identity_out
  // (c_identity_out, weights_0)      matmul_v2        -> matmul_out0
  // (matmul_out0, bias_0)            elementwise_add  -> eltadd_0
  // (eltadd_0)                       reshape2         -> reshape_0
  // (reshape_0)                      transpose2       -> transpose_0
  // (transpose_0)                    split            -> split_q, split_k,
  // split_v (split_k)                        concat           -> concat_k
  // (split_v)                        concat           -> concat_v
  // (concat_k)                       assign           -> assign_k
  // (concat_v)                       assign           -> assign_v
  // (split_q, split_k)               matmul_v2        -> matmul_qk
  // (matmul_qk)                      scale            -> scale_qk
  // (scale_qk, bias_qk)              elementwise_add  -> eltadd_qk
  // (eltadd_qk)                      softmax          -> softmax_qk
  // (softmax_qk, transpose_2)        matmul_v2        -> matmul_qkv
  // (matmul_qkv)                     transpose        -> transpose_qkv
  // (transpose_qkv)                  reshape          -> reshape_qkv
  // (reshape_qkv)                    matmul_v2        -> matmul_linear
  // (matmul_linear)                  c_allreduce_sum  -> c_all_reduce_out
  // (matmul_linear)                  elementwise_add  -> eltadd_linear
  // (eltadd_out)                     elementwise_add  -> attention_out
  //
  // (attention_out, scale, bias)     layer_norm       -> ffn_layer_norm_out
  // (ffn_layer_norm_out)             c_identity       -> ffn_c_identity_out
  // (layer_norm_out, ffn_matmul0_w)  matmul_v2        -> ffn_matmul0
  // (ffn_matmul0, ffn_bias0)         elementwise_add  -> ffn_eltadd0
  // (ffn_eltadd0)                    gelu             -> ffn_gelu
  // (ffn_gelu)                       matmul_v2        -> ffn_matmul1
  // (ffn_matmul1)                    c_allreduce_sum  -> c_allreduce_out
  // (ffn_matmul1, ffn_bias1)         elementwise_add  -> ffn_eltadd1
  // (attention_out, ffn_eltadd1)     elementwise_add  -> ffn_output
  //
  // (transpose_1, transpose_2)       while            -> decoder block

  Layers layers;
  // MHA: pre LayerNorm
  auto* x = layers.data("x", {1, 128, 1024});
  auto* ln_scale = layers.data("ln_scale", {1024}, true);
  auto* ln_bias = layers.data("ln_bias", {1024}, true);
  auto* ln_out = layers.layer_norm(x, ln_scale, ln_bias)[0];
  auto* c_identity_out = layers.c_identity(ln_out);

  // MHA: QKV fc
  auto* weights_0 = layers.data("weights0", {1024, 3072}, true);
  auto* matmul_out_0 =
      layers.matmul_v2(c_identity_out, weights_0, nullptr, false, true);

  auto* b0 = layers.data("bias_0", {3072}, true);
  auto* elementwise_out_0 =
      layers.elementwise_add(matmul_out_0, b0, nullptr, 2);

  std::vector<int> shape = {1, 128, 16, 64};
  auto* reshape_0 = layers.reshape2(elementwise_out_0, shape, true);

  std::vector<int> axis = {0, 2, 1, 3};
  auto* transpose_0 = layers.transpose2(reshape_0, axis, true);

  auto split_outs = layers.split(transpose_0, 3, 3);
  auto* split_q = split_outs[0];
  auto* split_k = split_outs[1];
  auto* split_v = split_outs[2];

  auto* cache_k = layers.data("cache_k", {1, 16, 128, 64});
  auto* cache_v = layers.data("cache_v", {1, 16, 128, 64});
  auto* concat_k = layers.concat({cache_k, split_k}, 2);
  auto* concat_v = layers.concat({cache_v, split_v}, 2);
  layers.assign(concat_k);
  layers.assign(concat_v);

  // MHA: QK matmul
  auto* matmul_qk = layers.matmul_v2(split_q, concat_k, nullptr, false, true);
  auto* scale_qk = layers.scale(matmul_qk, 0.125, 0, false);

  auto* bqk = layers.data("biasqk", {1, 12, 128, 128}, true);
  auto* elementwise_qk = layers.elementwise_add(scale_qk, bqk);
  auto* softmax_qk = layers.softmax(elementwise_qk, -1);

  // MHA: QKV matmul
  auto* matmul_qkv = layers.matmul_v2(softmax_qk, concat_v);

  auto* transpose_qkv = layers.transpose2(matmul_qkv, {0, 2, 1, 3}, true);
  auto* reshape_qkv_out = layers.reshape2(transpose_qkv, {1, 128, 1024}, true);

  // MHA: out Linear
  auto* weights_l = layers.data("weights_l", {1024, 1024}, true);
  auto* bias_l = layers.data("weightsl", {1024, 1024}, true);
  auto* linear_matmut_out =
      layers.matmul_v2(reshape_qkv_out, weights_l, nullptr, false, true);
  auto* c_allreduce_out = layers.c_allreduce_sum(linear_matmut_out);
  auto* linear_eltadd_out =
      layers.elementwise_add(c_allreduce_out, bias_l, nullptr, 2);

  auto* attention_out = layers.elementwise_add(x, linear_eltadd_out);

  // FFN: pre LayerNorm
  auto* ffn_ln_scale = layers.data("ffn_ln_scale", {1024}, true);
  auto* ffn_ln_bias = layers.data("ffn_ln_bias", {1024}, true);
  auto* ffn_ln_out =
      layers.layer_norm(attention_out, ffn_ln_scale, ffn_ln_bias)[0];
  auto* ffn_c_identity_out = layers.c_identity(ffn_ln_out);

  // FFN: fc1 -> gelu -> fc2
  auto* ffn_weights0 = layers.data("ffn_weights0", {1024, 4096}, true);
  auto* ffn_weights1 = layers.data("ffn_weights1", {4096, 1024}, true);
  auto* ffn_bias0 = layers.data("ffn_bias0", {4096}, true);
  auto* ffn_bias1 = layers.data("ffn_bias1", {1024}, true);
  auto* ffn_matmul0_out =
      layers.matmul_v2(ffn_c_identity_out, ffn_weights0, nullptr, false, true);
  auto* ffn_eltadd0_out =
      layers.elementwise_add(ffn_matmul0_out, ffn_bias0, nullptr, 2);
  auto* ffn_gelu_out = layers.gelu(ffn_eltadd0_out);
  auto* ffn_matmul1_out =
      layers.matmul_v2(ffn_gelu_out, ffn_weights1, nullptr, false, true);
  auto* ffn_c_allreduce_out = layers.c_allreduce_sum(ffn_matmul1_out);
  auto* ffn_eltadd1_out =
      layers.elementwise_add(ffn_c_allreduce_out, ffn_bias1, nullptr, 2);

  layers.elementwise_add(attention_out, ffn_eltadd1_out);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());
  graph->Set("enable_int8", new bool(false));

  auto pass = PassRegistry::Instance().Get(
      "multi_devices_fused_multi_transformer_decoder_fuse_qkv_pass");
  if (pass.get() == nullptr)
    LOG(INFO)
        << "get multi_devices_fused_multi_transformer_decoder_fuse_qkv_pass "
           "failed";
  int num_nodes_before = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  VLOG(3) << DebugString(graph);
  int num_fused_nodes_after = GetNumOpNodes(graph, "fused_multi_transformer");

  PADDLE_ENFORCE_EQ(
      num_nodes_before,
      num_nodes_after + 60,
      platform::errors::InvalidArgument(
          "After the fused_multi_transformer_decoder_fuse_qkv_pass, "
          "The node num in graph should be %d, but the result is %d",
          num_nodes_before - 60,
          num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fused_nodes_after,
                    1,
                    platform::errors::InvalidArgument(
                        "After the fused_multi_transformer_decoder_fuse_qkv "
                        "multi-devices pass, there should be one "
                        "fused_multi_transformer op, but the result is %d",
                        num_fused_nodes_after));
}

TEST(MultiDevicesFusedMultiTransformerDecoderFuseQKVPass,
     pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible(
              "multi_devices_fused_multi_transformer_decoder_fuse_qkv_pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fused_multi_transformer_decoder_pass);
USE_PASS(fused_multi_transformer_decoder_fuse_qkv_pass);
USE_PASS(multi_devices_fused_multi_transformer_decoder_fuse_qkv_pass);
