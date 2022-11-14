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

#include "paddle/fluid/framework/ir/fuse_multi_transformer_layer_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"

#define DEF_INPUT_DATA                                                  \
  Layers layers;                                                        \
  int num_layers = 3;                                                   \
  auto* x = layers.data("x", {1, 128, 1024});                           \
  auto* src_mask = layers.data("src_mask", {1, 16, 128, 128});          \
  auto* ln_scale = layers.data("ln_scale", {1024}, true);               \
  auto* ln_bias = layers.data("ln_bias", {1024}, true);                 \
  auto* ffn_ln_scale = layers.data("ffn_ln_scale", {1024}, true);       \
  auto* ffn_ln_bias = layers.data("ffn_ln_bias", {1024}, true);         \
  auto* qkv_w = layers.data("qkv_w", {3, 16, 64, 1024}, true);          \
  auto* out_linear_w = layers.data("out_linear_w", {1024, 1024}, true); \
  auto* ffn1_w = layers.data("ffn1_w", {1024, 4096}, true);             \
  auto* ffn2_w = layers.data("ffn2_w", {4096, 1024}, true);             \
  auto* qkv_bias = layers.data("qkv_bias", {3072}, true);               \
  auto* out_linear_bias = layers.data("out_linear_bias", {1024}, true); \
  auto* ffn1_bias = layers.data("ffn1_bias", {4096}, true);             \
  auto* ffn2_bias = layers.data("ffn2_bias", {1024}, true);

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
  AddVarToScope(param_scope, "ln_scale", {1024});
  AddVarToScope(param_scope, "ln_bias", {1024});
  AddVarToScope(param_scope, "ffn_ln_scale", {1024});
  AddVarToScope(param_scope, "ffn_ln_bias", {1024});

  AddVarToScope(param_scope, "qkv_w", {3, 16, 64, 1024});
  AddVarToScope(param_scope, "out_linear_w", {1024, 1024});
  AddVarToScope(param_scope, "ffn1_w", {1024, 4096});
  AddVarToScope(param_scope, "ffn2_w", {4096, 1024});
  AddVarToScope(param_scope, "qkv_bias", {3072});
  AddVarToScope(param_scope, "out_linear_bias", {1024});
  AddVarToScope(param_scope, "ffn1_bias", {4096});
  AddVarToScope(param_scope, "ffn2_bias", {1024});

  return param_scope;
}
TEST(FuseMultiTransformerLayerPass, encoder_fp) {
  DEF_INPUT_DATA

  // Layers
  for (int i = 0; i < num_layers; ++i) {
    auto* cache_kv = layers.fill_constant_batch_size_like(
        x,
        static_cast<int>(proto::VarType::FP32),
        0,
        1,
        {2, -1, 16, 1024, 64},
        0);
    auto* out = layers.fused_multi_transformer(x,
                                               cache_kv,
                                               src_mask,
                                               qkv_w,
                                               qkv_bias,
                                               out_linear_w,
                                               out_linear_bias,
                                               ffn1_w,
                                               ffn1_bias,
                                               ffn2_w,
                                               ffn2_bias,
                                               ln_scale,
                                               ln_bias,
                                               ffn_ln_scale,
                                               ffn_ln_bias,
                                               0.1,
                                               1e-12);

    x = out;
  }
  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());
  graph->Set(kFusedMultiTransformerEncoderFusionCount, new int(num_layers));

  auto pass = PassRegistry::Instance().Get("fuse_multi_transformer_layer_pass");
  if (pass.get() == nullptr)
    LOG(INFO) << "get fuse_multi_transformer_layer_pass failed";

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = GetNumOpNodes(graph, "fused_multi_transformer");

  PADDLE_ENFORCE_EQ(
      num_nodes_after,
      1,
      platform::errors::InvalidArgument(
          "After the fuse_multi_transformer_layer_pass, "
          "The node num in graph should be 1, but the result is %d",
          num_nodes_after));
}
TEST(FuseMultiTransformerLayerPass, decoder_fp) {
  DEF_INPUT_DATA

  x = layers.data("x", {1, 1, 1024});
  auto* cache_kv = layers.data("cache_kv", {2, 1, 16, 1024, 64}, true);
  src_mask = layers.data("src_mask", {1, 16, 1, 129});

  // Layers
  for (int i = 0; i < num_layers; ++i) {
    auto* shape_out = layers.shape(src_mask);
    auto* time_stamp = layers.slice(shape_out, {0}, {3}, {4});
    auto* out = layers.fused_multi_transformer(x,
                                               cache_kv,
                                               src_mask,
                                               qkv_w,
                                               qkv_bias,
                                               out_linear_w,
                                               out_linear_bias,
                                               ffn1_w,
                                               ffn1_bias,
                                               ffn2_w,
                                               ffn2_bias,
                                               ln_scale,
                                               ln_bias,
                                               ffn_ln_scale,
                                               ffn_ln_bias,
                                               0.1,
                                               1e-12,
                                               time_stamp);

    x = out;
  }
  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto param_scope = CreateParamScope();
  AddVarToScope(param_scope, "cache_kv", {2, 1, 16, 1024, 64});
  graph->Set("__param_scope__", param_scope);

  graph->Set(kFusedMultiTransformerDecoderFusionCount, new int(num_layers));

  auto pass = PassRegistry::Instance().Get("fuse_multi_transformer_layer_pass");
  if (pass.get() == nullptr)
    LOG(INFO) << "get fuse_multi_transformer_layer_pass failed";

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = GetNumOpNodes(graph, "fused_multi_transformer");

  PADDLE_ENFORCE_EQ(
      num_nodes_after,
      1,
      platform::errors::InvalidArgument(
          "After the fuse_multi_transformer_layer_pass, "
          "The node num in graph should be 1, but the result is %d",
          num_nodes_after));
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fuse_multi_transformer_layer_pass);
