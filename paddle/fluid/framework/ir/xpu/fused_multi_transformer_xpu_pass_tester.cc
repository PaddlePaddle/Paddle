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

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

#define DEF_INPUT_DATA                                                  \
  Layers layers;                                                        \
  auto* x = layers.data("x", {1, 128, 1024});                           \
  auto* src_mask = layers.data("src_mask", {1, 16, 128, 128});          \
  auto* ln_scale = layers.data("ln_scale", {1024}, true);               \
  auto* ln_bias = layers.data("ln_bias", {1024}, true);                 \
  auto* qkv_w = layers.data("qkv_w", {3, 16, 64, 1024}, true);          \
  auto* qkv_bias = layers.data("qkv_bias", {3, 16, 64}, true);          \
  auto* out_linear_w = layers.data("out_linear_w", {1024, 1024}, true); \
  auto* out_linear_bias = layers.data("out_linear_bias", {1024}, true); \
  auto* ffn_ln_scale = layers.data("ffn_ln_scale", {1024}, true);       \
  auto* ffn_ln_bias = layers.data("ffn_ln_bias", {1024}, true);         \
  auto* ffn1_w = layers.data("ffn1_w", {1024, 4096}, true);             \
  auto* ffn1_bias = layers.data("ffn1_bias", {4096}, true);             \
  auto* ffn2_w = layers.data("ffn2_w", {4096, 1024}, true);             \
  auto* ffn2_bias = layers.data("ffn2_bias", {1024}, true);

namespace paddle {
namespace framework {
namespace ir {

void AddVarToScope(Scope* param_scope,
                   const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<phi::DenseTensor>();
  tensor->Resize(dims);
  tensor->mutable_data<float>(phi::CPUPlace());
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

VarDesc* Data(paddle::framework::BlockDesc* block,
              std::string name,
              std::vector<int64_t> shape = {},
              bool is_persistable = false,
              proto::VarType::Type data_type = proto::VarType::FP32) {
  auto* var = block->Var(name);
  var->SetType(proto::VarType::DENSE_TENSOR);
  var->SetDataType(data_type);
  var->SetShape(shape);
  var->SetPersistable(is_persistable);
  return var;
}

TEST(RemoveAssignGather, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);

  auto* x = Data(block, "fused_multi_transformer_x", {1, 1, 1536});
  auto* cache_kv =
      Data(block, "fused_multi_transformer_cache_kv", {2, 1, 24, 512, 64});
  OpDesc* fused_multi_transformer_op = block->AppendOp();
  fused_multi_transformer_op->SetType("fused_multi_transformer");
  fused_multi_transformer_op->SetInput("X", {x->Name()});
  fused_multi_transformer_op->SetInput("CacheKV", {cache_kv->Name()});
  fused_multi_transformer_op->SetOutput("CacheKVOut", {cache_kv->Name()});

  auto* assign_out = Data(block, "assign_out", cache_kv->GetShape());
  OpDesc* assign_op = block->AppendOp();
  assign_op->SetType("assign");
  assign_op->SetInput("X", {cache_kv->Name()});
  assign_op->SetOutput("Out", {assign_out->Name()});

  OpDesc* gather_op = block->AppendOp();
  auto gather_index = Data(block, "gather_index", {10});
  gather_op->SetType("gather");
  gather_op->SetInput("X", {assign_out->Name()});
  gather_op->SetInput("Index", {gather_index->Name()});
  gather_op->SetAttr("axis", {1});
  gather_op->SetOutput("Out", {cache_kv->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto pass = PassRegistry::Instance().Get("fused_multi_transformer_xpu_pass");
  pass->Apply(graph.get());
  auto assign_num = GetNumOpNodes(graph, "assign");
  auto gather_num = GetNumOpNodes(graph, "gather");
  PADDLE_ENFORCE_EQ(assign_num,
                    0,
                    common::errors::PreconditionNotMet(
                        "assign op should be removed from the graph."));
  PADDLE_ENFORCE_EQ(gather_num,
                    0,
                    common::errors::PreconditionNotMet(
                        "gather op should be removed from the graph."));
}

TEST(FusedMultiTransformerXPUPass, context_stage) {
  DEF_INPUT_DATA

  auto* cache_kv = layers.fill_constant_batch_size_like(
      x,
      static_cast<int>(proto::VarType::FP32),
      0,
      1,
      {2, -1, 16, 1024, 64},
      0);

  layers.fused_multi_transformer(x,
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
  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());

  auto pass = PassRegistry::Instance().Get("fused_multi_transformer_xpu_pass");
  if (pass.get() == nullptr) {
    LOG(INFO) << "get fused_multi_transformer_xpu_pass failed";
  }

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = GetNumOpNodes(graph, "fused_multi_transformer_xpu");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(
      num_nodes_after,
      1,
      common::errors::InvalidArgument(
          "After the fuse_multi_transformer_layer_pass, "
          "The node num in graph should be 1, but the result is %d",
          num_nodes_after));
}

TEST(FusedMultiTransformerXPUPass, decoder_stage) {
  DEF_INPUT_DATA

  auto* cache_kv = layers.fill_constant_batch_size_like(
      x,
      static_cast<int>(proto::VarType::FP32),
      0,
      1,
      {2, -1, 16, 1024, 64},
      0);
  auto* time_step = layers.data("time_step", {1});
  layers.fused_multi_transformer(x,
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
                                 time_step);
  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());

  auto pass = PassRegistry::Instance().Get("fused_multi_transformer_xpu_pass");
  if (pass.get() == nullptr) {
    LOG(INFO) << "get fused_multi_transformer_xpu_pass failed";
  }

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = GetNumOpNodes(graph, "fused_multi_transformer_xpu");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(
      num_nodes_after,
      1,
      common::errors::InvalidArgument(
          "After the fuse_multi_transformer_layer_pass, "
          "The node num in graph should be 1, but the result is %d",
          num_nodes_after));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fused_multi_transformer_xpu_pass);
