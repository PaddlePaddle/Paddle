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

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

template <typename T = float>
void AddVarToScope(Scope* param_scope,
                   const std::string& name,
                   const DDim& dims,
                   T value = 0) {
  auto* tensor = param_scope->Var(name)->GetMutable<phi::DenseTensor>();
  tensor->Resize(dims);
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  auto* data = cpu_ctx->Alloc<T>(tensor);
  for (int64_t i = 0; i < tensor->numel(); i++) {
    data[i] = value;
  }
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

  OpDesc* beam_search_op = block->AppendOp();
  beam_search_op->SetType("beam_search");
  beam_search_op->SetAttr("beam_size", 1);

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
  gather_op->SetType("gather");
  gather_op->SetInput("X", {assign_out->Name()});
  gather_op->SetOutput("Out", {cache_kv->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto pass = PassRegistry::Instance().Get("one_beam_size_fuse_pass");
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

TEST(FoldShapeAssociatedOps, basic) {
  Layers layers;
  auto* block = layers.Block();

  OpDesc* beam_search_op = block->AppendOp();
  beam_search_op->SetType("beam_search");
  beam_search_op->SetAttr("beam_size", 1);

  auto* shape_x = layers.data("shape_x", {1, 46256});
  auto* shape_out = layers.shape(shape_x);
  auto* slice_out = layers.slice(shape_out, {0}, {0}, {1});
  auto* div_out = layers.elementwise_div(slice_out, slice_out);
  auto* cast0_out = layers.cast(div_out);
  auto* cast1_out = layers.cast(slice_out);
  auto* scale0_out = layers.scale(slice_out);
  auto* cast2_out = layers.cast(scale0_out);
  auto* range_out = layers.range(cast2_out, cast1_out, cast0_out);
  auto* unsqueeze2_out = layers.unsqueeze2(range_out);
  auto* scale1_out = layers.scale(unsqueeze2_out);
  auto* add_x = layers.data("add_x", {1, 2});
  auto* add_out = layers.elementwise_add(add_x, scale1_out);
  layers.flatten_contiguous_range(add_out);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("one_beam_size_fuse_pass");
  pass->Apply(graph.get());
  auto ops_num = GetNumOpNodes(graph);
  PADDLE_ENFORCE_EQ(
      ops_num,
      2,
      common::errors::PreconditionNotMet(
          "graph should only have 2 op nodes, but received %d.", ops_num));
}

TEST(RemoveBeamSearchAssociatedOps, basic) {
  Layers layers;
  auto* lod_reset_0_x = layers.data("lod_reset_0_x");
  auto* lod_reset_0_y = layers.data("lod_reset_0_y");
  auto* lod_reset_0_out = layers.lod_reset(lod_reset_0_x, lod_reset_0_y);
  auto* lod_reset_1_x = layers.data("lod_reset_1_x");
  auto* lod_reset_1_y = layers.data("lod_reset_1_y");
  auto* lod_reset_1_out = layers.lod_reset(lod_reset_1_x, lod_reset_1_y);

  auto* pre_ids = layers.data("pre_ids");
  auto* pre_scores = layers.data("pre_scores");
  auto beam_search_outs =
      layers.beam_search(lod_reset_0_out, lod_reset_1_out, pre_ids, pre_scores);
  auto* parent_idx = beam_search_outs[0];
  auto* selected_ids = beam_search_outs[1];
  auto* selected_scores = beam_search_outs[2];

  auto* write_to_array_0_i = layers.data("write_to_array_0_i");
  layers.write_to_array(selected_ids, write_to_array_0_i);
  auto* write_to_array_1_i = layers.data("write_to_array_1_i");
  layers.write_to_array(selected_scores, write_to_array_1_i);
  auto* is_empty_out = layers.is_empty(selected_ids);
  layers.logical_not(is_empty_out);
  layers.cast(parent_idx);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto* param_scope = new Scope();
  graph->Set("__param_scope__", param_scope);
  auto pass = PassRegistry::Instance().Get("one_beam_size_fuse_pass");
  pass->Apply(graph.get());
  auto beam_search_num = GetNumOpNodes(graph, "beam_search");
  PADDLE_ENFORCE_EQ(beam_search_num,
                    0,
                    common::errors::PreconditionNotMet(
                        "beam_search op should be removed from the graph."));
}

TEST(RemoveWriteReadArrayOps, basic) {
  Layers layers;
  auto* block = layers.Block();
  OpDesc* beam_search_op = block->AppendOp();
  beam_search_op->SetType("beam_search");
  beam_search_op->SetAttr("beam_size", 1);

  auto* write_x = layers.data("write_x", {1}, true);
  auto* write_i = layers.data("write_i");
  auto* write_out = layers.write_to_array(write_x, write_i);
  auto* read_i = layers.data("read_i");
  layers.read_from_array(write_out, read_i);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto* param_scope = new Scope();
  graph->Set("__param_scope__", param_scope);
  AddVarToScope(param_scope, write_x->Name(), {1});
  auto pass = PassRegistry::Instance().Get("one_beam_size_fuse_pass");
  pass->Apply(graph.get());
  auto write_read_num = GetNumOpNodes(graph, "write_to_array") +
                        GetNumOpNodes(graph, "read_from_array");
  PADDLE_ENFORCE_EQ(write_read_num,
                    0,
                    common::errors::PreconditionNotMet(
                        "write_to_array and read_from_array ops should be "
                        "removed from the graph."));
}

TEST(RemoveGatherOps, basic) {
  Layers layers;
  auto* block = layers.Block();
  OpDesc* beam_search_op = block->AppendOp();
  beam_search_op->SetType("beam_search");
  beam_search_op->SetAttr("beam_size", 1);

  auto* gather_x = layers.data("gather_x");
  auto* gather_i = layers.data("gather_i", {1}, true);
  layers.gather(gather_x, gather_i, 0);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto* param_scope = new Scope();
  graph->Set("__param_scope__", param_scope);
  AddVarToScope<int>(param_scope, gather_i->Name(), {1}, 0);
  auto pass = PassRegistry::Instance().Get("one_beam_size_fuse_pass");
  pass->Apply(graph.get());
  auto gather_num = GetNumOpNodes(graph, "gather");
  PADDLE_ENFORCE_EQ(gather_num,
                    0,
                    common::errors::PreconditionNotMet(
                        "gather op should be removed from the graph."));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(one_beam_size_fuse_pass);
