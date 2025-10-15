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

namespace paddle::framework::ir {

void AddVarToScope(Scope* param_scope,
                   const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<phi::DenseTensor>();
  tensor->Resize(dims);
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  cpu_ctx->Alloc<float>(tensor);
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

VarDesc* AddWriteToArray(BlockDesc* block,
                         std::vector<VarDesc*> x,
                         VarDesc* i,
                         VarDesc* out = nullptr) {
  if (out == nullptr) {
    out = Data(block, x[0]->Name() + "_out");
  }
  OpDesc* op = block->AppendOp();
  op->SetType("write_to_array");
  std::vector<std::string> x_names;
  x_names.reserve(x.size());
  for (auto k : x) {
    x_names.push_back(k->Name());
  }
  op->SetInput("X", x_names);
  op->SetInput("I", {i->Name()});
  op->SetOutput("Out", {out->Name()});
  return out;
}

VarDesc* AddReadFromArray(BlockDesc* block, VarDesc* x, VarDesc* i) {
  auto* out = Data(block, x->Name() + "_out");
  OpDesc* op = block->AppendOp();
  op->SetType("read_from_array");
  op->SetInput("X", {x->Name()});
  op->SetInput("I", {i->Name()});
  op->SetOutput("Out", {out->Name()});
  return out;
}

VarDesc* AddCast(BlockDesc* block,
                 VarDesc* input,
                 int in_dtype = 5,
                 int out_dtype = 5) {
  VarDesc* out = Data(block, input->Name() + "_out");
  OpDesc* op = block->AppendOp();
  op->SetType("cast");
  op->SetInput("X", {input->Name()});
  op->SetOutput("Out", {out->Name()});
  op->SetAttr("in_dtype", in_dtype);
  op->SetAttr("out_dtype", out_dtype);
  return out;
}

VarDesc* AddLodReset(BlockDesc* block, VarDesc* input) {
  VarDesc* out = Data(block, input->Name() + "_out");
  OpDesc* op = block->AppendOp();
  op->SetType("lod_reset");
  op->SetInput("X", {input->Name()});
  op->SetOutput("Out", {out->Name()});
  return out;
}

std::vector<VarDesc*> AddBeamSearchDecode(BlockDesc* block,
                                          VarDesc* ids,
                                          VarDesc* scores) {
  VarDesc* out_ids = Data(block, ids->Name() + "_out");
  VarDesc* out_scores = Data(block, scores->Name() + "_out");
  OpDesc* op = block->AppendOp();
  op->SetType("beam_search_decode");
  op->SetInput("Ids", {ids->Name()});
  op->SetInput("Scores", {scores->Name()});
  op->SetOutput("SentenceIds", {out_ids->Name()});
  op->SetOutput("SentenceScores", {out_scores->Name()});
  return {out_ids, out_scores};
}

int GetOpNum(Graph* graph, std::string op_type = "") {
  int num_nodes = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op() &&
        (node->Op()->Type() == op_type || op_type.empty())) {
      num_nodes++;
    }
  }
  return num_nodes;
}

TEST(ApplyCastWriteReadPass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block0 = program.MutableBlock(0);
  auto* block1 = program.AppendBlock(*block0);
  auto* write_0_x = Data(block0, "write_0_x", {1});
  auto* write_0_i = Data(block0, "write_0_i", {1});
  auto* write_0_out = AddWriteToArray(block0, {write_0_x}, write_0_i);
  OpDesc* while_loop = block0->AppendOp();
  while_loop->SetType("while");
  while_loop->SetInput("X", {write_0_out->Name()});
  while_loop->SetOutput("Out", {write_0_out->Name()});

  auto* cast_1_0_in = Data(block1, "cast_1_0", {1});
  auto* cast_1_0_out = AddCast(block1, cast_1_0_in, 4, 5);
  auto* write_1_i = Data(block1, "write_1_i", {1});
  auto* write_1_out = Data(block1, write_0_out->Name(), {1});
  AddWriteToArray(block1, {cast_1_0_out}, write_1_i, write_1_out);
  auto* read_1_i = Data(block1, "read_1_i", {1});
  auto* read_1_out = AddReadFromArray(block1, write_1_out, read_1_i);
  AddCast(block1, read_1_out, 5, 4);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get("delete_cast_op_pass");
  pass->Apply(graph.get());

  int cast_num_in_graph1 = GetOpNum(graph->GetSubGraph(1), "cast");
  PADDLE_ENFORCE_EQ(cast_num_in_graph1,
                    0,
                    common::errors::PreconditionNotMet(
                        "graph1 should have 0 cast after delete_cast_op_pass, "
                        "but actually has %d.",
                        cast_num_in_graph1));
  int cast_num_in_graph0 = GetOpNum(graph.get(), "cast");
  PADDLE_ENFORCE_EQ(cast_num_in_graph0,
                    1,
                    common::errors::PreconditionNotMet(
                        "graph0 should have 1 cast after delete_cast_op_pass, "
                        "but actually has %d.",
                        cast_num_in_graph0));
}

TEST(ApplyCastLodResetWriteReadPass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block0 = program.MutableBlock(0);
  auto* block1 = program.AppendBlock(*block0);

  auto* write_0_x = Data(block0, "write_0_x", {1});
  auto* write_0_i = Data(block0, "write_0_i", {1});
  auto* write_0_out = AddWriteToArray(block0, {write_0_x}, write_0_i);
  OpDesc* while_loop = block0->AppendOp();
  while_loop->SetType("while");
  while_loop->SetInput("X", {write_0_out->Name()});
  while_loop->SetOutput("Out", {write_0_out->Name()});
  auto* ids = Data(block0, "ids", {1});
  AddBeamSearchDecode(block0, ids, write_0_out);

  auto* cast_1_0_in = Data(block1, "cast_1_0", {1});
  auto* cast_1_0_out = AddCast(block1, cast_1_0_in, 4, 5);
  auto* lod_reset_out = AddLodReset(block1, cast_1_0_out);
  auto* write_1_i = Data(block1, "write_1_i", {1});
  auto* write_1_out = Data(block1, write_0_out->Name(), {1});
  AddWriteToArray(block1, {lod_reset_out}, write_1_i, write_1_out);
  auto* read_1_i = Data(block1, "read_1_i", {1});
  auto* read_1_out = AddReadFromArray(block1, write_1_out, read_1_i);
  AddCast(block1, read_1_out, 5, 4);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get("delete_cast_op_pass");
  pass->Apply(graph.get());

  int cast_num_in_graph1 = GetOpNum(graph->GetSubGraph(1), "cast");
  PADDLE_ENFORCE_EQ(cast_num_in_graph1,
                    0,
                    common::errors::PreconditionNotMet(
                        "graph1 should have 0 cast after delete_cast_op_pass, "
                        "but actually has %d.",
                        cast_num_in_graph1));
  int cast_num_in_graph0 = GetOpNum(graph.get(), "cast");
  PADDLE_ENFORCE_EQ(cast_num_in_graph0,
                    2,
                    common::errors::PreconditionNotMet(
                        "graph0 should have 2 cast after delete_cast_op_pass, "
                        "but actually has %d.",
                        cast_num_in_graph0));
}

TEST(ApplyCastIndexSamplePass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);
  auto* cast0_in = Data(block, "cast0_in", {1});
  auto* cast0_out = AddCast(block, cast0_in, 4, 5);
  auto* index_sample_out = Data(block, "index_sample_out", {1});
  OpDesc* index_sample = block->AppendOp();
  index_sample->SetType("index_sample");
  index_sample->SetInput("X", {cast0_out->Name()});
  index_sample->SetOutput("Out", {index_sample_out->Name()});
  AddCast(block, index_sample_out, 5, 4);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get("delete_cast_op_pass");
  pass->Apply(graph.get());
  int cast_num_in_graph = GetOpNum(graph->GetSubGraph(0), "cast");
  PADDLE_ENFORCE_EQ(GetOpNum(graph->GetSubGraph(0), "cast"),
                    0,
                    common::errors::PreconditionNotMet(
                        "graph should have 0 cast after delete_cast_op_pass, "
                        "but actually has %d.",
                        cast_num_in_graph));
}

TEST(ApplyCastScatterPass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);
  auto* cast0_in = Data(block, "cast0_in", {1});
  auto* cast0_out = AddCast(block, cast0_in, 4, 5);
  auto* cast1_in = Data(block, "cast1_in", {1});
  auto* cast1_out = AddCast(block, cast1_in, 4, 5);
  auto* scatter_out = Data(block, "scatter_out", {1});
  OpDesc* scatter = block->AppendOp();
  scatter->SetType("scatter");
  scatter->SetInput("X", {cast0_out->Name()});
  scatter->SetInput("Updates", {cast1_out->Name()});
  scatter->SetOutput("Out", {scatter_out->Name()});
  AddCast(block, scatter_out, 5, 4);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get("delete_cast_op_pass");
  pass->Apply(graph.get());
  int cast_num_in_graph = GetOpNum(graph->GetSubGraph(0), "cast");
  PADDLE_ENFORCE_EQ(GetOpNum(graph->GetSubGraph(0), "cast"),
                    0,
                    common::errors::PreconditionNotMet(
                        "graph should have 0 cast after delete_cast_op_pass, "
                        "but actually has %d.",
                        cast_num_in_graph));
}

TEST(ApplyCastLookupTablePass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);
  auto* lookup_table_w = Data(block, "lookup_table_w", {1}, true);
  auto* lookup_table_out = Data(block, "scatter_out", {1});
  OpDesc* lookup_table = block->AppendOp();
  lookup_table->SetType("lookup_table_v2");
  lookup_table->SetInput("W", {lookup_table_w->Name()});
  lookup_table->SetOutput("Out", {lookup_table_out->Name()});
  auto* cast_out = AddCast(block, lookup_table_out, 5, 4);
  OpDesc* relu = block->AppendOp();
  relu->SetType("relu");
  relu->SetInput("X", {cast_out->Name()});
  relu->SetOutput("Out", {"relu_out"});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  AddVarToScope(scope, lookup_table_w->Name(), {1});
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get("delete_cast_op_pass");
  pass->Apply(graph.get());
  int cast_num_in_graph = GetOpNum(graph->GetSubGraph(0), "cast");
  PADDLE_ENFORCE_EQ(GetOpNum(graph->GetSubGraph(0), "cast"),
                    0,
                    common::errors::PreconditionNotMet(
                        "graph should have 0 cast after delete_cast_op_pass, "
                        "but actually has %d.",
                        cast_num_in_graph));
}

TEST(ApplyCastPass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);
  auto* cast0_in = Data(block, "cast0_in", {1});
  AddCast(block, cast0_in, 3, 3);
  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get("delete_cast_op_pass");
  pass->Apply(graph.get());
  int cast_num_in_graph = GetOpNum(graph->GetSubGraph(0), "cast");
  PADDLE_ENFORCE_EQ(GetOpNum(graph->GetSubGraph(0), "cast"),
                    0,
                    common::errors::PreconditionNotMet(
                        "graph should have 0 cast after delete_cast_op_pass, "
                        "but actually has %d.",
                        cast_num_in_graph));
}

}  // namespace paddle::framework::ir

USE_PASS(delete_cast_op_pass);
