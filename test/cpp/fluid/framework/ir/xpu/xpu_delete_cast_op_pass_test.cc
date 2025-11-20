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

TEST(ApplyCastSoftmaxPass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);
  auto* cast0_in = Data(block, "cast0_in", {1});
  auto* cast0_out = AddCast(block, cast0_in, 4, 5);
  auto* softmax_out = Data(block, "softmax_out", {1});
  OpDesc* softmax = block->AppendOp();
  softmax->SetType("softmax");
  softmax->SetInput("X", {cast0_out->Name()});
  softmax->SetOutput("Out", {softmax_out->Name()});
  AddCast(block, softmax_out, 5, 4);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get("xpu_delete_cast_op_pass");
  pass->Apply(graph.get());
  int cast_num_in_graph = GetOpNum(graph->GetSubGraph(0), "cast");
  PADDLE_ENFORCE_EQ(
      GetOpNum(graph->GetSubGraph(0), "cast"),
      0,
      common::errors::PreconditionNotMet(
          "graph should have 0 cast after xpu_delete_cast_op_pass, "
          "but actually has %d.",
          cast_num_in_graph));
}

TEST(ApplyCastLayerNormPass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);
  auto* cast0_in = Data(block, "cast0_in", {1});
  auto* cast0_out = AddCast(block, cast0_in, 4, 5);
  auto* layer_norm_out = Data(block, "layer_norm_out", {1});
  OpDesc* layer_norm = block->AppendOp();
  layer_norm->SetType("layer_norm");
  layer_norm->SetInput("X", {cast0_out->Name()});
  layer_norm->SetOutput("Y", {layer_norm_out->Name()});
  AddCast(block, layer_norm_out, 5, 4);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get("xpu_delete_cast_op_pass");
  pass->Apply(graph.get());
  int cast_num_in_graph = GetOpNum(graph->GetSubGraph(0), "cast");
  PADDLE_ENFORCE_EQ(
      GetOpNum(graph->GetSubGraph(0), "cast"),
      0,
      common::errors::PreconditionNotMet(
          "graph should have 0 cast after xpu_delete_cast_op_pass, "
          "but actually has %d.",
          cast_num_in_graph));
}

TEST(ApplyCastCacheKVInitializationPass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);
  auto* shape_in =
      Data(block, "shape_in", {64, 128}, false, proto::VarType::INT64);
  auto* shape0_out =
      Data(block, "shape0_out", {2}, false, proto::VarType::INT32);
  auto* shape1_out =
      Data(block, "shape1_out", {2}, false, proto::VarType::INT32);
  auto* slice0_out =
      Data(block, "slice0_out", {1}, false, proto::VarType::INT32);
  auto* slice1_out =
      Data(block, "slice1_out", {1}, false, proto::VarType::INT32);
  auto* elementwise_add_in0 =
      Data(block, "elementwise_add_in0", {1}, false, proto::VarType::INT64);
  auto* elementwise_add_out =
      Data(block, "elementwise_add_out", {1}, false, proto::VarType::INT64);
  auto* scale_out = Data(block, "scale_out", {1}, false, proto::VarType::INT64);

  OpDesc* shape0 = block->AppendOp();
  shape0->SetType("shape");
  shape0->SetInput("X", {shape_in->Name()});
  shape0->SetOutput("Out", {shape0_out->Name()});

  OpDesc* shape1 = block->AppendOp();
  shape1->SetType("shape");
  shape1->SetInput("X", {shape_in->Name()});
  shape1->SetOutput("Out", {shape1_out->Name()});

  OpDesc* slice0 = block->AppendOp();
  slice0->SetType("slice");
  slice0->SetInput("X", {shape0_out->Name()});
  slice0->SetOutput("Out", {slice0_out->Name()});

  OpDesc* slice1 = block->AppendOp();
  slice1->SetType("slice");
  slice1->SetInput("X", {shape1_out->Name()});
  slice1->SetOutput("Out", {slice1_out->Name()});

  auto cast0_out = AddCast(block,
                           slice1_out,
                           static_cast<int>(proto::VarType::INT32),
                           static_cast<int>(proto::VarType::INT64));

  OpDesc* elementwise_add = block->AppendOp();
  elementwise_add->SetType("elementwise_add");
  elementwise_add->SetInput("X", {elementwise_add_in0->Name()});
  elementwise_add->SetInput("Y", {cast0_out->Name()});
  elementwise_add->SetOutput("Out", {elementwise_add_out->Name()});

  OpDesc* scale = block->AppendOp();
  scale->SetType("scale");
  scale->SetInput("X", {elementwise_add_out->Name()});
  scale->SetOutput("Out", {scale_out->Name()});
  scale->SetAttr("scale", 1.0f);
  scale->SetAttr("bias", 64.0f);

  auto* cast1_out = AddCast(block,
                            scale_out,
                            static_cast<int>(proto::VarType::INT64),
                            static_cast<int>(proto::VarType::INT32));

  OpDesc* fill_constant = block->AppendOp();
  fill_constant->SetType("fill_constant");
  fill_constant->SetInput("X", {slice0_out->Name()});
  fill_constant->SetInput("Y", {cast1_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get("xpu_delete_cast_op_pass");
  pass->Apply(graph.get());
  int shape_num_in_graph = GetOpNum(graph->GetSubGraph(0), "shape");
  PADDLE_ENFORCE_EQ(
      GetOpNum(graph->GetSubGraph(0), "shape"),
      1,
      common::errors::PreconditionNotMet("graph should have 1 shape after "
                                         "xpu_delete_cast_op_pass, "
                                         "but actually has %d.",
                                         shape_num_in_graph));
  int cast_num_in_graph = GetOpNum(graph->GetSubGraph(0), "cast");
  PADDLE_ENFORCE_EQ(
      GetOpNum(graph->GetSubGraph(0), "cast"),
      1,
      common::errors::PreconditionNotMet("graph should have 1 cast after "
                                         "xpu_delete_cast_op_pass, "
                                         "but actually has %d.",
                                         cast_num_in_graph));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(xpu_delete_cast_op_pass);
