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
  var->SetType(proto::VarType::LOD_TENSOR);
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

TEST(ApplyDeleteMulDataPrePass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block = program.MutableBlock(0);
  auto* shape_in =
      Data(block, "shape_in", {64, 128}, false, proto::VarType::INT64);

  auto* shape_a_out =
      Data(block, "shape_a_out", {2}, false, proto::VarType::INT32);
  auto* slice0_a_out =
      Data(block, "slice0_a_out", {1}, false, proto::VarType::INT32);
  auto* slice1_a_out =
      Data(block, "slice1_a_out", {1}, false, proto::VarType::INT32);
  auto* cast_a_in = Data(block, "cast_a_in", {1}, false, proto::VarType::INT64);
  auto* elementwise_add_a_out =
      Data(block, "elementwise_add_a_out", {1}, false, proto::VarType::INT32);
  auto* scale_a_out =
      Data(block, "scale_a_out", {1}, false, proto::VarType::INT32);
  auto* fill_constant_a_out = Data(
      block, "fill_constant_a_out", {2, 1, 16, 1, 64}, proto::VarType::FP16);

  auto* shape_b_out =
      Data(block, "shape_b_out", {2}, false, proto::VarType::INT32);
  auto* slice0_b_out =
      Data(block, "slice0_b_out", {1}, false, proto::VarType::INT32);
  auto* slice1_b_out =
      Data(block, "slice1_b_out", {1}, false, proto::VarType::INT32);
  auto* cast_b_in = Data(block, "cast_b_in", {1}, false, proto::VarType::INT64);
  auto* elementwise_add_b_out =
      Data(block, "elementwise_add_b_out", {1}, false, proto::VarType::INT32);
  auto* scale_b_out =
      Data(block, "scale_b_out", {1}, false, proto::VarType::INT32);
  auto* fill_constant_b_out = Data(
      block, "fill_constant_b_out", {2, 1, 16, 1, 64}, proto::VarType::FP16);

  // sub1
  OpDesc* shape_a = block->AppendOp();
  shape_a->SetType("shape");
  shape_a->SetInput("X", {shape_in->Name()});
  shape_a->SetOutput("Out", {shape_a_out->Name()});

  OpDesc* slice0_a = block->AppendOp();
  slice0_a->SetType("slice");
  slice0_a->SetInput("X", {shape_a_out->Name()});
  slice0_a->SetOutput("Out", {slice0_a_out->Name()});

  OpDesc* slice1_a = block->AppendOp();
  slice1_a->SetType("slice");
  slice1_a->SetInput("X", {shape_a_out->Name()});
  slice1_a->SetOutput("Out", {slice1_a_out->Name()});

  auto* cast_a_out = AddCast(block,
                             cast_a_in,
                             static_cast<int>(proto::VarType::INT64),
                             static_cast<int>(proto::VarType::INT32));

  OpDesc* elementwise_add_a = block->AppendOp();
  elementwise_add_a->SetType("elementwise_add");
  elementwise_add_a->SetInput("X", {cast_a_out->Name()});
  elementwise_add_a->SetInput("Y", {slice1_a_out->Name()});
  elementwise_add_a->SetOutput("Out", {elementwise_add_a_out->Name()});

  OpDesc* scale_a = block->AppendOp();
  scale_a->SetType("scale");
  scale_a->SetInput("X", {elementwise_add_a_out->Name()});
  scale_a->SetOutput("Out", {scale_a_out->Name()});
  scale_a->SetAttr("scale", 1.0f);
  scale_a->SetAttr("bias", 64.0f);

  OpDesc* fill_constant_a = block->AppendOp();
  fill_constant_a->SetType("fill_constant");
  fill_constant_a->SetInput("X", {slice0_a_out->Name()});
  fill_constant_a->SetInput("Y", {scale_a_out->Name()});
  fill_constant_a->SetOutput("Out", {fill_constant_a_out->Name()});

  // sub2
  OpDesc* shape_b = block->AppendOp();
  shape_b->SetType("shape");
  shape_b->SetInput("X", {shape_in->Name()});
  shape_b->SetOutput("Out", {shape_b_out->Name()});

  OpDesc* slice0_b = block->AppendOp();
  slice0_b->SetType("slice");
  slice0_b->SetInput("X", {shape_b_out->Name()});
  slice0_b->SetOutput("Out", {slice0_b_out->Name()});

  OpDesc* slice1_b = block->AppendOp();
  slice1_b->SetType("slice");
  slice1_b->SetInput("X", {shape_b_out->Name()});
  slice1_b->SetOutput("Out", {slice1_b_out->Name()});

  auto* cast_b_out = AddCast(block,
                             cast_b_in,
                             static_cast<int>(proto::VarType::INT64),
                             static_cast<int>(proto::VarType::INT32));

  OpDesc* elementwise_add_b = block->AppendOp();
  elementwise_add_b->SetType("elementwise_add");
  elementwise_add_b->SetInput("X", {cast_b_out->Name()});
  elementwise_add_b->SetInput("Y", {slice1_b_out->Name()});
  elementwise_add_b->SetOutput("Out", {elementwise_add_b_out->Name()});

  OpDesc* scale_b = block->AppendOp();
  scale_b->SetType("scale");
  scale_b->SetInput("X", {elementwise_add_b_out->Name()});
  scale_b->SetOutput("Out", {scale_b_out->Name()});
  scale_b->SetAttr("scale", 1.0f);
  scale_b->SetAttr("bias", 64.0f);

  OpDesc* fill_constant_b = block->AppendOp();
  fill_constant_b->SetType("fill_constant");
  fill_constant_b->SetInput("X", {slice0_b_out->Name()});
  fill_constant_b->SetInput("Y", {scale_b_out->Name()});
  fill_constant_b->SetOutput("Out", {fill_constant_b_out->Name()});

  OpDesc* fused_multi_transformer_dyquant_xpu = block->AppendOp();
  fused_multi_transformer_dyquant_xpu->SetType(
      "fused_multi_transformer_dyquant_xpu");
  fused_multi_transformer_dyquant_xpu->SetInput("X",
                                                {fill_constant_a_out->Name()});
  fused_multi_transformer_dyquant_xpu->SetInput("Y",
                                                {fill_constant_b_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto scope = new Scope();
  graph->Set("__param_scope__", scope);
  auto pass = PassRegistry::Instance().Get(
      "xpu_delete_mul_data_preparation_for_fill_pass");
  pass->Apply(graph.get());
  int shape_num_in_graph = GetOpNum(graph->GetSubGraph(0), "shape");
  PADDLE_ENFORCE_EQ(GetOpNum(graph->GetSubGraph(0), "shape"),
                    1,
                    platform::errors::PreconditionNotMet(
                        "graph should have 1 shape after "
                        "xpu_delete_mul_data_preparation_for_fill_pass, "
                        "but actually has %d.",
                        shape_num_in_graph));
  int slice_num_in_graph = GetOpNum(graph->GetSubGraph(0), "slice");
  PADDLE_ENFORCE_EQ(GetOpNum(graph->GetSubGraph(0), "slice"),
                    2,
                    platform::errors::PreconditionNotMet(
                        "graph should have 2 slice after "
                        "xpu_delete_mul_data_preparation_for_fill_pass, "
                        "but actually has %d.",
                        slice_num_in_graph));
  int elementwise_add_num_in_graph =
      GetOpNum(graph->GetSubGraph(0), "elementwise_add");
  PADDLE_ENFORCE_EQ(GetOpNum(graph->GetSubGraph(0), "elementwise_add"),
                    1,
                    platform::errors::PreconditionNotMet(
                        "graph should have 1 elementwise_add after "
                        "xpu_delete_mul_data_preparation_for_fill_pass, "
                        "but actually has %d.",
                        elementwise_add_num_in_graph));
  int scale_num_in_graph = GetOpNum(graph->GetSubGraph(0), "scale");
  PADDLE_ENFORCE_EQ(GetOpNum(graph->GetSubGraph(0), "scale"),
                    1,
                    platform::errors::PreconditionNotMet(
                        "graph should have 1 scale after "
                        "xpu_delete_mul_data_preparation_for_fill_pass, "
                        "but actually has %d.",
                        scale_num_in_graph));
  int fill_constant_num_in_graph =
      GetOpNum(graph->GetSubGraph(0), "fill_constant");
  PADDLE_ENFORCE_EQ(GetOpNum(graph->GetSubGraph(0), "fill_constant"),
                    1,
                    platform::errors::PreconditionNotMet(
                        "graph should have 1 fill_constant after "
                        "xpu_delete_mul_data_preparation_for_fill_pass, "
                        "but actually has %d.",
                        fill_constant_num_in_graph));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(xpu_delete_mul_data_preparation_for_fill_pass);
