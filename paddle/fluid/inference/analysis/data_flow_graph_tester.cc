/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/analysis/data_flow_graph.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

TEST(DataFlowGraph, BFS) {
  auto desc = LoadProgramDesc(FLAGS_inference_model_dir + "/__model__");
  auto dfg = ProgramDescToDFG(desc);
  dfg.Build();

  for (auto* in : dfg.inputs()) {
    LOG(INFO) << "inputs: " << in->name() << " "
              << static_cast<int>(in->type());
  }
  for (auto* out : dfg.outputs()) {
    LOG(INFO) << "outputs: " << out->name() << " "
              << static_cast<int>(out->type());
  }

  size_t count = 0;
  for (auto& node : GraphTraits<DataFlowGraph>(dfg).nodes()) {
    LOG(INFO) << "visiting " << node.name();
    ++count;
  }
  ASSERT_EQ(count, dfg.nodes.size());
}

TEST(DataFlowGraph, DFS) {
  auto desc = LoadProgramDesc(FLAGS_inference_model_dir + "/__model__");
  DataFlowGraph dfg;
  dfg.Build(desc);
  size_t count = 0;
  for (auto& node : GraphTraits<DataFlowGraph>(dfg).nodes_in_DFS()) {
    LOG(INFO) << "visiting " << node.name();
    ++count;
  }
  ASSERT_EQ(count, dfg.nodes.size());
}

// Topological sorting.
/*
 * Graph topology
 * inputs: 0, 1, 2
 * 0 -> 4
 * 0 -> 5
 * 1 -> 6
 * 2 -> 7
 * 4 -> 5
 * 4 -> 7
 * 4 -> 3
 * 7 -> 3
 */
TEST(DataFlowGraph, TS) {
  DataFlowGraph graph;

  for (int i = 0; i < 8; i++) {
    auto* node = graph.nodes.Create(Node::Type::kValue);
    node->SetName("node-" + std::to_string(i));
  }

  auto add_link = [&](int i, int j) {
    Node* source = graph.nodes.GetMutable(i);
    Node* target = graph.nodes.GetMutable(j);
    target->inlinks.push_back(source);
    source->outlinks.push_back(target);
  };

  add_link(0, 4);
  add_link(0, 5);
  add_link(1, 6);
  add_link(2, 7);
  add_link(4, 5);
  add_link(4, 7);
  add_link(4, 3);
  add_link(7, 3);
  graph.Build();

  auto its = GraphTraits<DataFlowGraph>(graph).nodes_in_TS();
  std::vector<int> sorted_ids;
  for (auto it = its.begin(); it != its.end(); ++it) {
    LOG(INFO) << it->name();
    sorted_ids.push_back(it->id());
  }

  // Assert a occurs prior to b in the sorted_ids.
  auto assert_positive_sequence_pair = [&](int a, int b) {
    auto a_offset = std::find(sorted_ids.begin(), sorted_ids.end(), a);
    auto b_offset = std::find(sorted_ids.begin(), sorted_ids.end(), b);
    ASSERT_LT(a_offset, b_offset);
  };

  assert_positive_sequence_pair(2, 7);
  assert_positive_sequence_pair(7, 3);
  assert_positive_sequence_pair(4, 3);
  assert_positive_sequence_pair(0, 4);
  assert_positive_sequence_pair(0, 5);
  assert_positive_sequence_pair(1, 6);
  assert_positive_sequence_pair(4, 5);
  assert_positive_sequence_pair(4, 7);
}

TEST(DataFlowGraph, Build_ProgramDesc) {
  auto desc = LoadProgramDesc(FLAGS_inference_model_dir + "/__model__");
  DataFlowGraph graph;
  graph.Build(desc);
  ASSERT_EQ(graph.nodes.size(), 38UL);
}

void SetOp(framework::ProgramDesc* prog, const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetInput("Xs", inputs);
  op->SetOutput("Xs", outputs);
  op->SetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(framework::OpRole::kForward));
}

TEST(DataFlowGraph, Build_IR_Graph) {
  framework::ProgramDesc prog;
  for (auto& v : std::vector<std::string>({"a", "b", "c", "d", "e", "f"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(framework::proto::VarType::SELECTED_ROWS);
    if (v == "c") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "OP0", std::vector<std::string>({"a"}),
        std::vector<std::string>({"b"}));
  SetOp(&prog, "OP1", std::vector<std::string>({"a"}),
        std::vector<std::string>({"c"}));
  SetOp(&prog, "mul", std::vector<std::string>({"b", "c"}),
        std::vector<std::string>({"d"}));
  SetOp(&prog, "elementwise_add", std::vector<std::string>({"d", "e"}),
        std::vector<std::string>({"f"}));

  DataFlowGraph graph;

  framework::ir::Graph ir_graph(prog);

  graph.Build(ir_graph);

  ASSERT_EQ(graph.nodes.size(), ir_graph.Nodes().size());
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
