// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <string>
#include <unordered_set>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/decomposer/test_helper.h"

namespace cinn::frontend {

using hlir::framework::Graph;

int CountAfterPassNodeSize(Graph* graph) {
  int node_size = 0, output_size = 0;
  for (auto group : graph->fusion_groups) {
    int group_size = group->CollectNodes().size();
    if (group_size == 1) {
      // CheckFusionAccuracyPass will skip if the group only has one node
      continue;
    }

    node_size += group_size;
    output_size += group->GetOutputNodeDatas().size();
  }

  // CheckFusionAccuracyPass will split each group, and add isclose+all+assert
  // node for each output
  return node_size + output_size * 3;
}

void RunTest(const Target& target,
             const std::shared_ptr<Graph>& graph,
             const std::vector<std::string>& input_names) {
  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);

  for (size_t i = 0; i < input_names.size(); ++i) {
    scope->Var<hlir::framework::Tensor>(input_names[i]);
    auto tensor = scope->GetTensor(input_names[i]);

    std::vector<float> vec;
    InitRandomVector<float>(&vec, tensor->shape().numel(), 0.0f, 1.0f);
    CopyFromVector<float>(vec, tensor, target);
  }

  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

TEST(CheckFusionAccuracyPass, ElementWise_Fusion) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_0");
  std::unordered_set<std::string> fetch_ids;
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(E, C);
    auto G = net_builder.Add(E, D);

    fetch_ids = {F->id, G->id};
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);
  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D"});
}

TEST(CheckFusionAccuracyPass, ElementWise_Fusion_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(C, D);
    auto G = net_builder.Add(E, F);
    auto I = net_builder.Add(E, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D"});
}

TEST(CheckFusionAccuracyPass, ElementWise_Fusion_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.CreateInput(Float(32), {h, w}, "E");
    auto F = net_builder.CreateInput(Float(32), {h, w}, "F");
    auto G = net_builder.Add(A, B);
    auto H = net_builder.Add(C, D);
    auto I = net_builder.Add(E, G);
    auto J = net_builder.Add(G, H);
    auto K = net_builder.Add(H, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D", "E", "F"});
}

TEST(CheckFusionAccuracyPass, ElementWise_Fusion_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.CreateInput(Float(32), {h, w}, "E");
    auto F = net_builder.CreateInput(Float(32), {h, w}, "F");
    auto G = net_builder.Add(A, B);
    auto H = net_builder.Add(G, C);
    auto I = net_builder.Add(G, D);
    auto J = net_builder.Add(G, E);
    auto K = net_builder.Add(G, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D", "E", "F"});
}

TEST(CheckFusionAccuracyPass, ElementWise_Fusion_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.CreateInput(Float(32), {h, w}, "E");
    auto F = net_builder.CreateInput(Float(32), {h, w}, "F");
    auto G = net_builder.Add(A, B);
    auto H = net_builder.Add(G, C);
    auto I = net_builder.Add(G, D);
    auto J = net_builder.Add(I, E);
    auto K = net_builder.Add(I, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D", "E", "F"});
}

TEST(CheckFusionAccuracyPass, ElementWise_Fusion_5) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.Add(A, B);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B"});
}

TEST(CheckFusionAccuracyPass, Broadcast_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(C, D);
    auto G = net_builder.Add(F, E);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D"});
}

TEST(CheckFusionAccuracyPass, Broadcast_Test_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(C, E);
    auto G = net_builder.Add(D, E);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D"});
}

TEST(CheckFusionAccuracyPass, Broadcast_Test_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.CreateInput(Float(32), {h, w}, "E");
    auto F = net_builder.Add(A, B);
    auto G = net_builder.Add(C, F);
    auto H = net_builder.Add(D, F);
    auto I = net_builder.Add(E, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D", "E"});
}

TEST(CheckFusionAccuracyPass, Broadcast_Test_5) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.CreateInput(Float(32), {h * w, w}, "E");
    auto F = net_builder.Add(A, B);
    auto G = net_builder.Add(C, F);
    auto H = net_builder.Add(D, F);
    auto I = net_builder.Add(E, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D", "E"});
}

TEST(CheckFusionAccuracyPass, Reduce_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.ReduceSum(C, {0});
    auto E = net_builder.ReduceSum(C, {0});
    auto F = net_builder.ReduceSum(C, {0});
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B"});
}

TEST(CheckFusionAccuracyPass, Reduce_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.ReduceSum(C, {0});
    auto E = net_builder.ReduceSum(C, {1});
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B"});
}

TEST(CheckFusionAccuracyPass, Reduce_Test_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0});
    auto F = net_builder.ReduceSum(D, {1});
    auto G = net_builder.Add(C, E);
    auto H = net_builder.Add(C, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C"});
}

TEST(CheckFusionAccuracyPass, Reduce_Test_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.ReduceSum(E, {0});
    auto G = net_builder.Add(C, F);
    auto H = net_builder.Add(D, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D"});
}

TEST(CheckFusionAccuracyPass, Reduce_Test_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.ReduceSum(E, {0});
    auto G = net_builder.Add(C, F);
    auto H = net_builder.Add(D, F);
    auto I = net_builder.Add(D, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B", "C", "D"});
}

TEST(CheckFusionAccuracyPass, Reduce_Test_5) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Test_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.ReduceSum(A, {1});
    auto E = net_builder.ReduceSum(B, {1});
    auto F = net_builder.ReduceSum(C, {1});
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<Graph>(program, target);

  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  int group_size_after =
      graph->fusion_groups.size() + CountAfterPassNodeSize(graph.get());

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  CHECK_EQ(graph->fusion_groups.size(), group_size_after);

  RunTest(target, graph, {"A", "B"});
}

}  // namespace cinn::frontend
