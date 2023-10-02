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

#include "paddle/cinn/frontend/decomposer/test_helper.h"

namespace cinn {
namespace frontend {

TEST(OpFusionPass, ElementWise_Fusion_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(C, D);
    auto G = net_builder.Add(E, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, ElementWise_Fusion_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(E, C);
    auto G = net_builder.Add(E, D);
    auto H = net_builder.Add(F, G);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, Brodcast_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Brodcast_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(C, A, 0);
    auto F = net_builder.Add(D, B, 0);
    auto G = net_builder.Add(E, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, Brodcast_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Brodcast_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(C, D);
    auto G = net_builder.Add(F, E, 0);
    auto H = net_builder.Add(G, C);
    auto I = net_builder.Add(H, D);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, Brodcast_Test_2) {
  int n = 2, c = 16, h = 32, w = 32;
  NetBuilder net_builder("Brodcast_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {c}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, c, h, w}, "B");
    auto C = net_builder.Reshape(A, {c, 1, 1});
    auto D = net_builder.Multiply(B, C);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, Reduce_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(C, D);
    auto G = net_builder.ReduceSum(F, {0});
    auto H = net_builder.Add(E, G);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(OpFusionPass, Reduce_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.ReduceSum(C, {0});
    auto G = net_builder.ReduceSum(D, {0});
    auto H = net_builder.Add(E, F);
    auto I = net_builder.Add(G, H);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, Reduce_Test_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ReduceSum(C, {0});
    auto F = net_builder.ReduceSum(D, {1});
    auto G = net_builder.Add(A, E);
    auto H = net_builder.Add(B, F);
    auto I = net_builder.Add(G, H);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(OpFusionPass, Injective_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Injective_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h * 2, w}, "D");

    auto E = net_builder.Add(A, B);
    auto F = net_builder.Concat({C, E}, 0);
    auto G = net_builder.Add(D, F);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OP_LOWERING, Injective_Test_1) {
  NetBuilder net_builder("Injective_Test_1");
  auto A = net_builder.CreateInput(Float(32), {1, 19}, "A");
  auto B = net_builder.CreateInput(Float(32), {1, 19, 204}, "B");
  auto C = net_builder.ExpandDims(A, {1});
  auto D = net_builder.BroadcastTo(C, {1, 204, 19}, {0, 1, 2});
  auto E = net_builder.Transpose(B, {0, 2, 1});
  auto F = net_builder.Add(D, E);

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(OpFusionPass, Test_Insert_BroadcastTo) {
  int h = 32, w = 32;
  NetBuilder net_builder("Test_Insert_BroadcastTo");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");

    auto E = net_builder.Add(C, A, -1);
    auto F = net_builder.Add(E, B, -1);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  CHECK_EQ(graph->fusion_groups.size(), 1);
}

}  // namespace frontend
}  // namespace cinn
