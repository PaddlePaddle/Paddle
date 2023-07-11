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

TEST(FusionMergePass, ElementWise_Fusion_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("ElementWise_Fusion_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(E, C);
    auto G = net_builder.Add(E, D);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(FusionMergePass, ElementWise_Fusion_1) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 4);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(FusionMergePass, ElementWise_Fusion_2) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 5);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(FusionMergePass, ElementWise_Fusion_3) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 5);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(FusionMergePass, ElementWise_Fusion_4) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 5);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(FusionMergePass, ElementWise_Fusion_5) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(FusionMergePass, Broadcast_Test_0) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(FusionMergePass, Broadcast_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(C, E);
    auto G = net_builder.Add(D, E);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

TEST(FusionMergePass, Broadcast_Test_2) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(FusionMergePass, Broadcast_Test_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h * w, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(C, E);
    auto G = net_builder.Add(D, E);
  }

  auto program = net_builder.Build();
  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(FusionMergePass, Broadcast_Test_4) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 4);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(FusionMergePass, Broadcast_Test_5) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 4);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 3);
}

TEST(FusionMergePass, Reduce_Test_0) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 4);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  // CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(FusionMergePass, Reduce_Test_1) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(FusionMergePass, Reduce_Test_2) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 2);
}

TEST(FusionMergePass, Reduce_Test_3) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 4);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  // CHECK_EQ(graph->fusion_groups.size(), 3);
}

TEST(FusionMergePass, Reduce_Test_4) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 5);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  // CHECK_EQ(graph->fusion_groups.size(), 3);
}

TEST(FusionMergePass, Reduce_Test_5) {
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
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);
}

}  // namespace frontend
}  // namespace cinn
