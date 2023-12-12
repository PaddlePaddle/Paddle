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

#include "paddle/cinn/ir/group_schedule/st_shape_group_scheduler.h"

#include <gtest/gtest.h>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/decomposer/test_helper.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"

PD_DECLARE_bool(cinn_new_group_scheduler);

namespace cinn {
namespace ir {

using frontend::NetBuilder;
using frontend::RunDecomposer;

void Compile(NetBuilder* net_builder) {
  auto program = net_builder->Build();
  auto target = cinn::common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict =
      graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>(
          "inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<
      absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");

  auto op_lowerer =
      hlir::framework::CreateOpLowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_group : graph->fusion_groups) {
    std::vector<ir::LoweredFunc> lowered_funcs =
        op_lowerer.Lower(fusion_group,
                         /* apply_op_schedule = */ true,
                         /* apply_group_schedule = */ false);
    CHECK_EQ(lowered_funcs.size(), 1);
    VLOG(1) << "without group schedule, lowered_func: "
            << lowered_funcs.front();

    FLAGS_cinn_new_group_scheduler = true;
    lowered_funcs = op_lowerer.Lower(fusion_group,
                                     /* apply_op_schedule = */ true,
                                     /* apply_group_schedule = */ true);
    CHECK_EQ(lowered_funcs.size(), 1);
    VLOG(1) << "after group schedule, lowered_func: " << lowered_funcs.front();
  }
}

void CheckAccuracy(NetBuilder* net_builder,
                   const std::vector<std::string>& input_names) {
  FLAGS_cinn_new_group_scheduler = true;
  auto program = net_builder->Build();
  auto target = cinn::common::DefaultTarget();

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPasses(graph.get(),
                               {"OpFusionPass", "FusionMergePass"});

  VLOG(1) << "Before CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});
  hlir::framework::ApplyPasses(
      graph.get(), {"CheckFusionAccuracyPass", "TransToCustomCallPass"});
  VLOG(1) << "After CheckFusionAccuracyPass:\n"
          << graph->DebugGroupedGraph(std::unordered_set<std::string>{});

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);

  for (size_t i = 0; i < input_names.size(); ++i) {
    scope->Var<hlir::framework::Tensor>(input_names[i]);
    auto tensor = scope->GetTensor(input_names[i]);

    std::vector<float> vec;
    frontend::InitRandomVector<float>(
        &vec, tensor->shape().numel(), 0.0f, 1.0f);
    frontend::CopyFromVector<float>(vec, tensor, target);
  }

  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

// Each unittest below tests a single reduce,
// these unittests are only used to observe the generated IR and debug.
// Accuracy testing is guaranteed by Python unittests named
// test_reduce_op_xxx.py.
TEST(GROUP_SCHEDULER, last_reduce_only_1) {
  NetBuilder net_builder("last_reduce_only_1");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {128, 64, 32}, "A");
    auto B = net_builder.ReduceSum(A, {2});
  };

  CreateModel();
  Compile(&net_builder);
}

TEST(GROUP_SCHEDULER, last_reduce_only_2) {
  NetBuilder net_builder("last_reduce_only_2");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {1024}, "A");
    auto B = net_builder.ReduceSum(A, {0});
  };

  CreateModel();
  Compile(&net_builder);
}

TEST(GROUP_SCHEDULER, last_reduce_only_3) {
  NetBuilder net_builder("last_reduce_only_3");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {512, 256}, "A");
    auto B = net_builder.ReduceSum(A, {1});
  };

  CreateModel();
  Compile(&net_builder);
}

TEST(GROUP_SCHEDULER, non_last_reduce_only_1) {
  NetBuilder net_builder("non_last_reduce_only_1");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {10, 10, 10}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1}, /* keep_dim = */ true);
  };

  CreateModel();
  Compile(&net_builder);
}

TEST(GROUP_SCHEDULER, non_last_reduce_only_2) {
  NetBuilder net_builder("non_last_reduce_only_2");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {64, 32, 16, 8, 4}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2, 3}, /* keep_dim = */ true);
  };

  CreateModel();
  Compile(&net_builder);
}

TEST(GROUP_SCHEDULER, shuffle_reduce_only_1) {
  NetBuilder net_builder("shuffle_reduce_only_1");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {32, 32, 32, 32}, "A");
    auto B = net_builder.ReduceSum(A, {0, 2, 3});
  };

  CreateModel();
  Compile(&net_builder);
}

TEST(GROUP_SCHEDULER, shuffle_reduce_only_2) {
  NetBuilder net_builder("shuffle_reduce_only_2");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {32, 64, 56, 56}, "A");
    auto B = net_builder.ReduceSum(A, {0, 2, 3});
  };

  CreateModel();
  Compile(&net_builder);
}

// Each of the following unittest tests a basic pattern composed of multiple
// basic op. And apply accuracy checks to ensure that the results of fusion
// groups and independently running each op are consistent.
TEST(GROUP_SCHEDULER, elementwise_1) {
  int h = 128, w = 128;
  NetBuilder net_builder("elementwise_1");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.Add(B, C);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_2) {
  int h = 128, w = 128;
  NetBuilder net_builder("elementwise_2");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.Cast(C, "float16");
    auto E = net_builder.Cast(C, "float16");
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_3) {
  int h = 128, w = 128;
  NetBuilder net_builder("elementwise_3");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.Cast(C, "float16");
    auto E = net_builder.Cast(C, "float16");
    auto F = net_builder.Cast(D, "float32");
    auto G = net_builder.Cast(E, "float32");
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_4) {
  int h = 128, w = 128;
  NetBuilder net_builder("elementwise_4");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.Cast(C, "float16");
    auto E = net_builder.Cast(C, "float16");
    auto F = net_builder.Add(D, E);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_broadcast) {
  NetBuilder net_builder("elementwise_broadcast");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {128}, "A");
    auto B = net_builder.CreateInput(Float(32), {128}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.BroadcastTo(C, {128, 128});
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_double_broadcast) {
  NetBuilder net_builder("elementwise_double_broadcast");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {128}, "A");
    auto B = net_builder.CreateInput(Float(32), {128}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.BroadcastTo(C, {128, 128});
    auto E = net_builder.BroadcastTo(C, {128, 128});
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, non_last_reduce_elementwise_1) {
  int h = 128, w = 128;
  NetBuilder net_builder("non_last_reduce_elementwise_1");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0});
    auto C = net_builder.Cast(B, "float16");
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, last_reduce_elementwise) {
  NetBuilder net_builder("last_reduce_elementwise");
  std::vector<std::string> input_names = {"A", "C"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {128, 64}, "A");
    auto B = net_builder.ReduceSum(A, {1});
    auto C = net_builder.CreateInput(Float(32), {128}, "C");
    auto D = net_builder.Add(B, C);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, keep_dim_reduce_elementwise_1) {
  NetBuilder net_builder("keep_dim_reduce_elementwise");
  std::vector<std::string> input_names = {"A", "C"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 112, 112}, "A");
    auto B = net_builder.CreateInput(Float(32), {1, 64, 1, 1}, "B");
    auto C = net_builder.ReduceSum(A, {0, 2, 3}, true);
    auto D = net_builder.Add(B, C);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, keep_dim_reduce_elementwise_2) {
  NetBuilder net_builder("keep_dim_reduce_elementwise_2");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 112, 112}, "A");
    auto B = net_builder.CreateInput(Float(32), {16, 64, 1, 1}, "B");
    auto C = net_builder.ReduceSum(A, {2, 3}, true);
    auto D = net_builder.Add(B, C);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, keep_dim_reduce_elementwise_3) {
  NetBuilder net_builder("keep_dim_reduce_elementwise_3");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 2048}, "A");
    auto B = net_builder.CreateInput(Float(32), {16, 64, 1}, "B");
    auto C = net_builder.ReduceSum(A, {2}, true);
    auto D = net_builder.Add(B, C);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, keep_dim_reduce_elementwise_4) {
  NetBuilder net_builder("keep_dim_reduce_elementwise_4");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 2048}, "A");
    auto B = net_builder.CreateInput(Float(32), {16, 1, 2048}, "B");
    auto C = net_builder.ReduceSum(A, {1}, true);
    auto D = net_builder.Add(B, C);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, keep_dim_reduce_elementwise_5) {
  NetBuilder net_builder("keep_dim_reduce_elementwise_5");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {16, 64, 16, 1024}, "A");
    auto B = net_builder.CreateInput(Float(32), {16, 1, 16, 1}, "B");
    auto C = net_builder.ReduceSum(A, {1, 3}, true);
    auto D = net_builder.Add(B, C);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_non_last_reduce) {
  int h = 128, w = 128;
  NetBuilder net_builder("elementwise_non_last_reduce");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.ReduceSum(C, {0});
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_last_reduce) {
  int h = 128, w = 128;
  NetBuilder net_builder("elementwise_last_reduce");
  std::vector<std::string> input_names = {"A", "C"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.ReduceSum(C, {1});
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_non_last_reduce_elementwise) {
  int h = 128, w = 128;
  NetBuilder net_builder("elementwise_non_last_reduce_elementwise");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(C, {0});
    auto F = net_builder.Cast(E, "float16");
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_last_reduce_elementwise) {
  int h = 128, w = 128;
  NetBuilder net_builder("elementwise_non_last_reduce_elementwise");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(C, {1});
    auto F = net_builder.Cast(E, "float16");
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_double_non_last_reduce_elementwise) {
  int h = 128, w = 128;
  NetBuilder net_builder("elementwise_double_non_last_reduce_elementwise");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(C, {0});
    auto F = net_builder.ReduceSum(C, {0});
    auto G = net_builder.Add(E, F);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, double_non_last_reduce_elementwise) {
  int h = 128, w = 128;
  NetBuilder net_builder("double_non_last_reduce_elementwise");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h * 2, w}, "B");
    auto E = net_builder.ReduceSum(A, {0});
    auto F = net_builder.ReduceSum(B, {0});
    auto G = net_builder.Add(E, F);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, triple_non_last_reduce) {
  int h = 128, w = 1024;
  NetBuilder net_builder("triple_non_last_reduce");
  std::vector<std::string> input_names = {"A", "B"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {128, 1024}, "A");
    auto B = net_builder.ReduceSum(A, {0});
    auto C = net_builder.ReduceSum(A, {0});
    auto D = net_builder.ReduceSum(A, {0});
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, reduce_broadcast_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("reduce_broadcast_1");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h * w}, "A");
    auto B = net_builder.ReduceSum(A, {0});
    auto C = net_builder.BroadcastTo(B, {h * w}, {0});
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, reduce_broadcast_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("reduce_broadcast_2");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1});
    auto C = net_builder.BroadcastTo(B, {h, w}, {1});
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, reduce_broadcast_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("reduce_broadcast_3");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2});
    auto C = net_builder.BroadcastTo(B, {h, h, w}, {0});
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, reduce_broadcast_reduce_broadcast) {
  int h = 32, w = 32;
  NetBuilder net_builder("reduce_broadcast_reduce_broadcast");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2});
    auto C = net_builder.BroadcastTo(B, {h, h, w}, {0});
    auto D = net_builder.ReduceSum(C, {1, 2});
    auto E = net_builder.BroadcastTo(D, {h, h, w}, {0});
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, reduce_broadcast_elementwise) {
  int h = 32, w = 32;
  NetBuilder net_builder("reduce_broadcast_elementwise");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {h, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2});
    auto C = net_builder.BroadcastTo(B, {h, h, w}, {0});
    auto D = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto E = net_builder.BroadcastTo(D, {h, h, w}, {1, 2});
    auto F = net_builder.Add(C, E);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_double_reduce_elementwise_1) {
  NetBuilder net_builder("elementwise_double_reduce_elementwise_1");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {32, 32}, "A");
    auto B = net_builder.CreateInput(Float(32), {32, 32}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.ReduceSum(C, {1}, false);
    auto E = net_builder.ReduceSum(C, {1}, false);
    auto F = net_builder.Add(D, E);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, elementwise_double_reduce_elementwise_2) {
  NetBuilder net_builder("elementwise_double_reduce_elementwise_2");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    auto A = net_builder.CreateInput(Float(32), {1, 1000}, "A");
    auto B = net_builder.CreateInput(Float(32), {1, 1000}, "B");
    auto C = net_builder.Add(A, B);
    auto D = net_builder.ReduceSum(C, {1}, false);
    auto E = net_builder.ReduceSum(C, {1}, false);
    auto F = net_builder.Add(D, E);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

// Each of following unittests tests a group composed of typical operators
TEST(GROUP_SCHEDULER, layernorm) {
  int h = 32, w = 1024;
  NetBuilder net_builder("layernorm");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    // x
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    // x * x
    auto B = net_builder.Multiply(A, A);
    // sum x
    auto C = net_builder.ReduceSum(A, {1});
    // sum x*x
    auto D = net_builder.ReduceSum(B, {1});
    // constant w
    auto E = net_builder.FillConstant<float>({h}, 1024.0f, "E");
    // mean
    auto F = net_builder.Divide(C, E);
    auto FF = net_builder.BroadcastTo(F, {h, w}, {0});
    // mean x*x
    auto G = net_builder.Divide(D, E);
    // mean * mean
    auto H = net_builder.Multiply(F, F);
    // var^2
    auto I = net_builder.Subtract(G, H);
    // eps
    auto J = net_builder.FillConstant<float>({h}, 1e-10f, "J");
    // eps + delta
    auto K = net_builder.Add(I, J);
    // var
    auto L = net_builder.Sqrt(K);
    auto LL = net_builder.BroadcastTo(L, {h, w}, {0});
    // x - mean
    auto M = net_builder.Subtract(A, FF);
    // /var
    auto N = net_builder.Divide(M, LL);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

TEST(GROUP_SCHEDULER, softmax) {
  int h = 32, w = 1024;
  NetBuilder net_builder("softmax");
  std::vector<std::string> input_names = {"A"};
  // create model
  auto CreateModel = [&]() {
    // softmax
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    // reduce max
    auto B = net_builder.ReduceMax(A, {1});
    // broadcast
    auto C = net_builder.BroadcastTo(B, {h, w}, {0});
    // x - max(x)
    auto D = net_builder.Subtract(A, C);
    // exp(x)
    auto E = net_builder.Exp(D);
    // reduce sum
    auto F = net_builder.ReduceSum(E, {1});
    // broadcast
    auto G = net_builder.BroadcastTo(F, {h, w}, {0});
    // exp(x)/sum(exp(x))
    auto H = net_builder.Divide(E, G);
  };

  CreateModel();
  Compile(&net_builder);
  CreateModel();
  CheckAccuracy(&net_builder, input_names);
}

}  // namespace ir
}  // namespace cinn
