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

#include <memory>

#include "paddle/cinn/auto_schedule/measure/schedule_measurer.h"
#include "paddle/cinn/auto_schedule/measure/simple_builder.h"
#include "paddle/cinn/auto_schedule/measure/simple_runner.h"
#include "paddle/cinn/auto_schedule/task/task_creator.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/runtime/flags.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::BuildScope;
using ::cinn::hlir::framework::CompilationContext;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::GraphCompiler;

frontend::Program CreateAddReluProgram() {
  constexpr int M = 32;
  constexpr int N = 24;
  frontend::NetBuilder builder("test");

  auto a = builder.CreateInput(Float(32), {M, N}, "A");
  auto b = builder.CreateInput(Float(32), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Relu(c);
  return builder.Build();
}

class TestMeasurer : public ::testing::Test {
 public:
  std::unique_ptr<GraphCompiler> graph_compiler;
  std::vector<TuneTask> tasks;
  std::vector<MeasureInput> inputs;

  void SetUp() override {
#ifdef CINN_WITH_CUDA
    Target target = common::DefaultNVGPUTarget();
#else
    Target target = common::DefaultHostTarget();
#endif
    std::unordered_set<std::string> fetch_ids;
    auto program = CreateAddReluProgram();
    auto graph = cinn::frontend::Optimize(&program, fetch_ids, target);
    auto scope = BuildScope(target, graph);
    CompilationContext context(graph, scope, target);
    graph_compiler = std::make_unique<GraphCompiler>(context);
    TaskCreator task_creator;
    tasks = task_creator.CreateTuneTaskOpLevel(graph.get());
    const auto& dtype_dict =
        graph->GetAttrs<absl::flat_hash_map<std::string, common::Type>>(
            "inferdtype");
    const auto& shape_dict = graph->GetAttrs<
        absl::flat_hash_map<std::string, hlir::framework::shape_t>>(
        "infershape");

    auto op_lowerer =
        hlir::framework::CreateOpLowerer(dtype_dict, shape_dict, target);
    inputs.reserve(tasks.size());
    for (int i = 0; i < tasks.size(); ++i) {
      auto* task = &tasks[i];
      task->Initialize(shape_dict, dtype_dict, &op_lowerer);
      MeasureInput input;
      input.task = task;
      input.lowered_funcs = task->lowered_funcs;
      inputs.emplace_back(input);
    }
  }
};

class ThrowExceptionBuilder : public ScheduleBuilder {
  struct Exception : public std::exception {
    const char* what() const throw() { return "BuildError"; }
  };
  BuildResult Build(const MeasureInput& input) override { throw Exception(); }
};

class ThrowExceptionRunner : public ScheduleRunner {
  struct Exception : public std::exception {
    const char* what() const throw() { return "RunError"; }
  };
  MeasureResult Run(const MeasureInput& input,
                    const BuildResult& build_result) override {
    throw Exception();
  }
};

TEST_F(TestMeasurer, Basic) {
  auto builder = std::make_unique<SimpleBuilder>(graph_compiler.get());
  auto runner = std::make_unique<SimpleRunner>(1);
  auto measurer =
      std::make_unique<ScheduleMeasurer>(builder.get(), runner.get());
  std::vector<MeasureResult> results = measurer->Measure(inputs);
  ASSERT_EQ(inputs.size(), results.size());
}

TEST_F(TestMeasurer, CatchException) {
  auto builder = std::make_unique<SimpleBuilder>(graph_compiler.get());
  auto runner = std::make_unique<SimpleRunner>(1);
  auto throw_builder = std::make_unique<ThrowExceptionBuilder>();
  auto throw_runner = std::make_unique<ThrowExceptionRunner>();
  auto measurer_with_build_error =
      std::make_unique<ScheduleMeasurer>(throw_builder.get(), runner.get(), 2);
  std::vector<MeasureResult> results =
      measurer_with_build_error->Measure(inputs);
  ASSERT_EQ(inputs.size(), results.size());
  EXPECT_EQ(results[0].error_msg, "Build failed, error: BuildError\n");

  // TODO(CtfGo): test parallel build after we support thread-safe compilation
  auto measurer_with_run_error =
      std::make_unique<ScheduleMeasurer>(builder.get(), throw_runner.get(), 1);
  results = measurer_with_run_error->Measure(inputs);
  ASSERT_EQ(inputs.size(), results.size());
  EXPECT_EQ(results[0].error_msg, "Run failed, error: RunError\n");
}

}  // namespace auto_schedule
}  // namespace cinn
