
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

#include "paddle/cinn/auto_schedule/measure/simple_runner.h"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::BuildScope;
using ::cinn::hlir::framework::CompilationContext;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::GraphCompiler;
using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Scope;

class TestSimpleRunner : public ::testing::Test {
 public:
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::shared_ptr<Graph> graph;
  std::shared_ptr<Scope> compiled_scope;
  std::unique_ptr<GraphCompiler> graph_compiler;
  std::unique_ptr<TuneTask> task;

  MeasureInput input;
  BuildResult build_result;

  static frontend::Program CreateAddReluProgram();
  void SetUp() override {
    std::unordered_set<std::string> fetch_ids;
    auto program = CreateAddReluProgram();
    auto graph = cinn::frontend::Optimize(&program, fetch_ids, target);
    compiled_scope = BuildScope(target, graph);
    CompilationContext context(graph, compiled_scope, target);
    graph_compiler = std::make_unique<GraphCompiler>(context);
    auto runtime_program = graph_compiler->Build();
    const auto& instructions = runtime_program->GetRunInstructions();
    ASSERT_EQ(1, instructions.size());

    build_result.compiled_scope = compiled_scope.get();
    build_result.runtime_program = std::move(runtime_program);

    task = std::make_unique<TuneTask>();
#ifdef CINN_WITH_CUDA
    task->target = common::DefaultNVGPUTarget();
#else
    task->target = common::DefaultHostTarget();
#endif
    task->subgraph = graph->fusion_groups.front();
    input.task = task.get();
  }
};

frontend::Program TestSimpleRunner::CreateAddReluProgram() {
  constexpr int M = 32;
  constexpr int N = 24;
  frontend::NetBuilder builder("test");

  auto a = builder.CreateInput(Float(32), {M, N}, "A");
  auto b = builder.CreateInput(Float(32), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Relu(c);
  return builder.Build();
}

TEST_F(TestSimpleRunner, MeasureWithRandomValue) {
  auto runner = std::make_unique<SimpleRunner>(1);
  ASSERT_NO_THROW(runner->Run(input, build_result));
}

TEST_F(TestSimpleRunner, MeasureWithSpecifiedArgs) {
  auto ta = compiled_scope->GetTensor("A");
  ta->mutable_data<float>(target);
  auto tb = compiled_scope->GetTensor("B");
  tb->mutable_data<float>(target);
  std::map<std::string, cinn_pod_value_t> preset_args;
  preset_args.emplace("A", ta->buffer());
  preset_args.emplace("B", tb->buffer());

  auto runner = std::make_unique<SimpleRunner>(1);
  // specific several execution args
  input.execution_args = &preset_args;
  ASSERT_NO_THROW(runner->Run(input, build_result));
}

TEST_F(TestSimpleRunner, TimeMeasured) {
  // set up a BuildResult object with one instruction of the `sleep` function
  void (*sleep_fn)(void*, int32_t) = [](void*, int32_t) -> void {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  };
  BuildResult build_result;
  build_result.compiled_scope = nullptr;
  std::vector<std::unique_ptr<Instruction>> instructions;
  instructions.emplace_back(new Instruction(common::DefaultHostTarget(),
                                            nullptr,
                                            {},
                                            {"empty_placeholder"},
                                            "sleep_fn"));
  instructions.back()->SetLoweredFunc(reinterpret_cast<void*>(sleep_fn));
  instructions.back()->Finalize();
  build_result.runtime_program = std::make_unique<hlir::framework::Program>(
      nullptr, std::move(instructions));

  // to skip the condition check of params in Instruction::PreparePodArgs
  std::map<std::string, cinn_pod_value_t> preset_args;
  preset_args.emplace("empty_placeholder", cinn_pod_value_t());
  input.execution_args = &preset_args;

  auto runner = std::make_unique<SimpleRunner>(2);
  MeasureResult measure_result = runner->Run(input, build_result);
  // because the kernel function will sleep 100 us,
  // the cost time of execution and span in total must
  // be greater than 100us and 200us (repeatedly running 2 times) respectively.
  ASSERT_GE(measure_result.execution_cost, 100);
  ASSERT_GE(measure_result.elapsed_time, 200);
}

}  // namespace auto_schedule
}  // namespace cinn
