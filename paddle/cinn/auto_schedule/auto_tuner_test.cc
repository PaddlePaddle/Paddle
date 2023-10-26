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

#include "paddle/cinn/auto_schedule/auto_tuner.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/runtime/flags.h"

PD_DECLARE_bool(auto_schedule_use_cost_model);

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::BuildScope;
using ::cinn::hlir::framework::CompilationContext;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::GraphCompiler;
using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Node;
using ::cinn::hlir::framework::Scope;

class TestAutoTuner : public ::testing::Test {
 public:
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::shared_ptr<Graph> graph;
  std::shared_ptr<Scope> compiled_scope;
  CompilationContext context;
  std::unique_ptr<GraphCompiler> graph_compiler;
  std::unique_ptr<AutoTuner> tuner;

  frontend::Program CreateAddReluProgram() {
    frontend::NetBuilder builder("test");

    auto a = builder.CreateInput(Float(32), {1, 64, 112, 112}, "A");
    auto b = builder.CreateInput(Float(32), {64}, "B");
    auto c = builder.Add(a, b, 1);
    auto d = builder.Relu(c);

    return builder.Build();
  }

  void SetUp() override {
    srand(0);
    std::unordered_set<std::string> fetch_ids;
    auto program = CreateAddReluProgram();
    auto graph = cinn::frontend::Optimize(&program, fetch_ids, target);
    compiled_scope = BuildScope(target, graph);
    context.graph = graph;
    context.scope = compiled_scope;
    context.target = target;
    graph_compiler = std::make_unique<GraphCompiler>(context);
    tuner = std::make_unique<AutoTuner>(target, graph.get());
  }

  TuningResult InitializeAndTune(const AutoTuner::Config& config,
                                 const TuningOptions& options) {
    tuner->Initialize(config, graph_compiler.get());
    return tuner->Tune(options);
  }

  virtual void BasicCheckResult(const TuningResult& result) {
    ASSERT_EQ(1, result.subgraphs.size());
    auto nodes = result.subgraphs.front()->CollectNodes();
    ASSERT_EQ(nodes.size(), 4UL);
    ASSERT_EQ(nodes[0]->op()->name, "broadcast_to");
    ASSERT_EQ(nodes[1]->op()->name, "fill_constant");
    ASSERT_EQ(nodes[2]->op()->name, "elementwise_add");
    ASSERT_EQ(nodes[3]->op()->name, "max");

    ASSERT_EQ(result.function_groups.size(), 1UL);
    ASSERT_EQ(result.function_groups[0].size(), 1UL);
  }

  virtual void ApplyTunedAndRun(const TuningResult& result) {
    // build runtime program with tuning result
    context.with_instantiate_variables = true;
    context.ApplyTuningResult(result);
    ASSERT_EQ(1, context.groups.size());
    ASSERT_EQ(1, context.lowered_funcs.size());
    VLOG(6) << "Print lowered_funcs before building";
    VLOG(6) << context.lowered_funcs[0][0];
    VLOG(6) << context.lowered_funcs[1][0];
    auto runtime_program = graph_compiler->Build(&context).runtime_program;
    ASSERT_EQ(1, runtime_program->size());
    runtime_program->Execute();
  }

  void ZeroMeasure() {
    // set config and options
    AutoTuner::Config tuning_config;
    tuning_config.task_schedule_strategy = "round_robin";

    TuningOptions tuning_options;
    tuning_options.num_measure_trials = 0;
    auto result = InitializeAndTune(tuning_config, tuning_options);
    BasicCheckResult(result);
    ApplyTunedAndRun(result);
  }

  void NonZeroMeasure() {
    // set config and options
    AutoTuner::Config tuning_config;
    tuning_config.task_schedule_strategy = "round_robin";

    TuningOptions tuning_options;
    tuning_options.num_measure_trials = 4;
    tuning_options.num_samples_per_iteration = 2;

    auto result = InitializeAndTune(tuning_config, tuning_options);
    BasicCheckResult(result);
    ApplyTunedAndRun(result);
  }
};

TEST_F(TestAutoTuner, ZeroMeasure_DisableCostModel) {
  FLAGS_auto_schedule_use_cost_model = false;
  ZeroMeasure();
}

TEST_F(TestAutoTuner, ZeroMeasure_EnableCostModel) {
  FLAGS_auto_schedule_use_cost_model = true;
  ZeroMeasure();
}

TEST_F(TestAutoTuner, NonZeroMeasure_DisableCostModel) {
  FLAGS_auto_schedule_use_cost_model = false;
  NonZeroMeasure();
}

TEST_F(TestAutoTuner, NonZeroMeasure_EnableCostModel) {
  FLAGS_auto_schedule_use_cost_model = true;
  NonZeroMeasure();
}

}  // namespace auto_schedule
}  // namespace cinn
