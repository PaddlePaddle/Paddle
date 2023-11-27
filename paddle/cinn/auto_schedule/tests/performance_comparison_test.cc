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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <bitset>
#include <iostream>

#include "paddle/cinn/auto_schedule/auto_tuner.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/paddle_model_convertor.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/data_util.h"
#include "test/cpp/cinn/program_builder.h"

/* This test is used as a tool to evaluate or compare performance of 3
 * schedules(no schedule, manual schedule, auto-schedule). One can specify which
 * schedules to be evaluated through `FLAGS_evaluate_knobs` and specify which
 * operator or model through `--gtest_filter=PerformanceTester.xx`, for example,
 * `FLAGS_evaluate_knobs=4
 * --gtest_filter=PerformanceTester.Matmul` means it will evaluate auto-schedule
 * on Matmul operator. You can refer to explanation of following flags or
 * parameters for more detail.
 */

PD_DEFINE_string(resnet50_model_dir,
                 "./ResNet50",
                 "the path to paddle model resnet50.");
// Flags that control which schedule tests will be run.
// Bit with index 0 controls no schedule test, means options = 1 = "001" will
// run no schedule test. Bit with index 1 controls manual schedule test, means
// options = 2 = "010" will run manual schedule test. Bit with index 2 controls
// auto schedule test, means options = 4 = "100" will run auto schedule test.
// The default value is -1, which means that this flag is disabled to set the
// options
PD_DEFINE_int32(evaluate_knobs,
                -1,
                "the options to control which schedule tests will be run.");
PD_DECLARE_double(cinn_infer_model_version);

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::BuildScope;
using ::cinn::hlir::framework::CompilationContext;
using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::GraphCompiler;
using ::cinn::hlir::framework::Instruction;
using ::cinn::hlir::framework::Scope;

class PerformanceTester : public ::testing::Test {
 public:
  struct Options {
    // times of compiled runtime program will be executed repeatedly.
    int repeat_times = 2;
    // the num_tuning_rounds for auto tuning
    int num_tuning_rounds = 2;
    // knobs to control which schedules will be measured, refer to
    // FLAGS_evaluate_knobs explanation
    std::bitset<3> evaluate_knobs = 0UL;
  };

  void Evaluate(const frontend::Program& program) {
    if (FLAGS_evaluate_knobs >= 0) {
      options_.evaluate_knobs = FLAGS_evaluate_knobs;
    }
    VLOG(3) << "evaluate_knobs = " << options_.evaluate_knobs;

    auto worker_fn = [this, &program](const std::string& schedule_name,
                                      BuildRuntimeProgramFn build_fn,
                                      bool execute = true) {
      Context::Global().ResetNameId();
      VLOG(3) << "Initialize graph.";
      auto graph = std::make_shared<hlir::framework::Graph>(program, target_);
      VLOG(3) << "Apply graph pass.";
      hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
      VLOG(3) << "Build " << schedule_name << " program.";
      auto scope = BuildScope(target_, graph);
      CompilationContext context(graph, scope, target_);
      auto graph_compiler = std::make_unique<GraphCompiler>(context);
      auto runtime_program =
          (this->*build_fn)(graph.get(), graph_compiler.get());
      if (execute) {
        VLOG(3) << "Execute " << schedule_name << " program.";
        runtime_program->ExecuteTest(options_.repeat_times);
      }
    };

    // if no one is set, build no/manual schedule cases to ensure their build
    // functions are valid
    if (options_.evaluate_knobs.none()) {
      worker_fn("no schedule",
                &PerformanceTester::BuildNoScheduleProgram,
                /* execute */ false);
      worker_fn("manual schedule",
                &PerformanceTester::BuildManualScheduleProgram,
                /* execute */ false);
    } else {
      if (options_.evaluate_knobs.test(0)) {
        worker_fn("no schedule", &PerformanceTester::BuildNoScheduleProgram);
      }
      if (options_.evaluate_knobs.test(1)) {
        worker_fn("manual schedule",
                  &PerformanceTester::BuildManualScheduleProgram);
      }
      if (options_.evaluate_knobs.test(2)) {
        worker_fn("auto schedule",
                  &PerformanceTester::BuildAutoScheduleProgram);
      }
    }
  }

 protected:
  using BuildRuntimeProgramFn = std::unique_ptr<hlir::framework::Program> (
      PerformanceTester::*)(Graph*, GraphCompiler*);

  std::unique_ptr<hlir::framework::Program> BuildNoScheduleProgram(
      Graph* graph, GraphCompiler* graph_compiler) {
    const auto& dtype_dict =
        graph->GetAttrs<absl::flat_hash_map<std::string, common::Type>>(
            "inferdtype");
    const auto& shape_dict = graph->GetAttrs<
        absl::flat_hash_map<std::string, hlir::framework::shape_t>>(
        "infershape");

    auto op_lowerer =
        hlir::framework::CreateOpLowerer(dtype_dict, shape_dict, target_);

    CompilationContext& context = graph_compiler->GetCompilationContext();
    context.with_instantiate_variables = true;

    if (graph->fusion_groups.empty()) {
      hlir::framework::ApplyPasses(graph, {"BuildNonFusedGroupsPass"});
    }
    context.groups = graph->fusion_groups;

    for (auto group : graph->fusion_groups) {
      context.lowered_funcs.push_back(
          op_lowerer.Lower(group,
                           /*apply_op_schedule = */ false,
                           /*apply_group_schedule=*/false));
    }

    VLOG(3) << "===========================No Schedule LoweredFunc "
               "Begin===========================";
    for (const auto& funcvec : context.lowered_funcs) {
      for (const auto& func : funcvec) {
        VLOG(3) << func;
      }
    }
    VLOG(3) << "===========================No Schedule LoweredFunc "
               "End=============================";

    return graph_compiler->Build();
  }

  std::unique_ptr<hlir::framework::Program> BuildManualScheduleProgram(
      Graph* graph, GraphCompiler* graph_compiler) {
    return graph_compiler->Build();
  }

  std::unique_ptr<hlir::framework::Program> BuildAutoScheduleProgram(
      Graph* graph, GraphCompiler* graph_compiler) {
    auto tuner = std::make_unique<AutoTuner>(target_, graph);

    AutoTuner::Config tuning_config;
    TuningOptions tuning_options;
    tuning_options.num_tuning_rounds = options_.num_tuning_rounds;
    tuning_options.num_measure_trials = 2;
    tuning_options.num_samples_per_iteration = 2;

    tuner->Initialize(tuning_config, graph_compiler);
    TuningResult tuning_result = tuner->Tune(tuning_options);

    CompilationContext& context = graph_compiler->GetCompilationContext();
    context.with_instantiate_variables = true;
    context.ApplyTuningResult(tuning_result);

    VLOG(3) << "===========================Auto Schedule LoweredFunc "
               "Begin===========================";
    for (const auto& funcvec : context.lowered_funcs) {
      for (const auto& func : funcvec) {
        VLOG(3) << func;
      }
    }
    VLOG(3) << "===========================Auto Schedule LoweredFunc "
               "End=============================";

    return graph_compiler->Build();
  }

#ifdef CINN_WITH_CUDA
  Target target_ = common::DefaultNVGPUTarget();
#else
  Target target_ = common::DefaultHostTarget();
#endif
  Options options_;
};

constexpr int batch_size = 2;

TEST_F(PerformanceTester, Mul) {
  Evaluate(tests::OpBuilder("mul").Build({{"X", {32, 16}}, {"Y", {16, 32}}}));
}

TEST_F(PerformanceTester, Add) {
  Evaluate(tests::OpBuilder("elementwise_add")
               .Build({{"X", {1, 56, 56, 256}}, {"Y", {1, 56, 56, 256}}}));
}

TEST_F(PerformanceTester, Matmul) {
  Evaluate(tests::OpBuilder("matmul").Build(
      {{"X", {batch_size, 2048}}, {"Y", {2048, 1000}}}));
}

TEST_F(PerformanceTester, Relu) {
  Evaluate(tests::OpBuilder("relu").Build({{"X", {batch_size, 64, 56, 56}}}));
}

TEST_F(PerformanceTester, Conv2d) {
  std::vector<int> strides{2, 2};
  std::vector<int> paddings{3, 3};
  std::vector<int> dilations{1, 1};
  int groups = 1;
  std::string conv_type = "forward";
  std::string data_format = "NCHW";
  std::string padding_algorithm = "EXPLICIT";

  Evaluate(tests::OpBuilder("conv2d").Build(
      {{"X", {batch_size, 3, 224, 224}}, {"W", {64, 3, 7, 7}}},
      {{"stride", strides},
       {"padding", paddings},
       {"dilation", dilations},
       {"groups", groups},
       {"conv_type", conv_type},
       {"data_format", data_format},
       {"padding_algorithm", padding_algorithm}}));
}

TEST_F(PerformanceTester, Pool2d) {
  std::vector<int32_t> input_shape{batch_size, 64, 112, 112};
  std::string pooling_type = "max";
  std::vector<int> ksize{3, 3};
  std::vector<int> strides{2, 2};
  std::vector<int> paddings{1, 1, 1, 1};
  bool ceil_mode = false;
  bool exclusive = true;
  bool global_pooling = false;
  std::string data_format = "NCHW";
  bool adaptive = false;
  std::string padding_algorithm = "EXPLICIT";

  Evaluate(tests::OpBuilder("pool2d").Build(
      {{"X", {batch_size, 64, 112, 112}}},
      {{"pool_type", pooling_type},
       {"kernel_size", ksize},
       {"stride_size", strides},
       {"padding_size", paddings},
       {"ceil_mode", ceil_mode},
       {"exclusive", exclusive},
       {"global_pooling", global_pooling},
       {"data_format", data_format},
       {"adaptive", adaptive},
       {"padding_algorithm", padding_algorithm}}));
}

TEST_F(PerformanceTester, BatchNorm) {
  std::vector<int32_t> input_shape{batch_size, 64, 112, 112};
  std::vector<int32_t> scale_shape{64};
  std::vector<int32_t> bias_shape{64};
  std::vector<int32_t> mean_shape{64};
  std::vector<int32_t> variance_shape{64};
  float epsilon = 1e-5f;
  float momentum = 0.9f;
  const std::string& data_layout = "NCHW";

  Evaluate(tests::OpBuilder("batch_norm")
               .Build({{"X", {batch_size, 64, 112, 112}},
                       {"scale", {64}},
                       {"bias", {64}},
                       {"mean", {64}},
                       {"variance", {64}}},
                      {{"epsilon", epsilon},
                       {"momentum", momentum},
                       {"data_layout", data_layout}}));
}

TEST_F(PerformanceTester, Reshape) {
  std::vector<int32_t> output_shape{batch_size, 2048};

  Evaluate(tests::OpBuilder("reshape").Build({{"X", {batch_size, 2048, 1, 1}}},
                                             {{"shape", output_shape}}));
}

TEST_F(PerformanceTester, Softmax) {
  std::vector<int> axes = {-1};
  std::string mode = "fast";
  std::string data_format = "AnyLayout";

  Evaluate(tests::OpBuilder("softmax").Build(
      {{"X", {batch_size, 1000}}},
      {{"axes", axes}, {"mode", mode}, {"data_format", data_format}}));
}

TEST_F(PerformanceTester, Scale) {
  float scale = 1.0f;
  float bias = 0.0f;
  bool bias_after_scale = true;

  Evaluate(tests::OpBuilder("scale").Build(
      {{"X", {batch_size, 1000}}},
      {{"scale", scale},
       {"bias", bias},
       {"bias_after_scale", bias_after_scale}}));
}

TEST_F(PerformanceTester, LookupTable) {
  int64_t padding_idx = -1;

  Evaluate(tests::OpBuilder("lookup_table")
               .Build({{"table", {50001, 768}},
                       {"ids", {10, 128, 1}, common::Int(64)}},
                      {{"padding_idx", padding_idx}}));
}

TEST_F(PerformanceTester, Gather) {
  int axis = 3;

  Evaluate(tests::OpBuilder("gather").Build(
      {{"operand", {10, 12, 128, 512}},
       {"index", {1, 1, 1, 128}, common::Int(32)}},
      {{"axis", axis}}));
}

// paddle model test
TEST_F(PerformanceTester, ResNet50) {
  CHECK_NE(FLAGS_resnet50_model_dir, "");
  FLAGS_cinn_infer_model_version = 1.0;
  std::unordered_map<std::string, std::vector<int64_t>> feeds = {
      {"inputs", {batch_size, 3, 224, 224}}};
  Evaluate(cinn::frontend::PaddleModelConvertor(common::DefaultNVGPUTarget())
               .LoadModel(FLAGS_resnet50_model_dir, true, feeds));
}

}  // namespace auto_schedule
}  // namespace cinn
