// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/utils/data_util.h"

PD_DEFINE_string(model_dir, "", "");

namespace cinn {
namespace frontend {

using hlir::framework::Scope;
using utils::Join;

TEST(const_conv, const_conv) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B", true);

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  auto c = program.conv2d(A, B, attrs);
  Target target = common::DefaultTarget();
  program.SetInputs({A, B});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "ConstPropagate");
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  auto scope = BuildScope(target, graph);

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();
  auto& prerun_instrs = runtime_program->GetPreRunInstructions();
  auto& run_instrs = runtime_program->GetRunInstructions();
  ASSERT_EQ(prerun_instrs.size(), 0);
  ASSERT_EQ(run_instrs.size(), 1);

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);

  runtime_program->Execute();
}

// fused_batch_norm
TEST(const_bn, const_bn) {
  Placeholder A(Float(32), {1, 64, 112, 112}, "A");

  Placeholder Scale(Float(32), {64}, "Scale", true);
  Placeholder Bias(Float(32), {64}, "Bias", true);
  Placeholder Mean(Float(32), {64}, "Mean", true);
  Placeholder Variance(Float(32), {64}, "Variance", true);

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["epsilon"] = static_cast<float>(0.001);
  auto a =
      program.fused_batchnorm_inference(A, Scale, Bias, Mean, Variance, attrs);

  Target target = common::DefaultTarget();
  program.SetInputs({A, Scale, Bias, Mean, Variance});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  auto scope = BuildScope(target, graph);

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();
  auto& prerun_instrs = runtime_program->GetPreRunInstructions();
  auto& run_instrs = runtime_program->GetRunInstructions();
  // Revert changes in PR #990 to pass the model unittests
  ASSERT_EQ(run_instrs.size(), 1);

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("Scale");
  scope->Var<hlir::framework::Tensor>("Bias");
  scope->Var<hlir::framework::Tensor>("Mean");
  scope->Var<hlir::framework::Tensor>("Variance");

  auto A1 = scope->GetTensor("A");
  auto Scale1 = scope->GetTensor("Scale");
  auto Bias1 = scope->GetTensor("Bias");
  auto Mean1 = scope->GetTensor("Mean");
  auto Variance1 = scope->GetTensor("Variance");
  SetRandData<float>(A1, target);
  SetRandData<float>(Scale1, target);
  SetRandData<float>(Bias1, target);
  SetRandData<float>(Mean1, target);
  SetRandData<float>(Variance1, target);

  runtime_program->Execute();
}

}  // namespace frontend
}  // namespace cinn
