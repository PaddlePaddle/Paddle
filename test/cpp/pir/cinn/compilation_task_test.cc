// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/framework/pir/compilation_task.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"

using cinn::hlir::framework::pir::CompatibleInfo;
using cinn::hlir::framework::pir::OpLoweringGroup;
using cinn::hlir::framework::pir::OpLoweringGroupPtr;

using ProgramInfo = std::tuple<std::shared_ptr<::pir::Program>,
                               std::vector<OpLoweringGroupPtr>>;
ProgramInfo BuildProgram(std::vector<int64_t> input_shape) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const float value_one = 1.0;
  auto full_op_x = builder.Build<paddle::dialect::FullOp>(
      input_shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace());

  std::vector<OpLoweringGroupPtr> groups;
  const std::string fn_name = CompatibleInfo::GroupOpsName(
      std::initializer_list<::pir::Operation*>({full_op_x.operation()}));
  groups.emplace_back(std::make_shared<OpLoweringGroup>(
      std::initializer_list<::pir::Operation*>({full_op_x.operation()}),
      fn_name));
  groups.back()->mut_output_ops().insert(full_op_x.operation());

  return {program, groups};
}

// TODO(LiuYang): This test is temporarily
// TEST(CompilationTask, Basic) {
//   auto prog_info = BuildProgram({4096, 128});
//   std::shared_ptr<::pir::Program> program = std::get<0>(prog_info);
//   LOG(INFO) << program->block()->size();
//   EXPECT_EQ(program->block()->size(), 1u);

//   std::stringstream ss;
//   program->Print(ss);
//   LOG(INFO) << ss.str();

//   auto target = cinn::common::DefaultNVGPUTarget();
//   auto scope = std::make_shared<cinn::hlir::framework::Scope>();

//   std::vector<GroupPtr> groups = std::get<1>(prog_info);
//   CHECK_EQ(groups.size(), 1);
//   cinn::hlir::framework::GroupCompilationContext compilation_context(
//       target, groups[0], scope);
//   cinn::hlir::framework::CompilationTask
//   compilation_task(&compilation_context); compilation_task.Lowering();
//   LOG(INFO) << compilation_context.PrintPredicate2Funcs();

//   compilation_task.CodegenAndJit();
//   auto instruction = compilation_task.BuildInstruction();
// }

// TEST(CompilationTask, CompileGroup) {
//   // Step 1: Construct pir::Program
//   int M = 4096, N = 128;
//   auto prog_info = BuildProgram({M, N});
//   std::shared_ptr<::pir::Program> program = std::get<0>(prog_info);
//   LOG(INFO) << program->block()->size();
//   EXPECT_EQ(program->block()->size(), 1u);

//   std::stringstream ss;
//   program->Print(ss);
//   LOG(INFO) << ss.str();

//   auto target = cinn::common::DefaultNVGPUTarget();
//   auto scope = std::make_shared<cinn::hlir::framework::Scope>();

//   std::vector<GroupPtr> groups = std::get<1>(prog_info);
//   CHECK_EQ(groups.size(), 1);

//   cinn::hlir::framework::PirCompiler ir_compiler(*program, target, scope);
//   auto runtime_program = ir_compiler.Build(groups);

//   // Step 3: Execute Runtime Instruction and check Scope.
//   ASSERT_NO_THROW(runtime_program->Execute());
//   for (auto& var_name : scope->var_names()) {
//     std::string name = {var_name.begin(), var_name.end()};
//     int64_t numel = scope->GetTensor(name)->shape().numel();
//     ASSERT_EQ(numel, M * N);
//     std::vector<float> data =
//         cinn::GetTensorData<float>(scope->GetTensor(name), target);
//     for (int i = 0; i < numel; ++i) {
//       ASSERT_EQ(data[i], 1.0);
//     }
//   }
// }
