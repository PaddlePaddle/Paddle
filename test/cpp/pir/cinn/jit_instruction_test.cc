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
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/new_executor/interpretercore.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"

#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/convert_to_dialect.h"
#include "paddle/cinn/hlir/framework/new_ir_compiler.h"
#include "paddle/cinn/utils/data_util.h"

std::unique_ptr<::pir::Program> BuildProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  auto program = std::make_unique<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const float value = 2.0;
  auto full_op_x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 128},
                                             value,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());

  auto full_op_y =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{128, 64},
                                             value,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());
  return std::move(program);
}

namespace paddle {
namespace framework {

TEST(CinnJitInstruction, Run) {
  // Step 1: Construct pir::Program
  std::unique_ptr<::pir::Program> program = BuildProgram();
  EXPECT_EQ(program->block()->size(), 2u);

  // Step 2: Compiler New pir::Program into Runtime Program
  auto target = cinn::common::DefaultNVGPUTarget();
  auto scope = cinn::hlir::framework::BuildScope(target, *program);
  ASSERT_EQ(scope->var_names().size(), 2);

  cinn::hlir::framework::NewIRCompiler ir_compiler(*program, target, scope);
  auto runtime_program = ir_compiler.Build();

  // Step 3: Convert into cinn::dialect::RuntimeDialect
  std::unique_ptr<::pir::Program> ir_runtime_program =
      cinn::hlir::framework::ConvertToRuntimeDialect(*runtime_program);

  std::set<std::string> out_names;
  for (auto& var_name : scope->var_names()) {
    std::string name = {var_name.begin(), var_name.end()};
    out_names.insert(name);
  }

  platform::Place place = platform::CUDAPlace(0);
  Scope exe_scope;

  InterpreterCore executor(place, {}, ir_runtime_program->block(), &exe_scope);
  executor.SetSkipGcVars(out_names);
  executor.Run({});

  // TODO(Aurelius84): Need to replace check with framework::Scope.
  const float value = 2.0;
  for (auto& name : out_names) {
    std::vector<float> data =
        cinn::GetTensorData<float>(scope->GetTensor(name), target);
    for (int i = 0; i < data.size(); ++i) {
      LOG_FIRST_N(INFO, 3) << "data: " << data[i];
      ASSERT_NEAR(data[i], value, 1e-5);
    }
  }
}

}  // namespace framework
}  // namespace paddle
