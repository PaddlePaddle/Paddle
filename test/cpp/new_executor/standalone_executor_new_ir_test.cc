// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/standalone_executor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <string>

#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/fluid/ir/pass/pd_op_to_kernel_pass.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"

#include "paddle/fluid/platform/init_phi.h"

DECLARE_FILE_SYMBOLS(kernel_dialect);

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full_int_array, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(uniform, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);

namespace paddle {
namespace framework {

TEST(StandaloneExecutor, run) {
  std::cerr << "here" << std::endl;

  ir::IrContext* ctx = ir::IrContext::Instance();
  ir::Program program((ctx));

  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

  ir::Builder builder = ir::Builder::AtBlockEnd(ctx, program.block());

  paddle::dialect::FullOp op1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::FullOp op2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  builder.Build<paddle::dialect::AddOp>(op1->GetResultByIndex(0),
                                        op2->GetResultByIndex(0));

  program.Print(std::cout);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  kernel_program->Print(std::cout);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(
      place, prog_desc.Block(0), &scope, kernel_program.get());

  test_core.Run({});

  auto tensor = scope.Var("inner_var_2")->Get<phi::DenseTensor>();

  std::cerr << "uot" << tensor << std::endl;
}

}  // namespace framework
}  // namespace paddle
