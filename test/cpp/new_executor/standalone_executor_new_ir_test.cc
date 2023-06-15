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

#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/platform/init_phi.h"

DECLARE_FILE_SYMBOLS(kernel_dialect);

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full_int_array, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(uniform, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);

namespace paddle {
namespace framework {

// TEST(StandaloneExecutor, run) {
//   std::cerr << "here" << std::endl;

//   ir::IrContext* ctx = ir::IrContext::Instance();
//   ir::Program program((ctx));

//   ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

//   ir::Builder builder = ir::Builder(ctx, program.block());

//   paddle::dialect::FullOp op1 = builder.Build<paddle::dialect::FullOp>(
//       std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32,
//       phi::CPUPlace());

//   paddle::dialect::FullOp op2 = builder.Build<paddle::dialect::FullOp>(
//       std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32,
//       phi::CPUPlace());

//   builder.Build<paddle::dialect::AddOp>(op1->result(0), op2->result(0));

//   program.Print(std::cout);

//   auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

//   kernel_program->Print(std::cout);

//   auto place = platform::CPUPlace();
//   Scope scope;

//   ProgramDesc prog_desc;
//   InterpreterCore test_core(
//       place, prog_desc.Block(0), &scope, kernel_program.get());

//   test_core.Run({});

//   auto tensor = scope.Var("inner_var_2")->Get<phi::DenseTensor>();

//   std::cerr << "uot" << tensor << std::endl;
// }

TEST(StandaloneExecutor, run) {
  std::cerr << "here" << std::endl;
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ir::Program program(ctx);
  ir::Builder builder(ctx, program.block());
  ir::Block* block = program.block();

  // Def: A = paddle::dialect::UniformOp(std::vector<int64_t> shape,
  // phi::DataType dtype, float min, float max, int seed, phi::Place place)
  paddle::dialect::UniformOp uniform1 =
      builder.Build<paddle::dialect::UniformOp>(std::vector<int64_t>{2, 2},
                                                phi::DataType::FLOAT32,
                                                0.0,
                                                1.0,
                                                2,
                                                phi::CPUPlace());
  EXPECT_EQ(uniform1->result(0).type().isa<paddle::dialect::DenseTensorType>(),
            true);
  EXPECT_EQ(block->size(), 4u);

  // Def: B = paddle::dialect::UniformOp(...)
  paddle::dialect::UniformOp uniform2 =
      builder.Build<paddle::dialect::UniformOp>(std::vector<int64_t>{2, 2},
                                                phi::DataType::FLOAT32,
                                                0.0,
                                                1.0,
                                                2,
                                                phi::CPUPlace());
  EXPECT_EQ(uniform2->result(0).type().isa<paddle::dialect::DenseTensorType>(),
            true);
  EXPECT_EQ(block->size(), 8u);

  // Def: C = paddle::dialect::AddOp(ir::OpResult x_, ir::OpResult y_)
  paddle::dialect::AddOp add = builder.Build<paddle::dialect::AddOp>(
      uniform1->result(0), uniform2->result(0));
  EXPECT_EQ(add->result(0).type().isa<paddle::dialect::DenseTensorType>(),
            true);
  EXPECT_EQ(block->size(), 9u);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);
  std::cerr << "after lower" << std::endl;

  kernel_program->Print(std::cout);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;

  std::cerr << "before init" << std::endl;
  InterpreterCore test_core(
      place, prog_desc.Block(0), &scope, kernel_program.get());

  std::cerr << "after init" << std::endl;
  test_core.Run({});
  std::cerr << "after run " << std::endl;

  // auto tensor = scope.Var("inner_var_2")->Get<phi::DenseTensor>();

  std::cerr << "uot" << tensor << std::endl;
}

}  // namespace framework
}  // namespace paddle
