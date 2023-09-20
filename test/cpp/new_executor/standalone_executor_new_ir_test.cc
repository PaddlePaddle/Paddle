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

#include "paddle/fluid/framework/new_executor/new_ir_interpreter.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"

#include "paddle/fluid/platform/init_phi.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"

DECLARE_FILE_SYMBOLS(kernel_dialect);

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full_int_array, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(uniform, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sqrt, CPU, ALL_LAYOUT);

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

namespace paddle {
namespace framework {

TEST(StandaloneExecutor, run) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program((ctx));

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Builder builder = pir::Builder(ctx, program.block());

  paddle::dialect::FullOp op1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::FullOp op2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  builder.Build<paddle::dialect::AddOp>(op1->result(0), op2->result(0));

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  ProgramDesc prog_desc;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);

  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string out_name = os.str() + "_inner_var_2";
  test_core.SetSkipGcVars({out_name});

  test_core.Run({});

  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(out_name)->Get<phi::DenseTensor>()
          : test_core.local_scope()->FindVar(out_name)->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 2.0);
  bool res1 = simple_cmp(out_tensor.data<float>()[1], 2.0);
  bool res2 = simple_cmp(out_tensor.data<float>()[2], 2.0);
  bool res3 = simple_cmp(out_tensor.data<float>()[3], 2.0);

  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
  EXPECT_EQ(res2, true);
  EXPECT_EQ(res3, true);
}

TEST(StandaloneExecutor, run_inplace_sqrt) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program((ctx));
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  paddle::dialect::FullOp full = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 4.0, phi::DataType::FLOAT32, phi::CPUPlace());

  builder.Build<paddle::dialect::Sqrt_Op>(full->result(0));

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);

  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string out_name = os.str() + "_inner_var_0";
  test_core.SetSkipGcVars({out_name});

  test_core.Run({});

  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(out_name)->Get<phi::DenseTensor>()
          : test_core.local_scope()->FindVar(out_name)->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 2.0);
  bool res1 = simple_cmp(out_tensor.data<float>()[1], 2.0);
  bool res2 = simple_cmp(out_tensor.data<float>()[2], 2.0);
  bool res3 = simple_cmp(out_tensor.data<float>()[3], 2.0);

  EXPECT_EQ(scope.kids().size(), 1u);
  EXPECT_EQ(scope.kids().front()->Size(), 1u);
  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
  EXPECT_EQ(res2, true);
  EXPECT_EQ(res3, true);
}

TEST(StandaloneExecutor, if_op) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  auto full_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, true, phi::DataType::BOOL);

  auto if_op = builder.Build<paddle::dialect::IfOp>(
      full_op.out(), std::vector<pir::Type>{full_op.result(0).type()});

  pir::Block* true_block = if_op.true_block();

  builder.SetInsertionPointToStart(true_block);

  auto full_op_1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2}, true, phi::DataType::BOOL);
  builder.Build<pir::YieldOp>(std::vector<pir::OpResult>{full_op_1.out()});

  pir::Block* false_block = if_op.false_block();

  builder.SetInsertionPointToStart(false_block);

  auto full_op_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{3}, true, phi::DataType::BOOL);
  builder.Build<pir::YieldOp>(std::vector<pir::OpResult>{full_op_2.out()});

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);

  std::stringstream os;
  os << reinterpret_cast<NewIRInterpreter*>(
      const_cast<InterpreterBaseImpl*>(test_core.Impl()));
  std::string out_name = os.str() + "_inner_var_1";
  test_core.SetSkipGcVars({out_name});

  test_core.Run({});

  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(out_name)->Get<phi::DenseTensor>()
          : test_core.local_scope()->FindVar(out_name)->Get<phi::DenseTensor>();

  bool res0 = out_tensor.data<bool>()[0] == true;
  bool res1 = out_tensor.data<bool>()[1] == true;

  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
}

}  // namespace framework
}  // namespace paddle
