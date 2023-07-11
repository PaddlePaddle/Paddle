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
#include "paddle/fluid/ir/transforms/pd_op_to_kernel_pass.h"
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
PD_DECLARE_KERNEL(sqrt, CPU, ALL_LAYOUT);

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

namespace paddle {
namespace framework {

TEST(StandaloneExecutor, run) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ir::Program program((ctx));

  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

  ir::Builder builder = ir::Builder(ctx, program.block());

  paddle::dialect::FullOp op1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::FullOp op2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  builder.Build<paddle::dialect::AddOp>(op1->result(0), op2->result(0));

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  kernel_program->Print(std::cout);

  auto place = platform::CPUPlace();
  Scope scope;

  InterpreterCore test_core(place, std::move(kernel_program), &scope);

  test_core.Run({});

  auto out_tensor = test_core.local_scope() == nullptr
                        ? scope.FindVar("inner_var_2")->Get<phi::DenseTensor>()
                        : test_core.local_scope()
                              ->FindVar("inner_var_2")
                              ->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 2.0);
  bool res1 = simple_cmp(out_tensor.data<float>()[1], 2.0);
  bool res2 = simple_cmp(out_tensor.data<float>()[2], 2.0);
  bool res3 = simple_cmp(out_tensor.data<float>()[3], 2.0);

  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
  EXPECT_EQ(res2, true);
  EXPECT_EQ(res3, true);
}

TEST(StandaloneExecutor, run_2) {
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

  paddle::dialect::ScaleOp scale =
      builder.Build<paddle::dialect::ScaleOp>(add->result(0), 1.0, 0.0, true);

  EXPECT_EQ(scale->result(0).type().isa<paddle::dialect::DenseTensorType>(),
            true);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  auto place = platform::CPUPlace();
  Scope scope;

  InterpreterCore test_core(place, std::move(kernel_program), &scope);

  test_core.Run({});

  auto out_tensor = test_core.local_scope() == nullptr
                        ? scope.FindVar("inner_var_10")->Get<phi::DenseTensor>()
                        : test_core.local_scope()
                              ->FindVar("inner_var_10")
                              ->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 1.80721);
  bool res1 = simple_cmp(out_tensor.data<float>()[1], 1.70047);
  bool res2 = simple_cmp(out_tensor.data<float>()[2], 1.56764);
  bool res3 = simple_cmp(out_tensor.data<float>()[3], 1.85063);
  std::cerr << out_tensor.data<float>()[0] << "\t"
            << out_tensor.data<float>()[1] << "\t"
            << out_tensor.data<float>()[2] << "\t"
            << out_tensor.data<float>()[3] << std::endl;
  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
  EXPECT_EQ(res2, true);
  EXPECT_EQ(res3, true);
}

#ifdef PADDLE_WITH_CUDA
TEST(StandaloneExecutor, data_transfer) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ir::Program program(ctx);
  ir::Builder builder(ctx, program.block());
  ir::Block* block = program.block();

  // Def: A = paddle::dialect::UniformOp(std::vector<int64_t> shape,
  // phi::DataType dtype, float min, float max, int seed, phi::Place place)
  paddle::dialect::UniformOp uniform1 =
      builder.Build<paddle::dialect::UniformOp>(std::vector<int64_t>{1},
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
      builder.Build<paddle::dialect::UniformOp>(std::vector<int64_t>{100, 100},
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

  program.Print(std::cout);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  kernel_program->Print(std::cout);

  auto place = platform::CPUPlace();
  Scope scope;

  InterpreterCore test_core(place, std::move(kernel_program), &scope);

  test_core.Run({});

  auto out_tensor = test_core.local_scope() == nullptr
                        ? scope.FindVar("inner_var_9")->Get<phi::DenseTensor>()
                        : test_core.local_scope()
                              ->FindVar("inner_var_9")
                              ->Get<phi::DenseTensor>();

  auto& pool = phi::DeviceContextPool::Instance();
  phi::DenseTensor out;
  phi::DeviceContext* dev_ctx = pool.Get(out_tensor.place());
  phi::Copy(*dev_ctx, out_tensor, place, true, &out);

  bool res0 = simple_cmp(out.data<float>()[0], 0.903649);
  bool res1 = simple_cmp(out.data<float>()[1], 1.07367);
  bool res2 = simple_cmp(out.data<float>()[2], 1.10631);
  bool res3 = simple_cmp(out.data<float>()[3], 1.68683);
  std::cerr << out.data<float>()[0] << "\t" << out.data<float>()[1] << "\t"
            << out.data<float>()[2] << "\t" << out.data<float>()[3]
            << std::endl;
  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
  EXPECT_EQ(res2, true);
  EXPECT_EQ(res3, true);
}
#endif

TEST(StandaloneExecutor, run_inplace_sqrt) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ir::Program program((ctx));
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ir::Builder builder = ir::Builder(ctx, program.block());

  paddle::dialect::FullOp full = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 4.0, phi::DataType::FLOAT32, phi::CPUPlace());

  builder.Build<paddle::dialect::Sqrt_Op>(full->result(0));

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);

  kernel_program->Print(std::cout);

  auto place = platform::CPUPlace();
  Scope scope;
  InterpreterCore test_core(place, std::move(kernel_program), &scope);
  test_core.Run({});

  auto out_tensor = test_core.local_scope() == nullptr
                        ? scope.FindVar("inner_var_0")->Get<phi::DenseTensor>()
                        : test_core.local_scope()
                              ->FindVar("inner_var_0")
                              ->Get<phi::DenseTensor>();

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

}  // namespace framework
}  // namespace paddle
