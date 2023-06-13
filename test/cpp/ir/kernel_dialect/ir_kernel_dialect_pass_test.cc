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

#include <gtest/gtest.h>
#include <sstream>

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/fluid/ir/pass/pd_op_to_kernel_pass.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

#include "paddle/fluid/platform/init.h"

#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "test/cpp/ir/core/phi_kernel_adaptor.h"

#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/ir/dialect/kernel_dialect.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full_int_array, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(uniform, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

TEST(program_test, program) {
  // (1) Init environment.
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

  paddle::framework::Scope scope;
  PhiKernelAdaptor phi_kernel_adaptor(&scope);
  phi_kernel_adaptor.run_kernel_prog(kernel_program.get());

  auto out_tensor =
      scope.Var(phi_kernel_adaptor.out_name)->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 2.0);
  bool res1 = simple_cmp(out_tensor.data<float>()[1], 2.0);
  bool res2 = simple_cmp(out_tensor.data<float>()[2], 2.0);
  bool res3 = simple_cmp(out_tensor.data<float>()[3], 2.0);

  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
  EXPECT_EQ(res2, true);
  EXPECT_EQ(res3, true);
}

TEST(dialect_attr, attr) {
  // (1) Init environment.
  ir::IrContext* ctx = ir::IrContext::Instance();
  ir::Program program((ctx));

  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  auto kernel_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::PaddleKernelDialect>();

  phi::KernelKey kernel_key(
      phi::Backend::CPU, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT32);
  auto attr = paddle::dialect::KernelAttribute::get(ctx, kernel_key);

  std::stringstream ss;

  kernel_dialect->PrintAttribute(attr, ss);

  EXPECT_EQ(
      ss.str() == "<backend:CPU|layout:Undefined(AnyLayout)|dtype:float32>",
      true);
}
