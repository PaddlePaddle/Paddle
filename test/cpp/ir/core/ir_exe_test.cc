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

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
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

#include "paddle/fluid/ir/phi_kernel_adaptor/phi_kernel_adaptor.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full_int_array, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(uniform, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

TEST(program_test, program) {
  // Prepare ir env
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ir::Program program(ctx);
  ir::Builder builder = ir::Builder::AtBlockEnd(ctx, program.block());
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
  EXPECT_EQ(uniform1->GetResultByIndex(0)
                .type()
                .isa<paddle::dialect::DenseTensorType>(),
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
  EXPECT_EQ(uniform2->GetResultByIndex(0)
                .type()
                .isa<paddle::dialect::DenseTensorType>(),
            true);
  EXPECT_EQ(block->size(), 8u);

  // Def: C = paddle::dialect::AddOp(ir::OpResult x_, ir::OpResult y_)
  paddle::dialect::AddOp add = builder.Build<paddle::dialect::AddOp>(
      uniform1->GetResultByIndex(0), uniform2->GetResultByIndex(0));
  EXPECT_EQ(
      add->GetResultByIndex(0).type().isa<paddle::dialect::DenseTensorType>(),
      true);
  EXPECT_EQ(block->size(), 9u);

  // Execute program
  std::cerr << "11" << std::endl;
  paddle::framework::Scope scope;
  PhiKernelAdaptor phi_kernel_adaptor(&scope);
  phi_kernel_adaptor.run(&program);

  auto out_tensor =
      scope.Var(phi_kernel_adaptor.out_name)->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 1.80721);
  bool res1 = simple_cmp(out_tensor.data<float>()[1], 1.70047);
  bool res2 = simple_cmp(out_tensor.data<float>()[2], 1.56764);
  bool res3 = simple_cmp(out_tensor.data<float>()[3], 1.85063);

  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
  EXPECT_EQ(res2, true);
  EXPECT_EQ(res3, true);
}
