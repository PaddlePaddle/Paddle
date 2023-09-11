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

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/utils.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

#include "paddle/fluid/platform/init.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"

#include "paddle/fluid/pir/phi_kernel_adaptor/phi_kernel_adaptor.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/pir/core/attribute.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full_int_array, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(uniform, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

TEST(program_test, program) {
  // Prepare ir env
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder(ctx, program.block());
  pir::Block* block = program.block();

  // Def: A = paddle::dialect::UniformOp(std::vector<int64_t> shape,
  // phi::DataType dtype, float min, float max, int seed, phi::Place place)
  pir::AttributeMap uniform1_attributes;
  uniform1_attributes.insert({"shape",
                              paddle::dialect::IntArrayAttribute::get(
                                  pir::IrContext::Instance(),
                                  phi::IntArray(std::vector<int64_t>{2, 2}))});
  uniform1_attributes.insert(
      {"dtype",
       paddle::dialect::DataTypeAttribute::get(pir::IrContext::Instance(),
                                               phi::DataType::FLOAT32)});
  uniform1_attributes.insert(
      {"min", pir::FloatAttribute::get(pir::IrContext::Instance(), 0.0)});
  uniform1_attributes.insert(
      {"max", pir::FloatAttribute::get(pir::IrContext::Instance(), 1.0)});
  uniform1_attributes.insert(
      {"seed", pir::Int32Attribute::get(pir::IrContext::Instance(), 2)});
  uniform1_attributes.insert(
      {"place",
       paddle::dialect::PlaceAttribute::get(pir::IrContext::Instance(),
                                            phi::CPUPlace())});
  paddle::dialect::UniformOp uniform1 =
      builder.Build<paddle::dialect::UniformOp>(uniform1_attributes);

  EXPECT_EQ(uniform1->result(0).type().isa<paddle::dialect::DenseTensorType>(),
            true);
  EXPECT_EQ(block->size(), 4u);

  pir::Attribute seed_attr = uniform1.attribute("seed");
  pir::Int32Attribute seed_attr1 =
      uniform1.attribute<pir::Int32Attribute>("seed");
  EXPECT_EQ(seed_attr.dyn_cast<pir::Int32Attribute>().data(),
            seed_attr1.data());

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

  // Def: C = paddle::dialect::AddOp(pir::OpResult x_, pir::OpResult y_)
  paddle::dialect::AddOp add = builder.Build<paddle::dialect::AddOp>(
      uniform1->result(0), uniform2->result(0));
  EXPECT_EQ(add->result(0).type().isa<paddle::dialect::DenseTensorType>(),
            true);
  EXPECT_EQ(block->size(), 9u);

  // Execute program
  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);
  paddle::framework::Scope scope;
  PhiKernelAdaptor phi_kernel_adaptor(&scope);
  phi_kernel_adaptor.run_kernel_prog(kernel_program.get());

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

TEST(program_test, mutable_attribute) {
  // Prepare ir env
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  pir::Block* block = program.block();

  // Def FullOp
  paddle::dialect::FullIntArrayOp full_shape_op =
      builder.Build<paddle::dialect::FullIntArrayOp>(
          std::vector<int64_t>{2, 2}, phi::DataType::INT64, phi::CPUPlace());
  pir::OpResult shape_ = full_shape_op->result(0);
  // Generate scalar mutable attribute: min
  paddle::dialect::FullOp full_min_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 0.0, phi::DataType::FLOAT32, phi::CPUPlace());
  pir::OpResult min_ = full_min_op->result(0);
  // Generate scalar mutable attribute: max
  paddle::dialect::FullOp full_max_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());
  pir::OpResult max_ = full_max_op->result(0);

  // Def: static void Build(pir::Builder &builder, pir::OperationArgument
  // &argument, pir::OpResult shape_, pir::OpResult min_, pir::OpResult max_,
  // phi::DataType dtype, int seed, phi::Place place={});
  paddle::dialect::UniformOp uniform1 =
      builder.Build<paddle::dialect::UniformOp>(
          shape_, min_, max_, phi::DataType::FLOAT32, 2, phi::CPUPlace());
  EXPECT_EQ(uniform1->result(0).type().isa<paddle::dialect::DenseTensorType>(),
            true);
  EXPECT_EQ(block->size(), 4u);

  // Def: B = paddle::dialect::UniformOp(...)
  paddle::dialect::UniformOp uniform2 =
      builder.Build<paddle::dialect::UniformOp>(
          shape_, min_, max_, phi::DataType::FLOAT32, 2, phi::CPUPlace());
  EXPECT_EQ(uniform2->result(0).type().isa<paddle::dialect::DenseTensorType>(),
            true);
  EXPECT_EQ(block->size(), 5u);

  // Def: C = paddle::dialect::AddOp(pir::OpResult x_, pir::OpResult y_)
  paddle::dialect::AddOp add = builder.Build<paddle::dialect::AddOp>(
      uniform1->result(0), uniform2->result(0));
  EXPECT_EQ(add->result(0).type().isa<paddle::dialect::DenseTensorType>(),
            true);
  EXPECT_EQ(block->size(), 6u);

  // Execute program
  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(&program);
  paddle::framework::Scope scope;
  PhiKernelAdaptor phi_kernel_adaptor(&scope);
  phi_kernel_adaptor.run_kernel_prog(kernel_program.get());

  auto out_tensor =
      scope.Var(phi_kernel_adaptor.out_name)->Get<phi::DenseTensor>();

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
