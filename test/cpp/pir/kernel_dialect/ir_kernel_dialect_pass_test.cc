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

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/phi_kernel_adaptor/phi_kernel_adaptor.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/utils.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full_int_array, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(uniform, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

TEST(program_test, program) {
  // (1) Init environment.
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

  EXPECT_EQ(kernel_program->block()->size(), 3u);
  EXPECT_EQ(kernel_program->block()
                ->front()
                ->dyn_cast<paddle::dialect::PhiKernelOp>()
                .op_name(),
            "pd_op.full");
  EXPECT_EQ(kernel_program->block()
                ->front()
                ->dyn_cast<paddle::dialect::PhiKernelOp>()
                .kernel_name(),
            "full");
  EXPECT_EQ(kernel_program->block()
                ->front()
                ->dyn_cast<paddle::dialect::PhiKernelOp>()
                .kernel_key()
                .dtype(),
            phi::DataType::FLOAT32);
}

TEST(dialect_attr, attr) {
  // (1) Init environment.
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program((ctx));

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  auto kernel_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

  phi::KernelKey kernel_key(
      phi::Backend::CPU, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT32);
  auto attr = paddle::dialect::KernelAttribute::get(ctx, kernel_key);

  std::stringstream ss;

  kernel_dialect->PrintAttribute(attr, ss);

  EXPECT_EQ(
      ss.str() == "<backend:CPU|layout:Undefined(AnyLayout)|dtype:float32>",
      true);
}

pir::AttributeMap CreateAttributeMap(std::vector<std::string> attribute_names,
                                     std::vector<std::string> attributes,
                                     std::string attr_name,
                                     phi::KernelKey kernel_key) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::AttributeMap attr_map;
  for (size_t i = 0; i < attribute_names.size(); i++) {
    pir::Attribute attr_value = pir::StrAttribute::get(ctx, attributes[i]);
    attr_map.insert(
        std::pair<std::string, pir::Attribute>(attribute_names[i], attr_value));
  }
  auto attr = paddle::dialect::KernelAttribute::get(ctx, kernel_key);
  attr_map.insert(std::pair<std::string, pir::Attribute>(attr_name, attr));
  return attr_map;
}

TEST(kernel_dialect, legacy_op_test) {
  // (1) Init environment.

  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program((ctx));

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  phi::KernelKey kernel_key(
      phi::Backend::CPU, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT32);

  pir::OpInfo kernel_op_info =
      ctx->GetRegisteredOpInfo(paddle::dialect::LegacyKernelOp::name());
  pir::OperationArgument argument(kernel_op_info);
  argument.attributes = CreateAttributeMap({"op_name", "kernel_name"},
                                           {"pd_op.kernel_op", "kernel_op"},
                                           "kernel_key",
                                           kernel_key);

  pir::Operation* op = pir::Operation::Create(std::move(argument));
  EXPECT_EQ("pd_op.kernel_op",
            op->dyn_cast<paddle::dialect::LegacyKernelOp>().op_name());
  EXPECT_EQ("kernel_op",
            op->dyn_cast<paddle::dialect::LegacyKernelOp>().kernel_name());
  EXPECT_EQ(kernel_key,
            op->dyn_cast<paddle::dialect::LegacyKernelOp>().kernel_key());
}

TEST(kernel_dialect, cond_op_test) {
  // (1) Init environment.
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  auto full_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, true, phi::DataType::BOOL);

  auto if_op = builder.Build<paddle::dialect::IfOp>(
      full_op.out(), std::vector<pir::Type>{builder.bool_type()});

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
}
