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

#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_op.h"

#include "test/cpp/pir/tools/test_pir_utils.h"

TEST(op_test, region_test) {
  // (1) Register Dialect, Operation1, Operation2 into IrContext.
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Dialect *test_dialect = ctx->GetOrRegisterDialect<test::TestDialect>();
  EXPECT_EQ(test_dialect != nullptr, true);

  // (2) Get registered operations.
  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(test::Operation1::name());
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(test::Operation2::name());

  pir::Operation *op1 = pir::Operation::Create(
      {},
      test::CreateAttributeMap({"op1_attr1", "op1_attr2"},
                               {"op1_attr1", "op1_attr2"}),
      {pir::Float32Type::get(ctx)},
      op1_info);
  pir::Operation *op_2 =
      pir::Operation::Create({}, {}, {pir::Float32Type::get(ctx)}, op2_info);

  pir::OperationArgument argument(op2_info);
  argument.output_types = {pir::Float32Type::get(ctx)};
  argument.AddRegion(nullptr);

  pir::Operation *op3 = pir::Operation::Create(std::move(argument));

  pir::Region &region = op3->region(0);
  EXPECT_EQ(region.empty(), true);

  // (3) Test custom operation printer
  std::stringstream ss;
  op1->Print(ss);
  EXPECT_EQ(ss.str(), "(%0) = \"test.operation1\" ()");

  region.push_back(new pir::Block());
  region.push_front(new pir::Block());
  region.insert(region.begin(), new pir::Block());
  auto &block = region.front();
  block.push_front(op1);
  block.insert(block.begin(), op_2);
  op3->Destroy();
}

TEST(op_test, module_op_death) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(pir::ModuleOp::name());

  std::vector<pir::Value> inputs{pir::Value()};
  pir::AttributeMap attrs{{"program", pir::Int32Attribute::get(ctx, 1)}};
  std::vector<pir::Type> output_types = {pir::Float32Type::get(ctx)};

  EXPECT_THROW(pir::Operation::Create(inputs, {}, {}, op_info),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(pir::Operation::Create({}, attrs, {}, op_info),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(pir::Operation::Create({}, {}, output_types, op_info),
               common::enforce::EnforceNotMet);

  pir::Program program(ctx);

  EXPECT_EQ(program.module_op().program(), &program);
  EXPECT_EQ(program.module_op().ir_context(), ctx);

  program.module_op()->set_attribute("program",
                                     pir::PointerAttribute::get(ctx, &program));
}

TEST(op_test, trait_and_interface) {
  pir::IrContext ctx;
  ctx.GetOrRegisterDialect<test::TestDialect>();
  pir::Program program(&ctx);
  auto block = program.block();
  pir::Builder builder(&ctx, block);
  auto op1 = builder.Build<test::Operation1>();
  auto op2 = builder.Build<test::Operation2>();

  EXPECT_EQ(op1->HasTrait<test::ReadOnlyTrait>(), false);
  EXPECT_EQ(op1->HasInterface<test::InferShapeInterface>(), false);
  EXPECT_EQ(op2->HasTrait<test::ReadOnlyTrait>(), true);
  EXPECT_EQ(op2->HasInterface<test::InferShapeInterface>(), true);

  pir::OperationArgument argument(&ctx, "test.region");
  EXPECT_THROW(builder.Build(std::move(argument)),
               common::enforce::EnforceNotMet);
}

TEST(op_test, op_traits_test) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;

  pir::DenseTensorType dense_tensor_dtype =
      pir::DenseTensorType::get(ctx, dtype, dims, data_layout, lod, offset);

  pir::Operation *op1 =
      test::CreateDenseTensorOp(ctx, dims, {"op1_temp"}, {"op1_attr"}, dtype);
  pir::Operation *op2 =
      test::CreateDenseTensorOp(ctx, dims, {"op2_temp"}, {"op2_attr"}, dtype);

  block->push_back(op1);
  block->push_back(op2);
  auto op3 = builder.Build<test::TraitExampleOp>(
      op1->result(0), op2->result(0), dense_tensor_dtype);

  EXPECT_EQ(op3->HasTrait<pir::SameOperandsShapeTrait>(), true);
  EXPECT_EQ(op3->HasTrait<pir::SameOperandsAndResultShapeTrait>(), true);
  EXPECT_EQ(op3->HasTrait<pir::SameOperandsElementTypeTrait>(), true);
  EXPECT_EQ(op3->HasTrait<pir::SameOperandsAndResultElementTypeTrait>(), true);
  EXPECT_EQ(op3->HasTrait<pir::SameOperandsAndResultTypeTrait>(), true);
  EXPECT_EQ(op3->HasTrait<pir::SameTypeOperandsTrait>(), true);
}

TEST(op_test, same_operands_shape_trait_test1) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  EXPECT_THROW(builder.Build<test::SameOperandsShapeTraitOp1>(),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_shape_trait_test2) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype1 = pir::Float32Type::get(ctx);
  phi::DDim dims1 = {2, 2};

  pir::Type dtype2 = pir::Float64Type::get(ctx);
  phi::DDim dims2 = {2, 2, 2};

  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;

  pir::DenseTensorType dense_tensor_dtype =
      pir::DenseTensorType::get(ctx, dtype1, dims1, data_layout, lod, offset);

  pir::Operation *op1 =
      test::CreateDenseTensorOp(ctx, dims1, {"op1_temp"}, {"op1_attr"}, dtype1);
  pir::Operation *op2 =
      test::CreateDenseTensorOp(ctx, dims2, {"op2_temp"}, {"op2_attr"}, dtype2);

  block->push_back(op1);
  block->push_back(op2);

  EXPECT_THROW(builder.Build<test::SameOperandsShapeTraitOp2>(
                   op1->result(0), op2->result(0), dense_tensor_dtype),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_and_result_shape_trait_test1) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  EXPECT_THROW(builder.Build<test::SameOperandsAndResultShapeTraitOp1>(),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_and_result_shape_trait_test2) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype = pir::Float64Type::get(ctx);
  phi::DDim dims = {2, 2, 2};

  pir::Operation *op1 =
      test::CreateDenseTensorOp(ctx, dims, {"op1_temp"}, {"op1_attr"}, dtype);
  pir::Operation *op2 =
      test::CreateDenseTensorOp(ctx, dims, {"op2_temp"}, {"op2_attr"}, dtype);

  block->push_back(op1);
  block->push_back(op2);

  EXPECT_THROW(builder.Build<test::SameOperandsAndResultShapeTraitOp2>(
                   op1->result(0), op2->result(0)),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_and_result_shape_trait_test3) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype1 = pir::Float32Type::get(ctx);
  phi::DDim dims1 = {2, 2};

  pir::Type dtype2 = pir::Float64Type::get(ctx);
  phi::DDim dims2 = {2, 2, 2};

  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;

  pir::DenseTensorType dense_tensor_dtype =
      pir::DenseTensorType::get(ctx, dtype1, dims1, data_layout, lod, offset);

  pir::Operation *op1 =
      test::CreateDenseTensorOp(ctx, dims1, {"op1_temp"}, {"op1_attr"}, dtype1);
  pir::Operation *op2 =
      test::CreateDenseTensorOp(ctx, dims2, {"op2_temp"}, {"op2_attr"}, dtype2);

  block->push_back(op1);
  block->push_back(op2);
  EXPECT_THROW(builder.Build<test::SameOperandsAndResultShapeTraitOp3>(
                   op1->result(0), op2->result(0), dense_tensor_dtype),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_element_type_trait_test1) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  EXPECT_THROW(builder.Build<test::SameOperandsElementTypeTraitOp1>(),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_element_type_trait_test2) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype1 = pir::Float32Type::get(ctx);
  pir::Type dtype2 = pir::Float64Type::get(ctx);

  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;

  pir::DenseTensorType dense_tensor_dtype =
      pir::DenseTensorType::get(ctx, dtype1, dims, data_layout, lod, offset);

  pir::Operation *op1 =
      test::CreateDenseTensorOp(ctx, dims, {"op1_temp"}, {"op1_attr"}, dtype1);
  pir::Operation *op2 =
      test::CreateDenseTensorOp(ctx, dims, {"op2_temp"}, {"op2_attr"}, dtype2);

  block->push_back(op1);
  block->push_back(op2);
  EXPECT_THROW(builder.Build<test::SameOperandsElementTypeTraitOp2>(
                   op1->result(0), op2->result(0), dense_tensor_dtype),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_and_result_element_type_trait_test1) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  EXPECT_THROW(builder.Build<test::SameOperandsAndResultElementTypeTraitOp1>(),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_and_result_element_type_trait_test2) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};

  pir::Operation *op1 =
      test::CreateDenseTensorOp(ctx, dims, {"op1_temp"}, {"op1_attr"}, dtype);
  pir::Operation *op2 =
      test::CreateDenseTensorOp(ctx, dims, {"op2_temp"}, {"op2_attr"}, dtype);

  block->push_back(op1);
  block->push_back(op2);
  EXPECT_THROW(builder.Build<test::SameOperandsAndResultElementTypeTraitOp2>(
                   op1->result(0), op2->result(0)),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_and_result_element_type_trait_test3) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype1 = pir::Float32Type::get(ctx);
  phi::DDim dims1 = {2, 2};

  pir::Type dtype2 = pir::Float64Type::get(ctx);
  phi::DDim dims2 = {2, 2, 2};

  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;

  pir::DenseTensorType dense_tensor_dtype1 =
      pir::DenseTensorType::get(ctx, dtype1, dims1, data_layout, lod, offset);
  pir::DenseTensorType dense_tensor_dtype2 =
      pir::DenseTensorType::get(ctx, dtype2, dims2, data_layout, lod, offset);

  pir::Operation *op1 =
      test::CreateDenseTensorOp(ctx, dims1, {"op1_temp"}, {"op1_attr"}, dtype1);
  pir::Operation *op2 =
      test::CreateDenseTensorOp(ctx, dims2, {"op2_temp"}, {"op2_attr"}, dtype2);

  block->push_back(op1);
  block->push_back(op2);
  EXPECT_THROW(builder.Build<test::SameOperandsAndResultElementTypeTraitOp3>(
                   op1->result(0),
                   op2->result(0),
                   dense_tensor_dtype1,
                   dense_tensor_dtype1),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(builder.Build<test::SameOperandsAndResultElementTypeTraitOp3>(
                   op1->result(0),
                   op1->result(0),
                   dense_tensor_dtype1,
                   dense_tensor_dtype2),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_and_result_type_trait_test1) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  EXPECT_THROW(builder.Build<test::SameOperandsAndResultTypeTraitOp1>(),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_and_result_type_trait_test2) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};

  pir::Operation *op1 =
      test::CreateDenseTensorOp(ctx, dims, {"op1_temp"}, {"op1_attr"}, dtype);
  pir::Operation *op2 =
      test::CreateDenseTensorOp(ctx, dims, {"op2_temp"}, {"op2_attr"}, dtype);

  block->push_back(op1);
  block->push_back(op2);
  EXPECT_THROW(builder.Build<test::SameOperandsAndResultTypeTraitOp2>(
                   op1->result(0), op2->result(0)),
               common::enforce::EnforceNotMet);
}

TEST(op_test, same_operands_and_result_type_trait_test3) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<test::TestDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  auto block = program.block();
  pir::Builder builder(ctx, block);

  pir::Type dtype1 = pir::Float32Type::get(ctx);
  phi::DDim dims1 = {2, 2};

  pir::Type dtype2 = pir::Float64Type::get(ctx);
  phi::DDim dims2 = {2, 2, 2};

  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;

  pir::DenseTensorType dense_tensor_dtype1 =
      pir::DenseTensorType::get(ctx, dtype1, dims1, data_layout, lod, offset);

  pir::DenseTensorType dense_tensor_dtype2 =
      pir::DenseTensorType::get(ctx, dtype2, dims2, data_layout, lod, offset);

  pir::DenseTensorType dense_tensor_dtype3 =
      pir::DenseTensorType::get(ctx, dtype1, dims2, data_layout, lod, offset);

  pir::Operation *op1 =
      test::CreateDenseTensorOp(ctx, dims1, {"op1_temp"}, {"op1_attr"}, dtype2);
  pir::Operation *op2 =
      test::CreateDenseTensorOp(ctx, dims2, {"op2_temp"}, {"op2_attr"}, dtype1);

  block->push_back(op1);
  block->push_back(op2);
  EXPECT_THROW(builder.Build<test::SameOperandsAndResultTypeTraitOp3>(
                   op1->result(0),
                   op2->result(0),
                   dense_tensor_dtype1,
                   dense_tensor_dtype2),
               common::enforce::EnforceNotMet);

  EXPECT_THROW(builder.Build<test::SameOperandsAndResultTypeTraitOp3>(
                   op1->result(0),
                   op2->result(0),
                   dense_tensor_dtype1,
                   dense_tensor_dtype3),
               common::enforce::EnforceNotMet);

  EXPECT_THROW(builder.Build<test::SameOperandsAndResultTypeTraitOp3>(
                   op1->result(0),
                   op2->result(0),
                   dense_tensor_dtype1,
                   dense_tensor_dtype1),
               common::enforce::EnforceNotMet);

  EXPECT_THROW(builder.Build<test::SameOperandsAndResultTypeTraitOp3>(
                   op2->result(0),
                   op1->result(0),
                   dense_tensor_dtype1,
                   dense_tensor_dtype1),
               common::enforce::EnforceNotMet);
}

TEST(printer_test, custom_hooks) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Dialect *test_dialect = ctx->GetOrRegisterDialect<test::TestDialect>();
  EXPECT_EQ(test_dialect != nullptr, true);

  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(test::Operation1::name());
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(test::Operation2::name());

  pir::Operation *op1 = pir::Operation::Create(
      {},
      test::CreateAttributeMap({"op1_attr1", "op1_attr2"},
                               {"op1_attr1", "op1_attr2"}),
      {pir::Float32Type::get(ctx)},
      op1_info);
  pir::Operation *op2 = pir::Operation::Create(
      {op1->result(0)}, {}, {pir::Float32Type::get(ctx)}, op2_info);

  pir::Program program(ctx);
  program.block()->push_back(op1);
  program.block()->push_back(op2);

  pir::PrintHooks hooks;
  // this one retains old printing and adds new info
  hooks.value_print_hook = [](pir::Value v, pir::IrPrinter &printer) {
    printer.IrPrinter::PrintValue(v);
    printer.os << " [extra info]";
  };
  // this one overrides old printing
  hooks.op_print_hook = [](const pir::Operation &op, pir::IrPrinter &printer) {
    printer.PrintOpResult(op);
    printer.os << " :=";

    printer.os << " \"" << op.name() << "\"";
    printer.PrintOpOperands(op);
    printer.PrintAttributeMap(op);
    printer.os << " :";
    printer.PrintOpReturnType(op);
  };

  hooks.attribute_print_hook = [](pir::Attribute attr,
                                  pir::IrPrinter &printer) {
    printer.os << "[PlaceHolder]";
  };
  hooks.type_print_hook = [](pir::Type type, pir::IrPrinter &printer) {
    printer.os << "[" << type << "]";
  };

  std::stringstream ss;

  ss << pir::CustomPrintHelper{program, hooks};
  EXPECT_EQ(
      ss.str(),
      "{\n"
      "(%0 [extra info]) := \"test.operation1\" () "
      "{op1_attr1:[PlaceHolder],op1_attr2:[PlaceHolder]} :[f32]\n"
      "(%1 [extra info]) := \"test.operation2\" (%0 [extra info]) {} :[f32]\n"
      "}\n");
}
