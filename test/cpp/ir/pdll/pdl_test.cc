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

#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/value.h"
#include "paddle/ir/pdll/pdl_dialect/pdl_dialect.h"
#include "paddle/ir/pdll/pdl_dialect/pdl_ops.h"
#include "paddle/ir/pdll/pdl_dialect/pdl_types.h"

#define INIT()                                      \
  ir::IrContext* ctx = ir::IrContext::Instance();   \
  ctx->GetOrRegisterDialect<ir::pdl::PDLDialect>(); \
  ir::Program program(ctx);                         \
  ir::Builder program_builder = ir::Builder(ctx, program.block())

TEST(pdl_dialect, pattern_op) {
  INIT();

  // case 1.
  program_builder.Build<ir::pdl::PDL_PatternOp>(2);

  // case 2.
  program_builder.Build<ir::pdl::PDL_PatternOp>(2, "conv_bn_pattern");

  std::stringstream os;
  program.Print(os);
  LOG(INFO) << os.str();
}

TEST(pdl_dialect, type_op) {
  INIT();

  auto pattern_op = program_builder.Build<ir::pdl::PDL_PatternOp>(2);
  auto* block = pattern_op.block();
  ir::Builder builder = ir::Builder(ctx, block);

  // case 1.
  builder.Build<ir::pdl::PDL_TypeOp>();

  // case 2.
  auto attr_type = ir::TypeAttribute::get(ctx, ir::Int32Type::get(ctx));
  auto type_type = ir::pdl::TypeType::get(ctx);
  builder.Build<ir::pdl::PDL_TypeOp>(attr_type, type_type);

  std::stringstream os;
  program.Print(os);
  LOG(INFO) << os.str();
}

TEST(pdl_dialect, operand_op) {
  INIT();
  auto pattern_op = program_builder.Build<ir::pdl::PDL_PatternOp>(2);
  auto* block = pattern_op.block();
  ir::Builder builder = ir::Builder(ctx, block);

  // case 1.
  builder.Build<ir::pdl::PDL_OperandOp>();

  // case 2.
  auto attr_type = ir::TypeAttribute::get(ctx, ir::Int32Type::get(ctx));
  auto type_type = ir::pdl::TypeType::get(ctx);
  auto type_op = builder.Build<ir::pdl::PDL_TypeOp>(attr_type, type_type);
  builder.Build<ir::pdl::PDL_OperandOp>(type_op->result(0),
                                        ir::pdl::ValueType::get(ctx));

  std::stringstream os;
  program.Print(os);
  LOG(INFO) << os.str();
}

TEST(pdl_dialect, attribute_op) {
  INIT();
  auto pattern_op = program_builder.Build<ir::pdl::PDL_PatternOp>(2);
  auto* block = pattern_op.block();
  ir::Builder builder = ir::Builder(ctx, block);

  // case 1.
  builder.Build<ir::pdl::PDL_AttributeOp>();

  // case 2.
  auto attr_type = ir::TypeAttribute::get(ctx, ir::Int32Type::get(ctx));
  auto type_type = ir::pdl::TypeType::get(ctx);
  auto type_op = builder.Build<ir::pdl::PDL_TypeOp>(attr_type, type_type);
  builder.Build<ir::pdl::PDL_AttributeOp>(type_op->result(0));

  // case 3.
  ir::Attribute attr = ir::StrAttribute::get(ctx, "hello");
  builder.Build<ir::pdl::PDL_AttributeOp>(attr);

  std::stringstream os;
  program.Print(os);
  LOG(INFO) << os.str();
}

TEST(pdl_dialect, operation_op) {
  INIT();
  auto pattern_op = program_builder.Build<ir::pdl::PDL_PatternOp>(2);
  auto* block = pattern_op.block();
  ir::Builder builder = ir::Builder(ctx, block);

  auto x = builder.Build<ir::pdl::PDL_OperandOp>();
  auto y = builder.Build<ir::pdl::PDL_OperandOp>();

  auto attr =
      builder.Build<ir::pdl::PDL_AttributeOp>(ir::Int32Attribute::get(ctx, 1));

  auto out_type = builder.Build<ir::pdl::PDL_TypeOp>();

  builder.Build<ir::pdl::PDL_OperationOp>(
      "pd.add",
      std::vector<std::string>{"axis"},
      std::vector<ir::OpResult>{x->result(0), y->result(0)} /*operands*/,
      std::vector<ir::OpResult>{attr->result(0)} /*attrs*/,
      std::vector<ir::OpResult>{out_type->result(0)} /*outputs*/,
      ir::pdl::OperationType::get(ctx));

  std::stringstream os;
  program.Print(os);
  LOG(INFO) << os.str();
}

TEST(pdl_dialect, erase_op) {
  INIT();
  auto pattern_op = program_builder.Build<ir::pdl::PDL_PatternOp>(2);
  auto* block = pattern_op.block();
  ir::Builder builder = ir::Builder(ctx, block);

  auto x = builder.Build<ir::pdl::PDL_OperandOp>();
  auto y = builder.Build<ir::pdl::PDL_OperandOp>();

  auto attr =
      builder.Build<ir::pdl::PDL_AttributeOp>(ir::Int32Attribute::get(ctx, 1));

  auto out_type = builder.Build<ir::pdl::PDL_TypeOp>();

  auto add_op = builder.Build<ir::pdl::PDL_OperationOp>(
      "pd.add",
      std::vector<std::string>{"axis"},
      std::vector<ir::OpResult>{x->result(0), y->result(0)} /*operands*/,
      std::vector<ir::OpResult>{attr->result(0)} /*attrs*/,
      std::vector<ir::OpResult>{out_type->result(0)} /*outputs*/,
      ir::pdl::OperationType::get(ctx));

  builder.Build<ir::pdl::PDL_EraseOp>(add_op->result(0));

  std::stringstream os;
  program.Print(os);
  LOG(INFO) << os.str();
}

TEST(pdl_dialect, result_op) {
  INIT();
  auto pattern_op = program_builder.Build<ir::pdl::PDL_PatternOp>(2);
  auto* block = pattern_op.block();
  ir::Builder builder = ir::Builder(ctx, block);

  auto x = builder.Build<ir::pdl::PDL_OperandOp>();
  auto y = builder.Build<ir::pdl::PDL_OperandOp>();

  auto attr =
      builder.Build<ir::pdl::PDL_AttributeOp>(ir::Int32Attribute::get(ctx, 1));

  auto out_type = builder.Build<ir::pdl::PDL_TypeOp>();

  auto add_op = builder.Build<ir::pdl::PDL_OperationOp>(
      "pd.add",
      std::vector<std::string>{"axis"},
      std::vector<ir::OpResult>{x->result(0), y->result(0)} /*operands*/,
      std::vector<ir::OpResult>{attr->result(0)} /*attrs*/,
      std::vector<ir::OpResult>{out_type->result(0)} /*outputs*/,
      ir::pdl::OperationType::get(ctx));

  builder.Build<ir::pdl::PDL_ResultOp>(
      0, add_op->result(0), ir::pdl::ValueType::get(ctx));

  std::stringstream os;
  program.Print(os);
  LOG(INFO) << os.str();
}

TEST(pdl_dialect, replace_op) {
  INIT();
  auto pattern_op = program_builder.Build<ir::pdl::PDL_PatternOp>(2);
  auto* block = pattern_op.block();
  ir::Builder builder = ir::Builder(ctx, block);

  auto x = builder.Build<ir::pdl::PDL_OperandOp>();
  auto y = builder.Build<ir::pdl::PDL_OperandOp>();

  auto attr =
      builder.Build<ir::pdl::PDL_AttributeOp>(ir::Int32Attribute::get(ctx, 1));

  auto out_type = builder.Build<ir::pdl::PDL_TypeOp>();

  auto add_op = builder.Build<ir::pdl::PDL_OperationOp>(
      "pd.add",
      std::vector<std::string>{"axis"},
      std::vector<ir::OpResult>{x->result(0), y->result(0)} /*operands*/,
      std::vector<ir::OpResult>{attr->result(0)} /*attrs*/,
      std::vector<ir::OpResult>{out_type->result(0)} /*outputs*/,
      ir::pdl::OperationType::get(ctx));

  // case 1.
  builder.Build<ir::pdl::PDL_ReplaceOp>(
      add_op.result(0), std::vector<ir::OpResult>{x->result(0)});

  // case 2.
  auto op2 = builder.Build<ir::pdl::PDL_OperationOp>(
      "pd.add",
      std::vector<std::string>{"axis"},
      std::vector<ir::OpResult>{x->result(0), y->result(0)} /*operands*/,
      std::vector<ir::OpResult>{attr->result(0)} /*attrs*/,
      std::vector<ir::OpResult>{out_type->result(0)} /*outputs*/,
      ir::pdl::OperationType::get(ctx));
  builder.Build<ir::pdl::PDL_ReplaceOp>(add_op.result(0), op2->result(0));

  std::stringstream os;
  program.Print(os);
  LOG(INFO) << os.str();
}

TEST(pdl_dialect, apply_native_constraint) {
  INIT();
  auto pattern_op = program_builder.Build<ir::pdl::PDL_PatternOp>(2);
  auto* block = pattern_op.block();
  ir::Builder builder = ir::Builder(ctx, block);

  auto type_op = builder.Build<ir::pdl::PDL_TypeOp>();
  auto attribute_op = builder.Build<ir::pdl::PDL_AttributeOp>();
  auto operand_op = builder.Build<ir::pdl::PDL_OperandOp>();
  auto operation_op = builder.Build<ir::pdl::PDL_OperationOp>(
      "test_op",
      std::vector<std::string>{},
      std::vector<ir::OpResult>{operand_op->result(0)} /*operands*/,
      std::vector<ir::OpResult>{attribute_op->result(0)} /*attrs*/,
      std::vector<ir::OpResult>{type_op->result(0)} /*outputs*/,
      ir::pdl::OperationType::get(ctx));

  builder.Build<ir::pdl::PDL_ApplyNativeConstraintOp>(
      "constraint_function",
      std::vector<ir::OpResult>{type_op->result(0),
                                attribute_op->result(0),
                                operand_op->result(0),
                                operation_op->result(0)});

  std::stringstream os;
  program.Print(os);
  LOG(INFO) << os.str();
}

TEST(pdl_dialect, rewrite_op) {
  INIT();
  auto pattern_op = program_builder.Build<ir::pdl::PDL_PatternOp>(2);
  auto* block = pattern_op.block();
  ir::Builder builder = ir::Builder(ctx, block);

  auto x = builder.Build<ir::pdl::PDL_OperandOp>();
  auto y = builder.Build<ir::pdl::PDL_OperandOp>();

  auto attr =
      builder.Build<ir::pdl::PDL_AttributeOp>(ir::Int32Attribute::get(ctx, 1));

  auto out_type = builder.Build<ir::pdl::PDL_TypeOp>();

  auto add_op = builder.Build<ir::pdl::PDL_OperationOp>(
      "pd.add",
      std::vector<std::string>{"axis"},
      std::vector<ir::OpResult>{x->result(0), y->result(0)} /*operands*/,
      std::vector<ir::OpResult>{attr->result(0)} /*attrs*/,
      std::vector<ir::OpResult>{out_type->result(0)} /*outputs*/,
      ir::pdl::OperationType::get(ctx));

  // case 1.
  auto rewrite_op = builder.Build<ir::pdl::PDL_RewriteOp>(add_op.result(0));
  auto rewrite_block = rewrite_op.block();
  auto rewrite_builder = ir::Builder(ctx, rewrite_block);
  rewrite_builder.Build<ir::pdl::PDL_ReplaceOp>(
      add_op.result(0), std::vector<ir::OpResult>{x->result(0)});

  // case 2.
  auto rewrite2_op = builder.Build<ir::pdl::PDL_RewriteOp>(add_op.result(0));
  auto rewrite2_block = rewrite2_op.block();
  auto rewrite2_builder = ir::Builder(ctx, rewrite2_block);
  rewrite2_builder.Build<ir::pdl::PDL_ApplyNativeRewriteOp>(
      "rewrite_function",
      std::vector<ir::OpResult>{add_op.result(0), x.result(0), attr.result(0)},
      std::vector<ir::Type>{ir::pdl::OperationType::get(ctx),
                            ir::pdl::ValueType::get(ctx),
                            ir::pdl::AttributeType::get(ctx)});

  std::stringstream os;
  program.Print(os);
  LOG(INFO) << os.str();
}
