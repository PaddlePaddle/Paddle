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

#include "paddle/ir/builtin_op.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/dialect.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/op_base.h"

// Define op1.
class Operation1 : public ir::Op<Operation1> {
 public:
  using Op::Op;
  static const char *name() { return "Operation1"; }
  static const char *attributes_name_[];
};
const char *Operation1::attributes_name_[] = {"op1_attr1", "op1_attr2"};

// Define op2.
class Operation2
    : public ir::Op<Operation2, ir::ReadOnlyTrait, ir::InferShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "Operation2"; }
  static const char *attributes_name_[];
  static void InferShape() {
    std::cout << "This is op2's InferShape interface." << std::endl;
  }
};
const char *Operation2::attributes_name_[] = {"op2_attr1", "op2_attr2"};

// Define a dialect, op1 and op2 will be registered by this dialect.
class TestDialect : public ir::Dialect {
 public:
  explicit TestDialect(ir::IrContext *context)
      : ir::Dialect(name(), context, ir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "op_test"; }

 private:
  void initialize() { RegisterOperations<Operation1, Operation2>(); }
};

ir::DictionaryAttribute CreateAttribute(std::string attribute_name,
                                        std::string attribute) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::StrAttribute attr_name = ir::StrAttribute::get(ctx, attribute_name);
  ir::Attribute attr_value = ir::StrAttribute::get(ctx, attribute);
  std::map<ir::StrAttribute, ir::Attribute> named_attr;
  named_attr.insert(
      std::pair<ir::StrAttribute, ir::Attribute>(attr_name, attr_value));
  return ir::DictionaryAttribute::get(ctx, named_attr);
}

TEST(op_test, op_test) {
  // (1) Register Dialect, Operation1, Operation2 into IrContext.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  std::cout << test_dialect << std::endl;

  // (2) Get registered operations
  std::unordered_map<ir::TypeId, ir::OpInfoImpl *> operations =
      ctx->registed_operation();
  EXPECT_EQ(operations.count(ir::TypeId::get<Operation1>()) == 1, true);
  EXPECT_EQ(operations.count(ir::TypeId::get<Operation2>()) == 1, true);
  ir::OpInfoImpl *op1_info = operations[ir::TypeId::get<Operation1>()];
  ir::OpInfoImpl *op2_info = operations[ir::TypeId::get<Operation2>()];
  EXPECT_EQ(op1_info->HasTrait<ir::ReadOnlyTrait>(), false);
  EXPECT_EQ(op1_info->HasInterface<ir::InferShapeInterface>(), false);
  EXPECT_EQ(op2_info->HasTrait<ir::ReadOnlyTrait>(), true);
  EXPECT_EQ(op2_info->HasInterface<ir::InferShapeInterface>(), true);

  // 1. Construct OP1: a = OP1()
  std::vector<ir::OpResult> op1_inputs = {};
  std::vector<ir::Type> op1_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op1 =
      ir::Operation::create(op1_inputs,
                            op1_output_types,
                            CreateAttribute("op1_name", "op1_attr"),
                            op2_info);

  ir::InferShapeInterface::Concept *concept =
      op2_info->GetInterfaceImpl<ir::InferShapeInterface>();
  concept->infer_shape_(op1);

  op1->destroy();
}
