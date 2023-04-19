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
  void initialize() { RegisterOperations<Operation1, Operation1>(); }
};

TEST(op_test, op_test) {
  // (1) Register Dialect, Operation1, Operation2 into IrContext.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  std::cout << test_dialect << std::endl;
}
