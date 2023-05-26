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

#include <cstring>

#include "glog/logging.h"

#include "paddle/ir/builtin_type.h"
#include "paddle/ir/dialect.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/op_base.h"
#include "paddle/ir/operation.h"
#include "paddle/pass/pass.h"
#include "paddle/pass/pass_manager.h"

ir::AttributeMap CreateAttributeMap(ir::IrContext *ctx,
                                    std::string attribute_name,
                                    std::string attribute) {
  ir::Attribute attr_value = ir::StrAttribute::get(ctx, attribute);
  ir::AttributeMap attr_map;
  attr_map.insert(
      std::pair<std::string, ir::Attribute>(attribute_name, attr_value));
  return attr_map;
}

class TestOp : public ir::Op<TestOp> {
 public:
  using Op::Op;
  static const char *name() { return "TestDialect.TestOp"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const ir::AttributeMap &attributes) {
    if (attributes.count("op1_attr1") == 0 ||
        !attributes.at("op1_attr1").isa<ir::StrAttribute>()) {
      throw("Type of attribute: parameter_name is not right.");
    }
  }
};
const char *TestOp::attributes_name[attributes_num] = {"op1_attr1"};

class TestDialect : public ir::Dialect {
 public:
  explicit TestDialect(ir::IrContext *context)
      : ir::Dialect(name(), context, ir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "TestDialect"; }

 private:
  void initialize() { RegisterOps<TestOp>(); }
};

class TestPass : public ir::Pass {
 public:
  TestPass() : ir::Pass("TestPass", 1) {}
  void Run(ir::Operation *op) override {
    auto test_op = op->dyn_cast<TestOp>();
    CHECK_EQ(test_op.operation(), op);
    CHECK_EQ(test_op.name(), test_op->op_info().name());
    LOG(INFO) << "In " << info_.name << ": " << test_op->op_info().name();
  }

  bool CanScheduleOn(ir::Operation *op) const override {
    return std::strcmp(op->op_info().name(), "TestDialect.TestOp") == 0;
  }
};

TEST(pass_manager_test, pass_manager_test) {
  // (1) Register Dialect, Operation into IrContext.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  CHECK_EQ(test_dialect != nullptr, true);

  // (2) Get registered operations.
  std::string op_name = std::string(TestOp::name());
  auto op_info = ctx->GetRegisteredOpInfo(op_name);
  CHECK_EQ(op_info != nullptr, true);

  // (3) Test uses for op.
  std::vector<ir::OpResult> op_inputs = {};
  std::vector<ir::Type> op_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op =
      ir::Operation::create(op_inputs,
                            op_output_types,
                            CreateAttributeMap(ctx, "op1_attr1", "op1_attr1"),
                            op_info);

  // (4) Test pass manager for op.
  ir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<TestPass>());
  CHECK_EQ(pm.Run(op), true);

  op->destroy();
}
