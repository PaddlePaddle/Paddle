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

#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/op_base.h"
#include "paddle/ir/core/region.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

#include "paddle/fluid/ir/interface/infermeta.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/nullary.h"

// Define op
class OperationTest
    : public ir::Op<OperationTest, paddle::dialect::InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "test.operation2"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];  // NOLINT
  static void Verify() {}
  static void InferMeta(phi::InferMetaContext *infer_meta) {
    auto fn = PD_INFER_META(phi::CreateInferMeta);
    fn(infer_meta);
  }
};
IR_DECLARE_EXPLICIT_TYPE_ID(OperationTest)
IR_DEFINE_EXPLICIT_TYPE_ID(OperationTest)

const char *OperationTest::attributes_name[attributes_num] = {  // NOLINT
    "op2_attr1",
    "op2_attr2"};

// Define a dialect, op1 and op2 will be registered by this dialect.
class TestDialect : public ir::Dialect {
 public:
  explicit TestDialect(ir::IrContext *context)
      : ir::Dialect(name(), context, ir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "test"; }

 private:
  void initialize() { RegisterOps<OperationTest>(); }
};
IR_DECLARE_EXPLICIT_TYPE_ID(TestDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(TestDialect)

TEST(infershape_test, infershape_test) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  EXPECT_EQ(test_dialect != nullptr, true);

  // (2) Get registered operations.

  std::string op_name = OperationTest::name();
  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);

  std::vector<ir::OpResult> op_inputs = {};
  std::vector<ir::Type> op_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op =
      ir::Operation::Create(op_inputs, {}, op_output_types, op_info);

  paddle::dialect::InferMetaInterface interface =
      op->dyn_cast<paddle::dialect::InferMetaInterface>();
  phi::InferMetaContext infer_meta_ctx;
  infer_meta_ctx.EmplaceBackAttr(phi::IntArray({5, 6}));
  infer_meta_ctx.EmplaceBackAttr(phi::DataType::FLOAT32);

  phi::DenseTensor tensor;
  infer_meta_ctx.EmplaceBackOutput(phi::MetaTensor(&tensor));
  interface.InferMeta(&infer_meta_ctx);

  EXPECT_EQ(tensor.dims().size(), 2);
  EXPECT_EQ(tensor.dims()[0], 5);
  EXPECT_EQ(tensor.dims()[1], 6);
}
