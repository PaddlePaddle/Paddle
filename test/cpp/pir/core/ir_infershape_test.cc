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

#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/region.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/nullary.h"

#include "test/cpp/pir/tools/macros_utils.h"

// Define op
class OperationTest
    : public pir::Op<OperationTest, paddle::dialect::InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "test.operation2"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];  // NOLINT
  static void VerifySig() {}
  static void InferMeta(phi::InferMetaContext *infer_meta) {
    auto fn = PD_INFER_META(phi::CreateInferMeta);
    fn(infer_meta);
  }
  static std::vector<pir::Type> InferMeta(
      const std::vector<pir::Value> &input_values, pir::AttributeMap *) {
    VLOG(4) << "Start infermeta OperationTest";
    std::vector<pir::Type> argument_outputs;
    return argument_outputs;
  }
};
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(OperationTest)
IR_DEFINE_EXPLICIT_TYPE_ID(OperationTest)

const char *OperationTest::attributes_name[attributes_num] = {  // NOLINT
    "op2_attr1",
    "op2_attr2"};

// Define a dialect, op1 and op2 will be registered by this dialect.
class TestDialect : public pir::Dialect {
 public:
  explicit TestDialect(pir::IrContext *context)
      : pir::Dialect(name(), context, pir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "test"; }

 private:
  void initialize() { RegisterOps<OperationTest>(); }
};
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(TestDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(TestDialect)

TEST(infershape_test, infershape_test) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Dialect *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  EXPECT_EQ(test_dialect != nullptr, true);

  // (2) Get registered operations.

  std::string op_name = OperationTest::name();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);

  std::vector<pir::Value> op_inputs = {};
  std::vector<pir::Type> op_output_types = {pir::Float32Type::get(ctx)};
  pir::Operation *op =
      pir::Operation::Create(op_inputs, {}, op_output_types, op_info);

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
