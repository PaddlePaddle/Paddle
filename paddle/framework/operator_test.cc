/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/operator.h"
#include "gtest/gtest.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace framework {

class OperatorTest : public OperatorWithKernel {
 public:
  void Run(const OpRunContext* ctx) const override {
    float scale = ctx->op_->GetAttr<float>("scale");
    PADDLE_ENFORCE(ctx->Input(0) == nullptr, "Input(0) should not initialized");
    PADDLE_ENFORCE(ctx->Output(0) == nullptr,
                   "Output(1) should not initialized");
    auto output1 = ctx->scope_->CreateVariable("output1");
    PADDLE_ENFORCE(output1 != nullptr, "should create output1 from scope");
    printf("get attr %s = %f\n", "scale", scale);
    printf("%s\n", DebugString().c_str());
  }
};

class OperatorTestProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  OperatorTestProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of test op");
    AddOutput("output", "output of test op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddType("test_operator");
    AddComment("This is test op");
  }
};

REGISTER_OP(OperatorTest, OperatorTestProtoAndCheckerMaker, test_operator)

TEST(OperatorBase, DebugString) {
  OpDesc op_desc;
  op_desc.set_type("test_operator");
  std::vector<std::string> inputs = {"IN1", "IN2"};
  for (auto& input : inputs) {
    op_desc.add_inputs(input);
  }
  std::vector<std::string> outputs = {"OUT1", "OUT2"};
  for (auto& output : outputs) {
    op_desc.add_outputs(output);
  }
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  float scale = 3.14;
  attr->set_f(scale);

  DeviceContext device_context;
  auto scope = std::make_shared<Scope>();

  OperatorBase* op = paddle::framework::OpRegistry::CreateOp(op_desc);
  ASSERT_EQ(op->inputs_, inputs);
  ASSERT_EQ(op->outputs_, outputs);
  ASSERT_EQ(op->GetAttr<float>("scale"), scale);
  op->Run(scope, &device_context);
}

}  // namespace framework
}  // namespace paddle