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

class OperatorTest : public OperatorBase {
 public:
  void Run(OpContext* context) const override {
    float scale = GetAttr<float>("scale");
    PADDLE_ENFORCE(Input(context->scope, 0) == nullptr, "Input(0) should not initialized");
    PADDLE_ENFORCE(Input(context->scope, 1) == nullptr, "Input(1) should not initialized");
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
  std::string op_name = "op1";
  op_desc.set_type("test_operator");
  op_desc.set_name(op_name);
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

  Scope* scope = new Scope();
  DeviceContext* device_context = new DeviceContext();
  OpContext* op_context = new OpContext();
  op_context->scope = scope;
  op_context->device_context = device_context;

  OperatorBase* op = paddle::framework::OpRegistry::CreateOp(op_desc);
  ASSERT_EQ(op->desc().name(), op_name);
  ASSERT_EQ(op->inputs(), inputs);
  ASSERT_EQ(op->outputs(), outputs);
  ASSERT_EQ(op->GetAttr<float>("scale"), scale);
  op->Run(op_context);
}

}  // namespace framework
}  // namespace paddle