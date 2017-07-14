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

#include "gtest/gtest.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

USE_OP_WITHOUT_KERNEL(test_operator);
USE_OP(op_with_kernel);

TEST(OperatorBase, all) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("test_operator");
  *op_desc.mutable_inputs()->Add() = "IN1";
  *op_desc.mutable_outputs()->Add() = "OUT1";
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  float scale = 3.14;
  attr->set_f(scale);

  paddle::platform::CPUDeviceContext device_context;
  auto scope = std::make_shared<paddle::framework::Scope>();

  paddle::framework::OperatorPtr op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  ASSERT_EQ(op->GetAttr<float>("scale"), scale);
  scope->CreateVariable("OUT1");
  op->Run(scope, device_context);
  std::cout << op->DebugString() << std::endl;
}

TEST(OpKernel, all) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("op_with_kernel");
  *op_desc.mutable_inputs()->Add() = "IN1";
  *op_desc.mutable_outputs()->Add() = "OUT1";
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(3.14);

  paddle::platform::CPUDeviceContext cpu_device_context;
  auto scope = std::make_shared<paddle::framework::Scope>();

  paddle::framework::OperatorPtr op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  op->Run(scope, cpu_device_context);
}
