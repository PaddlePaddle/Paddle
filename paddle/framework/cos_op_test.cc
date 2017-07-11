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
#include "paddle/framework/cos_op.h"

namespace paddle {
namespace framework {

using namespace paddle::platform;

TEST(CosinOp, Run) {
  OpDesc op_desc;
  op_desc.set_type("cos");
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

  DeviceContext* cpu_ctx = new CPUDeviceContext();
  DeviceContext* dev_ctx = new DeviceContext();
  auto scope = std::make_shared<Scope>();

  OperatorBase* op = paddle::framework::OpRegistry::CreateOp(op_desc);
  ASSERT_EQ(op->inputs_, inputs);
  ASSERT_EQ(op->outputs_, outputs);
  ASSERT_EQ(op->GetAttr<float>("scale"), scale);

  // will run on cpu kernel
  op->Run(scope, cpu_ctx);

  // will run on gpu kernel
  op->Run(scope, dev_ctx);
}

}  // namespace framework
}  // namespace paddle
