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
#include "paddle/framework/scope.h"
#include "gtest/gtest.h"

namespace paddle {
namespace framework {

class OperatorTest: public OperatorBase {
  void Run(OpRunContext* context) const override {}
  void InferShape(const Scope* scope) const override {}
};

TEST(OperatorBase, DebugString) {
  Scope* scope = new Scope();
  DeviceContext* device_context = new DeviceContext();
  OpRunContext* op_context = new OpRunContext();
  op_context->scope = scope;
  op_context->device_context = device_context;

  auto op = new OperatorTest();
  op->inputs.push_back("X");
  op->inputs.push_back("Y");
  op->outputs.push_back("O");
  op->attrs["scale"] = 0;

  printf("%s\n", op->DebugString().c_str());

}

} // namespace framework
} // namespace paddle