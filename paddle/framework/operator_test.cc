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
#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

class OperatorTest : public OperatorBase {
 public:
  void Run(OpRunContext* context) const override {
    printf("%s\n", DebugString().c_str());
  }
};



TEST(OperatorBase, DebugString) {
  Scope* scope = new Scope();
  DeviceContext* device_context = new DeviceContext();
  OpRunContext* op_context = new OpRunContext();
  op_context->scope = scope;
  op_context->device_context = device_context;

  auto op = new OperatorTest();
  op->Run(op_context);
}

}  // namespace framework
}  // namespace paddle