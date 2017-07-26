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

#include <gtest/gtest.h>
#include "paddle/framework/op_registry.h"
namespace paddle {
namespace framework {

class EmptyOp : public OperatorBase {
 public:
  void InferShape(const std::shared_ptr<Scope> &scope) const override {}
  void Run(const std::shared_ptr<Scope> &scope,
           const platform::DeviceContext &dev_ctx) const override {}
};

class RowwiseAddOp : public EmptyOp {};
class RowwiseAddOpMaker : public OpProtoAndCheckerMaker {
 public:
  RowwiseAddOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input X of Add").IgnoreGradient();
    AddInput("b", "Bias of Add").IgnoreGradient();
    AddOutput("Out", "Out of Add").IgnoreGradient();
    AddComment("Add Op");
  }
};

class RowwiseAddGradOp : public EmptyOp {};
}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;
REGISTER_OP(rowwise_add, f::RowwiseAddOp, f::RowwiseAddOpMaker);
REGISTER_GRADIENT_OP(rowwise_add, rowwise_add_grad, f::RowwiseAddGradOp);

TEST(Backward, simple_grad) {
  auto fwd = f::OpRegistry::CreateOp("rowwise_add", {"X", "b"}, {"Out"}, {});
  ASSERT_NE(fwd, nullptr);
}