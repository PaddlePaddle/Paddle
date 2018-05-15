/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/has_data_op.h"
#include <string>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class HasDataOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  HasDataOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    // inputs and outputs stored in proto
    AddInput("X", "(LoDTensor) the LoDTensor to check");
    AddOutput("Out", "(LoDTensor) the ouput of has_data_op");
    AddComment(R"DOC(
Has Data Operator.

This operator tests whether the input tensor has data or not.
Out is a boolean scalar.
    )DOC");
  }
};

class HasDataOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of HasDataOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of HasDataOp should not be null.");
    ctx->SetOutputDim("Out", {1});
    ctx->ShareLoD("X", "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        platform::CPUPlace());
    return kt;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(has_data, ops::HasDataOp, ops::HasDataOpMaker);
REGISTER_OP_CPU_KERNEL(
    has_data, ops::HasDataOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::HasDataOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::HasDataOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::HasDataOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
