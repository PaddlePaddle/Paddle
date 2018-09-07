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

#include "paddle/fluid/operators/reshape_op.h"

#include <string>
#include <vector>

namespace paddle {
namespace operators {

class ReshapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor). The input tensor of reshape operator.");
    AddInput("Shape",
             "(Tensor<int32>, optional). If provided, reshape according to "
             "this given shape. That is to say it has a higher priority than "
             "the shape attribute, while the shape attribute still should be "
             "set correctly to gurantee shape inference in compile time.")
        .AsDispensable();
    AddOutput("Out", "(Tensor). The output tensor of reshape operator.");
    AddAttr<std::vector<int>>(
        "shape", "(std::vector<int>) Target shape of reshape operator.");
    AddAttr<bool>("inplace",
                  "(default: false) Change the source tensor's shape without "
                  "memory copy. When Attr(inplace) is set true, the output "
                  "tensor shares memory with Input(X), otherwise, a new output "
                  "tensor is created, and its data are copied from Input(x).")
        .SetDefault(false);
    AddComment(R"DOC(
Reshape Operator.

Reshape Input(X) into the shape specified by Attr(shape) or Input(Shape). The
data in Input(X) are unchanged.

Examples:

1. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [6, 8], the reshape operator will transform Input(X)
into a 2-D tensor with shape [6, 8] and leaving Input(X)'s data unchanged.

2. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [2, 3, -1, 2], the reshape operator will transform
Input(X) into a 4-D tensor with shape [2, 3, 4, 2] and leaving Input(X)'s data
unchanged. In this case, one and only dimension of Attr(shape) can be set to -1,
the value of this dimension is inferred from the total element number of
Input(X) and remaining dimensions.

3. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [-1, 0, 3, 2], the reshape operator will transform
Input(X) into a 4-D tensor with shape [2, 4, 3, 2] and leaving Input(X)'s data
unchanged. In this case, besides -1, 0 means the actual dimension value is going
to be copied from the corresponding dimension of Input(X).

Note:

1. One and only one dimension in Attr(shape) can be set -1. In this case,
the actual dimension value will be infered from the total element number of
Input(X) and remaining dimensions.

2. More than one dimensions in Attr(shape) can be set to 0, which means the real
dimension value will be copied from Input(X) at runtime. Note that the index of
0 can not exceed Rank(X). For example, Input(X) is a 3-D tensor with shape
[2, 3, 4], Attr(shape) = [2, 3, 2, 0] is an invalid input.

3. Input(Shape) has a higher priority than Attr(shape) if it is provided, while
Attr(shape) still should be set correctly to gurantee shape inference in 
compile-time.

)DOC");
  }
};

class ReshapeGradOp : public framework::OperatorWithKernel {
 public:
  ReshapeGradOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(reshape, ops::ReshapeOp, ops::ReshapeOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(reshape_grad, ops::ReshapeGradOp);
REGISTER_OP_CPU_KERNEL(reshape, ops::ReshapeKernel<CPU, float>,
                       ops::ReshapeKernel<CPU, double>,
                       ops::ReshapeKernel<CPU, int>,
                       ops::ReshapeKernel<CPU, int64_t>);
REGISTER_OP_CPU_KERNEL(reshape_grad, ops::ReshapeGradKernel<CPU, float>,
                       ops::ReshapeGradKernel<CPU, double>,
                       ops::ReshapeGradKernel<CPU, int>,
                       ops::ReshapeGradKernel<CPU, int64_t>);
