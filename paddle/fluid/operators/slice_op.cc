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

#include "paddle/fluid/operators/slice_op.h"
#include "paddle/fluid/operators/net_op.h"

namespace paddle {
namespace operators {
using framework::Tensor;

class SliceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SliceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Starts"),
                   "Input(Starts) of SliceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Ends"),
                   "Input(Ends) of SliceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Outputs(Out) of SliceOp should not be null.");
    auto input_dims = ctx->GetInputsDim("X");

    auto starts_dims = ctx->GetInputsDim("Starts");
    auto ends_dims = ctx->GetInputsDim("Ends");

    PADDLE_ENFORCE_EQ(starts_dims.size(), ends_dims.size());

    if (ctx->HasInput("Axes")) {
      auto axes_dims = ctx->GetInputsDim("Axes");
      PADDLE_ENFORCE_EQ(axes_dims.size(), starts_dims.size());
      PADDLE_ENFORCE_GE(input_dims.size(), starts_dims.size());
    } else {
      PADDLE_ENFORCE_EQ(input_dims.size(), starts_dims.size());
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
        const framework::ExecutionContext& ctx) const override {
      return framework::OpKernelType(
          framework::ToDataType(ctx.Input<framework::Tensor>("X")->type()),
          ctx.device_context());
  }
};

class SliceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "The gradient of Out should not be null.");
    PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName("X")),
                   "The gradient of X should not be null.");
    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("X")->type()),
        ctx.device_context());
  }
};

class SliceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SliceOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor) ,"
             "the input of the slice operator, not support lod sequnce.");
    AddInput("Axes",
             "(Axes, list of ints, optional) ,"
             "to which 'starts' and 'ends' apply to. if not present, will be "
             "treated as [0, 1, ..., len('starts')-1].").AsDispensable();
    AddInput("Starts",
             "(Starts, list of ints, required) ,"
             "Starting indices of corresponding axis in 'axes'");
    AddInput("Ends",
             "(Ends, list of ints, required) ,"
             "Ending indices (exclusive) of corresponding axis in 'axes'");
    AddOutput("Out",
             "(Out, T, required) ,"
             "Tensor of data to extract slices from.");
    AddComment(R"DOC(
Slice operator

Produces a slice of the input tensor along multiple axes. It does't support sequnce(LoD) Tensor.

Slices uses axes, starts and ends attributes to specify the start and end dimension for each axis in the list of axes, it uses this information to slice the input data tensor. If a negative value is passed for any of the start or end indices, it represent number of elements before the end of that dimension. If the value passed to start or end is larger than the n (the number of elements in this dimension), it represents n. For slicing to the end of a dimension with unknown size, it is recommended to pass in INT_MAX. If axes are omitted, they are set to [0, ..., ndim-1].

Example:
  X = [[1,2,3,4],
       [5,6,7,8]]
  Axes = [0,1]
  Starts = [1,0]
  Ends = [2,3]

  Output = [
      [5,6,7],
  ]

    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(slice, ops::SliceOp, ops::SliceOpMaker,
            slice_grad, ops::SliceGradOp);
REGISTER_OP_CPU_KERNEL(
        slice,
        ops::SliceOpKernel<paddle::platform::CPUDeviceContext, float>,
        ops::SliceOpKernel<paddle::platform::CPUDeviceContext, double>,
        ops::SliceOpKernel<paddle::platform::CPUDeviceContext, int>,
        ops::SliceOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
        slice_grad,
        ops::SliceGradOpKernel<paddle::platform::CPUDeviceContext, float>,
        ops::SliceGradOpKernel<paddle::platform::CPUDeviceContext, double>,
        ops::SliceGradOpKernel<paddle::platform::CPUDeviceContext, int>,
        ops::SliceGradOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
