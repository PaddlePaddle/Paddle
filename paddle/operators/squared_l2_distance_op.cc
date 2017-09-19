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

#include "paddle/operators/squared_l2_distance_op.h"

namespace paddle {
namespace operators {

class SquaredL2DistanceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(
        ctx.InputVar("X"),
        "Input(X) of SquaredL2DistanceOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(
        ctx.InputVar("Y"),
        "Input(Y) of SquaredL2DistanceOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(
        ctx.OutputVar("sub_result"),
        "Output(sub_result) of SquaredL2DistanceOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(
        ctx.OutputVar("Out"),
        "Output(Out) of SquaredL2DistanceOp should not be null.");

    auto* x = ctx.Input<Tensor>("X");
    auto x_dims = x->dims();
    auto* y = ctx.Input<Tensor>("Y");
    auto y_dims = y->dims();

    PADDLE_ENFORCE_EQ(framework::arity(x_dims), framework::arity(y_dims),
                      "Tensor rank of both SquaredL2DistanceOp's "
                      "inputs must be same.");

    int rank = framework::arity(x_dims);
    PADDLE_ENFORCE_GE(rank, 2, "Tensor rank should be at least equal to 2.");
    PADDLE_ENFORCE_EQ(x->numel() / x_dims[0], y->numel() / y_dims[0],
                      "Product of dimensions expcet the first dimension of "
                      "input and target must be equal.");
    PADDLE_ENFORCE(y_dims[0] == 1 || y_dims[0] == x_dims[0],
                   "First dimension of target must be equal to input "
                   "or to 1.");

    ctx.Output<framework::LoDTensor>("sub_result")
        ->Resize({x_dims[0], x->numel() / x_dims[0]});
    ctx.Output<framework::LoDTensor>("Out")->Resize({x_dims[0], 1});
  }
};

class SquaredL2DistanceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SquaredL2DistanceOpMaker(framework::OpProto* proto,
                           framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of SquaredL2DistanceOp.");
    AddInput("Y", "Target of SquaredL2DistanceOp.");
    AddOutput("sub_result",
              "Buffering substraction result which "
              "will be reused in backward.")
        .AsIntermediate();
    AddOutput("Out", "Squared l2 distance between input and target.");
    AddComment(R"DOC(
    SquaredL2DistanceOp will cacluate the squared L2 distance for
    input and target. Number of distance value equals to the
    first dimension of input. First dimension of target could be equal to
    input or to 1. If the first dimension of target is 1, SquaredL2DistanceOp
    will broadcast target's first dimension to input's first dimension.
    You can decide whether calculate the gradient of input and target.
    )DOC");
  }
};

class SquaredL2DistanceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Gradient of Out should not be null");
    auto out_dims = ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto y_dims = ctx.Input<Tensor>("Y")->dims();
    PADDLE_ENFORCE_EQ(out_dims[0], x_dims[0],
                      "First dimension of output gradient and "
                      "input value must be equal.");
    PADDLE_ENFORCE_EQ(out_dims[1], 1,
                      "Second dimension of output gradient "
                      "must be 1.");
    auto* x_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* y_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));
    if (x_grad) x_grad->Resize(x_dims);
    if (y_grad) y_grad->Resize(y_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(squared_l2_distance, ops::SquaredL2DistanceOp,
            ops::SquaredL2DistanceOpMaker, squared_l2_distance_grad,
            ops::SquaredL2DistanceGradOp);
REGISTER_OP_CPU_KERNEL(
    squared_l2_distance,
    ops::SquaredL2DistanceKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    squared_l2_distance_grad,
    ops::SquaredL2DistanceGradKernel<paddle::platform::CPUPlace, float>);
