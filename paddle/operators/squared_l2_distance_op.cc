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
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input of SquaredL2DistanceOp "
                            "must be initialized.");
    PADDLE_ENFORCE_EQ(ctx.Input<Tensor>("X")->dims(),
                      ctx.Input<Tensor>("Y")->dims(),
                      "Dimensions of SquaredL2DistanceOp's two inputs "
                      "must be same.")
    framework::DDim dims = ctx.Input<Tensor>("X")->dims();
    ctx.Output<Tensor>("sub_result")->Resize(dims);
    ctx.Output<Tensor>("Out")->Resize(framework::make_ddim({dims[0], 1}));
  }
};

class SquaredL2DistanceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SquaredL2DistanceOpMaker(framework::OpProto *proto,
                           framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input value.");
    AddInput("Y", "Target value.");
    AddOutput("sub_result",
              "Buffering substraction result which "
              "will be reused in backward.")
        .AsIntermediate();
    AddOutput("Out", "Squared l2 distance between input and target.");
    AddComment(R"DOC(
    SquaredL2DistanceOp will cacluate the squared L2 distances for
    input and target. Number of distance value equals to the
    first dimension of input.
    )DOC");
  }
};

class SquaredL2DistanceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    ctx.Output<Tensor>(framework::GradVarName("X"))
        ->Resize(ctx.Input<Tensor>("X")->dims());
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
