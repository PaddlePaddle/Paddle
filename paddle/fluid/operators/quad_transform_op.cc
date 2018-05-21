/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/quad_transform_op.h"

namespace paddle {
namespace operators {

class QuadTransformOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input (Input) of quad transform op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Output"),
                   "Output (Output) of quad transform op should not be null.");

    auto in_dim = ctx->GetInputDim("Input");

    PADDLE_ENFORCE_EQ(in_dim.size(), 4, "input's rank must be 4.");
    PADDLE_ENFORCE_EQ(in_dim[1], 8, "input's second dimension must be 8");

    ctx->SetOutputDim("Input", in_dim);
    ctx->ShareLoD("Input", /*->*/ "Output");
  }
};

class QuadTransformOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input with shape [batch_size, 8, height, width]");
    AddOutput("Output", "The output with the same shape as input");

    AddComment(R"DOC(
QuadTransform Operator.
The input is the final geometry output in detection network.
We use 8 numbers to denote the coordinate shift from four corner vertices of
the quadrangle to the pixel location. As each distance offset contains two numbers (xi, yi),
the geometry output contains 8 channels.
QuadTransform Operator is used to transform the coordinate shift to the real coordinate.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(quad_transform, ops::QuadTransformOp,
                  ops::QuadTransformOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    quad_transform, ops::QuadTransformKernel<paddle::platform::CPUPlace, float>,
    ops::QuadTransformKernel<paddle::platform::CPUPlace, double>);
