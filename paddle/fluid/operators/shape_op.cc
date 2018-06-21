/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/shape_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class ShapeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input (Input) of get_shape op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output (Out) of get_shape op should not be null.");
    auto in_dim = ctx->GetInputDim("Input");
    ctx->SetOutputDim("Out", {in_dim.size()});
  }
};

class ShapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor), The input tensor.");
    AddOutput("Out",
              "(Tensor), The shape of input tensor, the data type of the shape"
              " is int64_t, will be on the same device with the input Tensor.");
    AddComment(R"DOC(
Shape Operator

Get the shape of input tensor. Only support CPU input Tensor now.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(shape, ops::ShapeOp, ops::ShapeOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(shape, ops::ShapeKernel<int>, ops::ShapeKernel<int64_t>,
                       ops::ShapeKernel<float>, ops::ShapeKernel<double>);
