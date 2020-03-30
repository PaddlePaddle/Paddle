/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/inverse_op.h"

namespace paddle {
namespace operators {

class InverseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::NotFound("Input of inverse_op is not set."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Output"), true,
        platform::errors::NotFound("Output of inverse_op is not set."));

    auto input_dims = ctx->GetInputDim("Input");
    int64_t input_rank = input_dims.size();
    PADDLE_ENFORCE_GE(
        input_rank, 2,
        platform::errors::InvalidArgument(
            "Expected the rank of Input >= 2. Recived %d.", input_rank));
    PADDLE_ENFORCE_EQ(input_dims[input_rank - 1], input_dims[input_rank - 2],
                      platform::errors::InvalidArgument(
                          "The last two dimensions are expected to be equal."));

    ctx->SetOutputDim("Output", input_dims);
    ctx->ShareLoD("Input", /*->*/ "Output");
  }
};

class InverseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(Tensor) A square matrix (2-D Tensor) or batches of square matrixs"
        " to inverse.");
    AddOutput("Output", "(Tensor) The inverse matrix of input.");
    AddComment(R"DOC(
Inverse Operator

Takes the inverse of the square matrix.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(inverse, ops::InverseOp, ops::InverseOpMaker);
REGISTER_OP_CPU_KERNEL(
    inverse, ops::InverseKernel<paddle::platform::CPUDeviceContext, float>);
