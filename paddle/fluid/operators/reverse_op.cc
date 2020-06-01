// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/reverse_op.h"
#include <memory>
#include <vector>

namespace paddle {
namespace operators {

class ReverseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::InvalidArgument("Output(Out) should not be null"));
    const auto& x_dims = ctx->GetInputDim("X");
    const auto& axis = ctx->Attrs().Get<std::vector<int>>("axis");
    PADDLE_ENFORCE_NE(axis.empty(), true, platform::errors::InvalidArgument(
                                              "'axis' can not be empty."));
    for (int a : axis) {
      PADDLE_ENFORCE_LT(a, x_dims.size(),
                        paddle::platform::errors::OutOfRange(
                            "The axis must be less than input tensor's rank. "
                            "but got %d >= %d",
                            a, x_dims.size()));
      PADDLE_ENFORCE_GE(
          a, -x_dims.size(),
          paddle::platform::errors::OutOfRange(
              "The axis must be greater than the negative number of "
              "input tensor's rank, but got %d < %d",
              a, -x_dims.size()));
    }
    ctx->SetOutputDim("Out", x_dims);
  }
};

class ReverseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The LoDTensor to be flipped.");
    AddOutput("Out", "The LoDTensor after flipping.");
    AddAttr<std::vector<int>>(
        "axis", "The axises that along which order of elements is reversed.");
    AddComment(R"DOC(
      Reverse Operator.

      Reverse the order of elements in the input LoDTensor along given axises.

      Case 1:
        Given
            X = [[1, 2, 3, 4, 5]
                 [6, 7, 8, 9, 10]
                 [11, 12, 13, 14, 15]],
        and
            axis = [0],
        we get:
            Out = [[11, 12, 13, 14, 15]
                   [6, 7, 8, 9, 10]
                   [1, 2, 3, 4, 5]].
        
      Case 2:
        Given
            X = [[[1, 2, 3, 4]
                  [5, 6, 7, 8]]
                 [[9, 10, 11, 12]
                  [13, 14, 15, 16]]],
        and
            axis = [0, 2],
        we get:
            Out = [[[12, 11, 10, 9]
                    [16, 15, 14, 13]]
                   [[4, 3, 2, 1]
                    [8, 7, 6, 5]]],
    )DOC");
  }
};

template <typename T>
class ReverseGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("reverse");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttr("axis", this->GetAttr("axis"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(reverse, ops::ReverseOp, ops::ReverseOpMaker,
                  ops::ReverseGradMaker<paddle::framework::OpDesc>,
                  ops::ReverseGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(reverse_grad, ops::ReverseOp);
REGISTER_OP_CPU_KERNEL(
    reverse, ops::ReverseKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ReverseKernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::ReverseKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ReverseKernel<paddle::platform::CPUDeviceContext, bool>,
    ops::ReverseKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ReverseKernel<paddle::platform::CPUDeviceContext, double>)
