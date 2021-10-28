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
#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {

class InverseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "Inverse");
    OP_INOUT_CHECK(ctx->HasOutput("Output"), "Output", "Output", "Inverse");

    auto input_dims = ctx->GetInputDim("Input");
    int64_t input_rank = input_dims.size();
    PADDLE_ENFORCE_GE(
        input_rank, 2,
        platform::errors::InvalidArgument(
            "The dimension of Input(Input) is expected to be no less than 2. "
            "But recieved: Input(Input)'s dimension = %d, shape = [%s].",
            input_rank, input_dims));
    for (int64_t i = 0; i < input_rank; ++i) {
      PADDLE_ENFORCE_EQ(
          (input_dims[i] == -1) || (input_dims[i] > 0), true,
          platform::errors::InvalidArgument(
              "Each dimension of input tensor is expected to be -1 or a "
              "positive number, but recieved %d. Input's shape is [%s].",
              input_dims[i], input_dims));
    }
    if (input_dims[input_rank - 2] > 0 && input_dims[input_rank - 1] > 0) {
      PADDLE_ENFORCE_EQ(input_dims[input_rank - 2], input_dims[input_rank - 1],
                        platform::errors::InvalidArgument(
                            "The last two dimensions are expected to be equal. "
                            "But recieved: %d and %d; "
                            "Input(Input)'s shape = [%s].",
                            input_dims[input_rank - 2],
                            input_dims[input_rank - 1], input_dims));
    }

    ctx->SetOutputDim("Output", input_dims);
    ctx->ShareLoD("Input", /*->*/ "Output");
  }
};

class InverseOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{
        {"Input", /*->*/ "Output"}};
    return m;
  }
};

class InverseGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto input_grad = framework::GradVarName("Input");
    auto output_grad = framework::GradVarName("Output");

    OP_INOUT_CHECK(ctx->HasInput("Output"), "Input", "Output", "InverseGrad");
    OP_INOUT_CHECK(ctx->HasInput(output_grad), "Input", output_grad,
                   "InverseGrad");

    if (ctx->HasOutput(input_grad)) {
      ctx->SetOutputDim(input_grad, ctx->GetInputDim(output_grad));
    }
  }
};

class InverseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(Tensor) A square matrix (2-D Tensor) or batches of square matrices"
        " to inverse.");
    AddOutput("Output", "(Tensor) The inverse of input matrix.");
    AddComment(R"DOC(
Inverse Operator

Takes the inverse of the square matrix.
)DOC");
  }
};

template <typename T>
class InverseGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad) const override {
    grad->SetType(this->ForwardOpType() + "_grad");
    grad->SetInput("Output", this->Output("Output"));
    grad->SetInput(framework::GradVarName("Output"),
                   this->OutputGrad("Output"));
    grad->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(inverse, ops::InverseOp, ops::InverseOpMaker,
                  ops::InverseOpInferVarType,
                  ops::InverseGradOpMaker<paddle::framework::OpDesc>,
                  ops::InverseGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(inverse_grad, ops::InverseGradOp);

REGISTER_OP_CPU_KERNEL(
    inverse, ops::InverseKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InverseKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    inverse_grad,
    ops::InverseGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InverseGradKernel<paddle::platform::CPUDeviceContext, double>);
