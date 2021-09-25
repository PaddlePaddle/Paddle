// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/matrix_power_op.h"

namespace paddle {
namespace operators {

class MatrixPowerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "matrix_power");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "matrix_power");
    auto dims = ctx->GetInputDim("X");
    auto n_dim = dims.size();
    PADDLE_ENFORCE_GE(n_dim, 2,
                      platform::errors::InvalidArgument(
                          "The Input(X) should have at least 2 dimensions. But "
                          "received a %d dimension tensor.",
                          n_dim));
    PADDLE_ENFORCE_EQ(dims[n_dim - 2], dims[n_dim - 1],
                      platform::errors::InvalidArgument(
                          "The inner-most 2 dimensions of Input(X) all should "
                          "be square matrices "
                          "But received X's shape[-2] = %d and shape[-1] = %d.",
                          dims[n_dim - 2], dims[n_dim - 1]));
    ctx->SetOutputDim("Out", dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class MatrixPowerOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor), The input tensor of matrix_power op. Its shape should be "
        "[*, M, M] where * is zero or more batch dimensions, and matrices "
        "on the inner-most 2 dimensions all should be square matrices.");
    AddOutput("Out",
              "(Tensor), The output tensor of matrix_power op. It has the same "
              "shape as the input.");
    AddAttr<int>("n", "(int), The exponent used to calculate the power of X.");
    AddComment(R"DOC(
Matrix Power Operator.

Computes the n-th power of a square matrix or a batch of square matrices.

)DOC");
  }
};

class MatrixPowerOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> u_map{
        {"X", /*->*/ "Out"}};
    return u_map;
  }
};

class MatrixPowerGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matrix_power_grad");
    OP_INOUT_CHECK(context->HasInput("Out"), "Input", "Out",
                   "matrix_power_grad");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "matrix_power_grad");
    auto x_dims = context->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (context->HasOutput(x_grad_name)) {
      context->SetOutputDim(x_grad_name, x_dims);
    }
  }
};

template <typename T>
class MatrixPowerGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(matrix_power, ops::MatrixPowerOp, ops::MatrixPowerOpMaker,
                  ops::MatrixPowerOpInferVarType,
                  ops::MatrixPowerGradOpMaker<paddle::framework::OpDesc>,
                  ops::MatrixPowerGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(matrix_power_grad, ops::MatrixPowerGradOp);

REGISTER_OP_CPU_KERNEL(
    matrix_power,
    ops::MatrixPowerKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MatrixPowerKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    matrix_power_grad,
    ops::MatrixPowerGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MatrixPowerGradKernel<paddle::platform::CPUDeviceContext, double>);
