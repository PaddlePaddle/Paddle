/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/eigh_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class EighOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Eigh");
    OP_INOUT_CHECK(ctx->HasOutput("Eigenvalues"), "Output", "Eigenvalues",
                   "Eigh");
    OP_INOUT_CHECK(ctx->HasOutput("Eigenvectors"), "Output", "Eigenvectors",
                   "Eigh");

    auto input_dim = ctx->GetInputDim("X");
    auto rank = input_dim.size();

    PADDLE_ENFORCE_GE(rank, 2,
                      platform::errors::InvalidArgument(
                          "The Input(X) should have at least 2 dimensions."
                          "But received a %d dimension tensor.",
                          rank));
    PADDLE_ENFORCE_EQ(
        input_dim[rank - 2], input_dim[rank - 1],
        platform::errors::InvalidArgument(
            "Eigh op is designed for square matrix, consequently"
            "inner-most 2 dimensions of Input(X) should be symmetric."
            "But received X's shape[-2] = %d and shape[-1] = %d.",
            input_dim[rank - 2], input_dim[rank - 1]));

    std::vector<int64_t> values_dim;

    for (auto i = 0; i < rank - 1; i++) {
      values_dim.emplace_back(input_dim[i]);
    }

    ctx->SetOutputDim("Eigenvalues", phi::make_ddim(values_dim));
    ctx->SetOutputDim("Eigenvectors", input_dim);
  }
};

class EignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), Hermitian or real symmetric matrices."
             "Its shape should be [*, N, N] where * is zero or"
             "more batch dimensions. The data type is float32 ,"
             "float64, complex64, complex128.");
    AddOutput("Eigenvalues",
              "(Tensor), The eigenvalues in ascending order."
              "The data type is float32 or float64.");
    AddOutput(
        "Eigenvectors",
        "(Tensor), The column is the normalized eigenvector "
        "corresponding to the eigenvalue. The data type is the same as ``X``.");
    AddAttr<std::string>(
        "UPLO",
        "(string, default 'L'), 'L' represents the lower triangular matrix,"
        "'U' represents the upper triangular matrix.")
        .SetDefault("L");
    AddComment(R"DOC(
Eigh Operator.

Computes the eigenvalues and eigenvectors of a complex Hermitian
 (conjugate symmetric) or a real symmetric matrix.

)DOC");
  }
};

class EighGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Eigenvalues"), "Input", "Eigenvalues",
                   "EighGrad");
    OP_INOUT_CHECK(ctx->HasInput("Eigenvectors"), "Input", "Eigenvectors",
                   "EighGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Eigenvalues")),
                   "Input", "Eigenvalues@GRAD", "EighGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Eigenvectors")),
                   "Input", "Eigenvectors@GRAD", "EighGrad");
    auto dims = ctx->GetInputDim("Eigenvectors");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(
            ctx, framework::GradVarName("Eigenvectors")),
        ctx.device_context());
  }
};

template <typename T>
class EighGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Eigenvalues", this->Output("Eigenvalues"));
    op->SetInput("Eigenvectors", this->Output("Eigenvectors"));
    op->SetInput(framework::GradVarName("Eigenvalues"),
                 this->OutputGrad("Eigenvalues"));
    op->SetInput(framework::GradVarName("Eigenvectors"),
                 this->OutputGrad("Eigenvectors"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(eigh, ops::EighOp, ops::EignOpMaker,
                  ops::EighGradOpMaker<paddle::framework::OpDesc>,
                  ops::EighGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(eigh_grad, ops::EighGradOp);

REGISTER_OP_CPU_KERNEL(
    eigh, ops::EighKernel<paddle::platform::CPUDeviceContext, float>,
    ops::EighKernel<paddle::platform::CPUDeviceContext, double>,
    ops::EighKernel<paddle::platform::CPUDeviceContext,
                    paddle::platform::complex<float>>,
    ops::EighKernel<paddle::platform::CPUDeviceContext,
                    paddle::platform::complex<double>>);

REGISTER_OP_CPU_KERNEL(
    eigh_grad, ops::EighGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext,
                        paddle::platform::complex<float>>,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext,
                        paddle::platform::complex<double>>);
