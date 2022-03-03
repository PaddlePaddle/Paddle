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

#include "paddle/fluid/operators/eigvalsh_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class EigvalshOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Eigvalsh");
    OP_INOUT_CHECK(ctx->HasOutput("Eigenvalues"), "Output", "Eigenvalues",
                   "Eigvalsh");

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
            "Eigvalsh op is designed for square matrix, consequently"
            "inner-most 2 dimensions of Input(X) should be symmetric."
            "But received X's shape[-2] = %d and shape[-1] = %d.",
            input_dim[rank - 2], input_dim[rank - 1]));

    std::vector<int64_t> values_dim;

    for (auto i = 0; i < rank - 1; i++) {
      values_dim.emplace_back(input_dim[i]);
    }

    ctx->SetOutputDim("Eigenvalues", phi::make_ddim(values_dim));

    if (ctx->HasOutput("Eigenvectors")) {
      ctx->SetOutputDim("Eigenvectors", input_dim);
    }
  }
};

class EigvalshOpMaker : public framework::OpProtoAndCheckerMaker {
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
        "corresponding to the eigenvalue. The data type is the same as ``X``."
        "Eigenvectors are required to calculate gradient when backward.");
    AddAttr<std::string>(
        "UPLO",
        "(string, default 'L'), 'L' represents the lower triangular matrix,"
        "'U' represents the upper triangular matrix.")
        .SetDefault("L");
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training.")
        .SetDefault(false);
    AddComment(R"DOC(
Eigvalsh Operator.

Computes the eigenvalues of a complex Hermitian
 (conjugate symmetric) or a real symmetric matrix.

)DOC");
  }
};

class EigvalshGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Eigenvectors"), "Input", "Eigenvectors",
                   "EigvalshGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Eigenvalues")),
                   "Input", "Eigenvalues@GRAD", "EigvalshGrad");
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
        OperatorWithKernel::IndicateVarDataType(ctx, "Eigenvectors"),
        ctx.device_context());
  }
};

template <typename T>
class EigvalshGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Eigenvectors", this->Output("Eigenvectors"));
    op->SetInput(framework::GradVarName("Eigenvalues"),
                 this->OutputGrad("Eigenvalues"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(eigvalsh, ops::EigvalshOp, ops::EigvalshOpMaker,
                  ops::EigvalshGradOpMaker<paddle::framework::OpDesc>,
                  ops::EigvalshGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(eigvalsh_grad, ops::EigvalshGradOp);

REGISTER_OP_CPU_KERNEL(
    eigvalsh,
    ops::EigvalshKernel<paddle::platform::CPUDeviceContext, float, float>,
    ops::EigvalshKernel<paddle::platform::CPUDeviceContext, double, double>,
    ops::EigvalshKernel<paddle::platform::CPUDeviceContext, float,
                        paddle::platform::complex<float>>,
    ops::EigvalshKernel<paddle::platform::CPUDeviceContext, double,
                        paddle::platform::complex<double>>);

REGISTER_OP_CPU_KERNEL(
    eigvalsh_grad,
    ops::EigvalshGradKernel<paddle::platform::CPUDeviceContext, float, float>,
    ops::EigvalshGradKernel<paddle::platform::CPUDeviceContext, double, double>,
    ops::EigvalshGradKernel<paddle::platform::CPUDeviceContext, float,
                            paddle::platform::complex<float>>,
    ops::EigvalshGradKernel<paddle::platform::CPUDeviceContext, double,
                            paddle::platform::complex<double>>);
