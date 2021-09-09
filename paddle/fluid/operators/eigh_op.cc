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
            "The inner-most 2 dimensions of Input(X) should be symmetric."
            "But received X's shape[-2] = %d and shape[-1] = %d.",
            input_dim[rank - 2], input_dim[rank - 1]));

    int64_t batch_size = 1;
    for (int i = 0; i < rank - 2; i++) {
      batch_size *= input_dim[i];
    }

    std::vector<int64_t> v_dim = {input_dim[1]};
    if (rank > 2) {
      v_dim = {batch_size, input_dim[1]};
    }

    ctx->SetOutputDim("Eigenvalues", framework::make_ddim(v_dim));
    ctx->SetOutputDim("Eigenvectors", input_dim);
  }
};

class EignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), Hermitian or real symmetric matrices."
             "Its shape should be [*, M, M] where "
             "* is zero or more batch dimensions");
    AddOutput("Eigenvalues", "(Tensor), The eigenvalues in ascending order.");
    AddOutput("Eigenvectors",
              "(Tensor), The column is the normalized eigenvector "
              "corresponding to the eigenvalue.");
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
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("Eigenvalues")),
                   "Input", "Eigenvalues@GRAD", "EighGrad");
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("Eigenvectors")),
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
    eigh, ops::EighKernel<paddle::platform::CPUDeviceContext, float, float>,
    ops::EighKernel<paddle::platform::CPUDeviceContext, double, double>,
    ops::EighKernel<paddle::platform::CPUDeviceContext, float,
                    paddle::platform::complex<float>>,
    ops::EighKernel<paddle::platform::CPUDeviceContext, double,
                    paddle::platform::complex<double>>);

REGISTER_OP_CPU_KERNEL(
    eigh_grad,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext, float, float>,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext, double, double>,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext, float,
                        paddle::platform::complex<float>>,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext, double,
                        paddle::platform::complex<double>>);
