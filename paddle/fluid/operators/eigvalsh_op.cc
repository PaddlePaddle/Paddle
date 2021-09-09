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
    OP_INOUT_CHECK(ctx->HasOutput("OutValue"), "Output", "OutValue",
                   "Eigvalsh");
    OP_INOUT_CHECK(ctx->HasOutput("OutVector"), "Output", "OutVector",
                   "Eigvalsh");

    auto input_dim = ctx->GetInputDim("X");
    auto rank = input_dim.size();
    int64_t batch_size = 1;
    for (int i = 0; i < rank - 2; i++) {
      batch_size *= input_dim[i];
    }
    std::vector<int64_t> v_dim = {input_dim[1]};
    if (batch_size > 1) {
      v_dim = {batch_size, input_dim[1]};
    }

    PADDLE_ENFORCE_GE(rank, 2,
                      platform::errors::InvalidArgument(
                          "The Input(X) should have at least 2 dimensions. But "
                          "received a %d dimension tensor.",
                          rank));
    PADDLE_ENFORCE_EQ(
        input_dim[rank - 2], input_dim[rank - 1],
        platform::errors::InvalidArgument(
            "The inner-most 2 dimensions of Input(X) all should be symmetric "
            "Input matrices and have the same size. But received "
            "X's shape[-2] = %d and shape[-1] = %d.",
            input_dim[rank - 2], input_dim[rank - 1]));

    ctx->SetOutputDim("OutValue", framework::make_ddim(v_dim));
    ctx->SetOutputDim("OutVector", input_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.GetPlace());
  }
};

class EigvalshOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor), Hermitian or real symmetric matrices whose eigenvalues "
        "are to be computed. Its shape should be [*, M, M] where "
        "* "
        "is zero or more batch dimensions,and matrices on the inner-most 2 "
        "dimensions"
        "all should be symmetric");
    AddOutput("OutValue",
              "(Tensor), The eigenvalues in ascending order, "
              "each repeated according to its multiplicity.");
    AddOutput("OutVector",
              "(Tensor), The column v[:, i] is the normalized eigenvector "
              "corresponding to the,"
              "eigenvalue w[i]. Will return a matrix object if a is a matrix "
              "object. Used when backward.")
        .AsIntermediate();
    AddAttr<std::string>("UPLO",
                         "(string, default L), the lower triangular part of a "
                         "(‘L’, default) or the upper "
                         "triangular part (‘U’)")
        .SetDefault("L");
    AddComment(R"DOC(
Eigvalsh Operator.

Return the eigenvalues of a complex Hermitian
 (conjugate symmetric) or a real symmetric matrix.
)DOC");
  }
};

class EigvalshGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("OutValue"), "Input", "OutValue",
                   "EigvalshGrad");
    OP_INOUT_CHECK(ctx->HasInput("OutVector"), "Input", "OutVector",
                   "EigvalshGrad");
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("OutValue")), "Input",
                   "OutValue@GRAD", "EigvalshGrad");
    auto dims = ctx->GetInputDim("OutVector");
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
            ctx, framework::GradVarName("OutVector")),
        ctx.GetPlace());
  }
};

template <typename T>
class EigvalshGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("OutValue", this->Output("OutValue"));
    op->SetInput("OutVector", this->Output("OutVector"));
    op->SetInput(framework::GradVarName("OutValue"),
                 this->OutputGrad("OutValue"));
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
    eigvals_grad,
    ops::EigvalshGradKernel<paddle::platform::CPUDeviceContext, float, float>,
    ops::EigvalshGradKernel<paddle::platform::CPUDeviceContext, double, double>,
    ops::EigvalshGradKernel<paddle::platform::CPUDeviceContext, float,
                            paddle::platform::complex<float>>,
    ops::EigvalshGradKernel<paddle::platform::CPUDeviceContext, double,
                            paddle::platform::complex<double>>);
