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

#include "paddle/fluid/operators/eig_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class EigOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Eig");
    OP_INOUT_CHECK(ctx->HasOutput("Eigenvalues"), "Output", "Eigenvalues",
                   "Eig");
    OP_INOUT_CHECK(ctx->HasOutput("Eigenvectors"), "Output", "Eigenvectors",
                   "Eig");

    auto x_dims = ctx->GetInputDim("X");
    int rank = x_dims.size();
    PADDLE_ENFORCE_GE(rank, 2, platform::errors::InvalidArgument(
                                   "Expects input tensor x to be not less than "
                                   "2 dimentions, but got dimention %d",
                                   rank));
    PADDLE_ENFORCE_EQ(x_dims[rank - 2], x_dims[rank - 1],
                      platform::errors::InvalidArgument(
                          "The input matrix must be a square matrix, "
                          "but receive a matrix with %d rows and %d colums",
                          x_dims[rank - 2], x_dims[rank - 1]));

    std::vector<int> batch_dims_vec{};
    for (int i = 0; i < rank - 1; ++i) {
      batch_dims_vec.emplace_back(x_dims[i]);
    }

    ctx->SetOutputDim("Eigenvectors", x_dims);
    ctx->SetOutputDim("Eigenvalues", phi::make_ddim(batch_dims_vec));
  }

 protected:
  // The output of eig is always complex-valued even for real-valued inputs
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    if (dtype != framework::proto::VarType::FP32 &&
        dtype != framework::proto::VarType::FP64 &&
        dtype != framework::proto::VarType::COMPLEX64 &&
        dtype != framework::proto::VarType::COMPLEX128) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "unsupported data type: %s!", dtype));
    }
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

class EigOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor), A complex-valued or real-valued tensor with shape (*, "
        "n, n). The accepted datatype is one of float32, float64, complex64 "
        "or complex128");
    AddOutput("Eigenvalues",
              "(Tensor), The output eigenvalues tensor with shape (*, n). The "
              "datatype is complex64 or complex128");
    AddOutput("Eigenvectors",
              "(Tensor), The output eigenvectors tensor with shape (*, n, n). "
              "The datatype is complex64 or complex128");

    AddComment(R"DOC(
        Eig Operator.

This API processes eigen decomposition for general square matrices.

)DOC");
  }
};

class EigGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Eigenvalues"), "Input", "Eigenvalues",
                   "EigGrad");
    OP_INOUT_CHECK(ctx->HasInput("Eigenvectors"), "Input", "Eigenvectors",
                   "EigGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Eigenvalues")),
                   "Input", "Eigenvalues@GRAD", "EigGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Eigenvectors")),
                   "Input", "Eigenvectors@GRAD", "EigGrad");

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
class EigGradOpMaker : public framework::SingleGradOpMaker<T> {
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
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

using complex64 = paddle::platform::complex<float>;
using complex128 = paddle::platform::complex<double>;

namespace ops = paddle::operators;
REGISTER_OPERATOR(eig, ops::EigOp, ops::EigOpMaker,
                  ops::EigGradOpMaker<paddle::framework::OpDesc>,
                  ops::EigGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(eig_grad, ops::EigGradOp);

REGISTER_OP_CPU_KERNEL(
    eig, ops::EigKernel<paddle::platform::CPUDeviceContext, float, complex64>,
    ops::EigKernel<paddle::platform::CPUDeviceContext, double, complex128>,
    ops::EigKernel<paddle::platform::CPUDeviceContext, complex64, complex64>,
    ops::EigKernel<paddle::platform::CPUDeviceContext, complex128, complex128>);

REGISTER_OP_CPU_KERNEL(
    eig_grad,
    ops::EigGradKernel<paddle::platform::CPUDeviceContext, float, complex64>,
    ops::EigGradKernel<paddle::platform::CPUDeviceContext, double, complex128>,
    ops::EigGradKernel<paddle::platform::CPUDeviceContext, complex64,
                       complex64>,
    ops::EigGradKernel<paddle::platform::CPUDeviceContext, complex128,
                       complex128>);
