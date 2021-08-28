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

#include "paddle/fluid/operators/eigh_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class EighOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Eigh");
    OP_INOUT_CHECK(ctx->HasOutput("OutValue"), "Output", "OutValue", "Eigh");
    OP_INOUT_CHECK(ctx->HasOutput("OutVector"), "Output", "OutVector", "Eigh");

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
            "positive-definite matrices and have the same size. But received "
            "X's shape[-2] = %d and shape[-1] = %d.",
            input_dim[rank - 2], input_dim[rank - 1]));

    ctx->SetOutputDim("OutValue", framework::make_ddim(v_dim));
    ctx->SetOutputDim("OutVector", input_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class EignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Hermitian or real symmetric matrices whose eigenvalues and "
             "eigenvectors are to be computed ");
    AddOutput("OutValue",
              "The eigenvalues in ascending order, "
              "each repeated according to its multiplicity.");
    AddOutput(
        "OutVector",
        "The column v[:, i] is the normalized eigenvector corresponding to the,"
        "eigenvalue w[i]. Will return a matrix object if a is a matrix "
        "object.");
    AddAttr<std::string>(
        "UPLO",
        "the lower triangular part of a (‘L’, default) or the upper "
        "triangular part (‘U’)")
        .SetDefault("L");
    AddComment(R"DOC(
Eigh Operator.

Return the eigenvalues and eigenvectors of a complex Hermitian
 (conjugate symmetric) or a real symmetric matrix.

Returns two objects, a 1-D array containing the eigenvalues of a,
 and a 2-D square array or matrix (depending on the input type) 
of the corresponding eigenvectors (in columns).
)DOC");
  }
};

class EighGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // std::cout << "InferShape>>>>>>> " << std::endl;
    OP_INOUT_CHECK(ctx->HasInput("OutValue"), "Input", "OutValue", "EighGrad");
    OP_INOUT_CHECK(ctx->HasInput("OutVector"), "Input", "OutVector",
                   "EighGrad");
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("OutValue")), "Input",
                   "OutValue@GRAD", "EighGrad");
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("OutVector")), "Input",
                   "OutVector@GRAD", "EighGrad");
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
        ctx.device_context());
  }
};

template <typename T>
class EighGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    // std::cout << "this->ForwardOpType(): " << this->ForwardOpType() <<
    // std::endl;
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("OutValue", this->Output("OutValue"));
    op->SetInput("OutVector", this->Output("OutVector"));
    op->SetInput(framework::GradVarName("OutValue"),
                 this->OutputGrad("OutValue"));
    op->SetInput(framework::GradVarName("OutVector"),
                 this->OutputGrad("OutVector"));
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
    ops::EighKernel<paddle::platform::CPUDeviceContext, double, double>);

REGISTER_OP_CPU_KERNEL(
    eigh_grad,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext, float, float>,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext, double, double>,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext, float,
                        paddle::platform::complex<float>>,
    ops::EighGradKernel<paddle::platform::CPUDeviceContext, double,
                        paddle::platform::complex<double>>);
