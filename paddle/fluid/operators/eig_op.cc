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
    PADDLE_ENFORCE_EQ(true, ctx->HasInput("X"),
                      platform::errors::PreconditionNotMet(
                          "Input(X) of EigOp should not be null."));
    PADDLE_ENFORCE_EQ(true, ctx->HasOutput("OutValues"),
                      platform::errors::PreconditionNotMet(
                          "Output(OutValues) of EigOp should not be null."));
    PADDLE_ENFORCE_EQ(true, ctx->HasOutput("OutVectors"),
                      platform::errors::PreconditionNotMet(
                          "Output(OutVectors) of EigOp should not be null."));

    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Eig");
    OP_INOUT_CHECK(ctx->HasOutput("OutValues"), "Output", "OutValues", "Eig");
    OP_INOUT_CHECK(ctx->HasOutput("OutVectors"), "Output", "OutVectors", "Eig");

    auto x_dims = ctx->GetInputDim("X");
    int num_dims = x_dims.size();
    // int order = x_dims[num_dims - 1];

    std::vector<int> batch_dims_vec{};
    for (int i = 0; i < num_dims - 1; ++i) {
      batch_dims_vec.emplace_back(x_dims[i]);
    }

    PADDLE_ENFORCE_GE(num_dims, 2, platform::errors::OutOfRange(
                                       "expects the Tensor to be not less than "
                                       "2 dimentions, but got dimention is %d",
                                       num_dims));
    PADDLE_ENFORCE_EQ(
        x_dims[num_dims - 2], x_dims[num_dims - 1],
        platform::errors::InvalidArgument(
            "ShapeError: The input matrix must be a square matrix, "
            "but receive a matrix with %d rows and %d colums",
            x_dims[num_dims - 2], x_dims[num_dims - 1]));

    // 指定输出需要占用的空间个数
    ctx->SetOutputDim("OutVectors", x_dims);
    ctx->SetOutputDim("OutValues", framework::make_ddim(batch_dims_vec));
  }

 protected:
  // 输出的dtype是空，所以这里需要设定输出的dtype
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
    // 链接Python和c++的输入输出对象名
    AddInput("X", "The input matrix as a Tensor of Eig op.");
    AddOutput("OutValues", "The eigen-values calculated by this op");
    AddOutput("OutVectors", "The eigen-vectors calculated by this op");

    AddComment(R"DOC(
        Eig Operator.

        Eig refers to eigenvalues Decomposition. ...
        ...
        ...
        )DOC");
  }
};

class EigGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("OutValues"), "Input", "OutValues", "EigGrad");
    OP_INOUT_CHECK(ctx->HasInput("OutVectors"), "Input", "OutVectors",
                   "EigGrad");
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("OutValues")), "Input",
                   "OutValues@GRAD", "EigGrad");
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("OutVectors")),
                   "Input", "OutVectors@GRAD", "EigGrad");

    auto dims = ctx->GetInputDim("OutVectors");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateOrPromoteVarDataTypes(
        ctx, framework::GradVarName("OutValues"),
        framework::GradVarName("OutVectors"));
    return framework::OpKernelType(input_data_type, ctx.device_context());
  }
};

template <typename T>
class EigGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("OutValues", this->Output("OutValues"));
    op->SetInput("OutVectors", this->Output("OutVectors"));
    op->SetInput(framework::GradVarName("OutValues"),
                 this->OutputGrad("OutValues"));
    op->SetInput(framework::GradVarName("OutVectors"),
                 this->OutputGrad("OutVectors"));
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
    eig,
    ops::EigKernel<paddle::platform::CPUDeviceContext, float, complex64, float>,
    ops::EigKernel<paddle::platform::CPUDeviceContext, double, complex128,
                   double>,
    ops::EigKernel<paddle::platform::CPUDeviceContext, complex64, complex64,
                   float>,
    ops::EigKernel<paddle::platform::CPUDeviceContext, complex128, complex128,
                   double>);

REGISTER_OP_CPU_KERNEL(
    eig_grad,
    ops::EigGradKernel<paddle::platform::CPUDeviceContext, float, complex64>,
    ops::EigGradKernel<paddle::platform::CPUDeviceContext, double, complex128>,
    ops::EigGradKernel<paddle::platform::CPUDeviceContext, complex64,
                       complex64>,
    ops::EigGradKernel<paddle::platform::CPUDeviceContext, complex128,
                       complex128>);