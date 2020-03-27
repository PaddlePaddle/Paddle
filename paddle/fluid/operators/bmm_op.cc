/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/operators/bmm_op.h"
#include <vector>

namespace paddle {
namespace operators {

class BmmOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of BmmOp should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::NotFound("Input(Y) of BmmOp should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::NotFound("Output(Out) of BmmOp should not be null."));

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(x_dims.size(), 3,
                      "Input(X) of BmmOp must be 3-dimensional.");
    PADDLE_ENFORCE_EQ(y_dims.size(), 3,
                      "Input(X) of BmmOp must be 3-dimensional.");
    PADDLE_ENFORCE_EQ(x_dims[0], y_dims[0],
                      "Input(X) and Input(Y) must have the same dimension");
    PADDLE_ENFORCE_EQ(x_dims[2], y_dims[1],
                      "First matrix's width must be equal with second matrix's "
                      "height in BmmOp.");

    std::vector<int64_t> dim_out;
    dim_out.push_back(x_dims[0]);
    dim_out.push_back(x_dims[1]);
    dim_out.push_back(y_dims[2]);
    ctx->SetOutputDim("Out", framework::make_ddim(dim_out));
    ctx->ShareLoD("X", /*->*/ "OUt");
  }
};

class BmmOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of bmm op.");
    AddInput("Y", "(Tensor), The second input tensor of bmm op.");
    AddOutput("Out", "(Tensor), The output tensor of bmm op.");
    AddComment(R"DOC(
The Bmm operator is used to perform batched matrix multiplication
over the last two dimensions of the input tensors `X` and `Y` 
which are both 3-dimentionsal. 

Examples:
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, N]

      )DOC");
  }
};

class BmmGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of BmmOp should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::NotFound("Input(Y) of BmmOp should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Out")), true,
                      platform::errors::NotFound(
                          "Output(Out@GRAD) of BmmOp should not be null."));

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

template <typename T>
class BmmOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("bmm_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(bmm, ops::BmmOp, ops::BmmOpMaker,
                  ops::BmmOpGradMaker<paddle::framework::OpDesc>,
                  ops::BmmOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(bmm_grad, ops::BmmGradOp);
REGISTER_OP_CPU_KERNEL(
    bmm, ops::BmmKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BmmKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    bmm_grad, ops::BmmGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BmmGradKernel<paddle::platform::CPUDeviceContext, double>);
