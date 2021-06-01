/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// .cc文件用于Op定义和OpMaker的定义及注册、CPU版本kernel的注册

#include "paddle/fluid/operators/add_equal_dim_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

// Op
class AddEqualDimOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "addequaldim");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "addequaldim");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "addequaldim");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto x_rank = x_dims.size();
    auto y_rank = y_dims.size();
    PADDLE_ENFORCE_EQ(
        x_rank, y_rank,
        platform::errors::InvalidArgument(
            "For mode 'element', rank of y must be equal to the rank of "
            "input(x). But recevied y's rank: %d, x's rank: %d .",
            y_rank, x_rank));

    int i = 0;
    for (i = 0; i < x_rank; i++) {
      PADDLE_ENFORCE_EQ(
          x_dims[i], y_dims[i],
          platform::errors::InvalidArgument(
              "For mode 'element', dim of y must be equal to dim of "
              "input(x). But recevied y's dim: %d, x's rank: %d .",
              x_dims[i], y_dims[i]));
    }

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

// OpMaker
class AddEqualDimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of AddEqualDim operator.");
    AddInput("Y", "(Tensor), The second input tensor of AddEqualDim operator.");
    AddOutput("Out", "(Tensor), The output tensor of AddEqualDim operator.");
    AddComment(R"DOC(
AddEqualDim Operator.
This operator is used to perform matrix addition for input $X$ and $Y$.
The equation is:
$$Out = X + Y$$
The input $X$ and $Y$ can carry the LoD (Level of Details) information,
or not. And the output shares the LoD information with input $X$.
)DOC");
  }
};

// GradOp
class AddEqualDimGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "addequaldim");
    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, dout_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, dout_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

// GradOpMaker
template <typename T>
class AddEqualDimGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("addequaldim_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(addequaldim, ops::AddEqualDimOp, ops::AddEqualDimOpMaker,
                  ops::AddEqualDimGradOpMaker<paddle::framework::OpDesc>,
                  ops::AddEqualDimGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(addequaldim_grad, ops::AddEqualDimGradOp);

REGISTER_OP_CPU_KERNEL(
    addequaldim,
    ops::AddEqualDimKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AddEqualDimKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    addequaldim_grad,
    ops::AddEqualDimGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AddEqualDimGradKernel<paddle::platform::CPUDeviceContext, double>);
