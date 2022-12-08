/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/cvm_op.h"

#include <memory>

#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

class CVMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "CVM");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "CVM");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(
        x_dims.size(),
        2UL,
        platform::errors::InvalidArgument(
            "Input(X)'s rank should be 2, but got %d", x_dims.size()));

    if (ctx->Attrs().Get<bool>("use_cvm")) {
      ctx->SetOutputDim("Y", {x_dims[0], x_dims[1]});
    } else {
      ctx->SetOutputDim("Y", {x_dims[0], x_dims[1] - 2});
    }
    ctx->ShareLoD("X", /*->*/ "Y");
  }

 protected:
  // Explicitly set that the data type of computation kernel of
  // cvm
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class CVMGradientOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "CVMGradient");
    OP_INOUT_CHECK(ctx->HasInput("CVM"), "Input", "CVM", "CVMGradient");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")),
                   "Input",
                   framework::GradVarName("Y"),
                   "CVMGradient");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   framework::GradVarName("X"),
                   "CVMGradient");

    auto x_dims = ctx->GetInputDim("X");
    auto cvm_dims = ctx->GetInputDim("CVM");
    auto dy_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    PADDLE_ENFORCE_EQ(
        x_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "Expect Input(X)'s rank == 2, but got %d", x_dims.size()));
    PADDLE_ENFORCE_EQ(
        dy_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "Expect Input(X)'s rank == 2, but got %d", dy_dims.size()));
    PADDLE_ENFORCE_EQ(
        cvm_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "Expect Input(X)'s rank == 2, but got %d", cvm_dims.size()));

    PADDLE_ENFORCE_EQ(
        x_dims[0],
        dy_dims[0],
        platform::errors::InvalidArgument(
            "The 1st dimension of Input(X) and Input(Y@Grad) should "
            "be equal, X is %d, Y@Grad is %d",
            x_dims[0],
            dy_dims[0]));

    PADDLE_ENFORCE_EQ(
        cvm_dims[1],
        2,
        platform::errors::InvalidArgument(
            "When Attr(soft_label) == false, the 2nd dimension of "
            "Input(CVM) should be 2, but got %d cvm_dims[1]"));
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  // Explicitly set that the data type of computation kernel of
  // cvm
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Y")),
                                   ctx.device_context());
  }
};

class CVMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LodTensor, default LodTensor<float>), a 2-D tensor with shape "
             "[N x D],"
             " where N is the batch size and D is the emebdding dim. ");
    AddInput("CVM",
             "(Tensor),  a 2-D Tensor with shape [N x 2], where N is the batch "
             "size, 2 is show and click.");
    AddOutput("Y",
              "(LodTensor, default LodTensor<float>), a 2-D tensor with shape "
              "[N x K].");
    AddAttr<bool>("use_cvm", "bool, use cvm or not").SetDefault(true);
    AddComment(R"DOC(
CVM Operator.

      We assume that input X is a embedding vector with cvm_feature(show and click), which shape is [N * D] (D is 2(cvm_feature) + embedding dim, N is batch_size)
      if use_cvm is True, we will log(cvm_feature), and output shape is [N * D].
      if use_cvm is False, we will remove cvm_feature from input, and output shape is [N * (D - 2)].

)DOC");
  }
};

template <typename T>
class CVMGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("cvm_grad");
    op->SetInput("CVM", this->Input("CVM"));
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(CVMNoNeedBufferVarInferer, "CVM");
DECLARE_NO_NEED_BUFFER_VARS_INFERER(CVMGradNoNeedBufferVarInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(cvm,
                  ops::CVMOp,
                  ops::CVMOpMaker,
                  ops::CVMGradOpMaker<paddle::framework::OpDesc>,
                  ops::CVMGradOpMaker<paddle::imperative::OpBase>,
                  ops::CVMNoNeedBufferVarInferer);

REGISTER_OPERATOR(cvm_grad,
                  ops::CVMGradientOp,
                  ops::CVMGradNoNeedBufferVarInferer);

REGISTER_OP_CPU_KERNEL(cvm, ops::CVMOpKernel<float>, ops::CVMOpKernel<double>);

REGISTER_OP_CPU_KERNEL(cvm_grad,
                       ops::CVMGradOpKernel<float>,
                       ops::CVMGradOpKernel<double>);
