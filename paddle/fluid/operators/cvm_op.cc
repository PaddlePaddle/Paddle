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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class CVMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("CVM"), "Input(CVM) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto cvm_dims = ctx->GetInputDim("CVM");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2UL, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(cvm_dims.size(), 2UL, "Input(CVM)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(cvm_dims[1], 2UL,
                      "The 2nd dimension of "
                      "Input(CVM) should be 2.");

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
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   platform::CPUPlace());
  }
};

class CVMGradientOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("CVM"), "Input(CVM) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto cvm_dims = ctx->GetInputDim("CVM");
    auto dy_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(dy_dims.size(), 2, "Input(Y@Grad)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(cvm_dims.size(), 2, "Input(CVM)'s rank should be 2.");

    PADDLE_ENFORCE_EQ(x_dims[0], dy_dims[0],
                      "The 1st dimension of Input(X) and Input(Y@Grad) should "
                      "be equal.");

    PADDLE_ENFORCE_EQ(cvm_dims[1], 2,
                      "When Attr(soft_label) == false, the 2nd dimension of "
                      "Input(CVM) should be 2.");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  // Explicitly set that the data type of computation kernel of
  // cvm
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   platform::CPUPlace());
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

class CVMGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("cvm_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("CVM", Input("CVM"));
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(cvm, ops::CVMOp, ops::CVMOpMaker, ops::CVMGradOpDescMaker);

REGISTER_OPERATOR(cvm_grad, ops::CVMGradientOp);

REGISTER_OP_CPU_KERNEL(cvm, ops::CVMOpKernel<float>, ops::CVMOpKernel<double>);

REGISTER_OP_CPU_KERNEL(cvm_grad, ops::CVMGradOpKernel<float>,
                       ops::CVMGradOpKernel<double>);
