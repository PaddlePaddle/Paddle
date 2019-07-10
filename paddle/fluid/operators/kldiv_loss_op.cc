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

#include "paddle/fluid/operators/kldiv_loss_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class KLDivLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of KLDivLossOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Target"),
                   "Input(Target) of KLDivLossOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Loss"),
                   "Output(Loss) of KLDivLossOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");
    auto dim_target = ctx->GetInputDim("Target");
    PADDLE_ENFORCE_EQ(dim_x.size(), dim_target.size(),
                      "Input(X) rank and Input(Target) rank should be same.");
    for (int i = 0; i < dim_x.size(); i++) {
      if (ctx->IsRuntime() || (dim_x[i] > 0 && dim_target[i] > 0)) {
        PADDLE_ENFORCE_EQ(dim_x[i], dim_target[i],
                          "Input(X) and Input(Target) should in same shape.");
      }
    }

    auto reduction = ctx->Attrs().Get<std::string>("reduction");

    PADDLE_ENFORCE(
        "mean" == reduction || "sum" == reduction || "batchmean" == reduction ||
            "none" == reduction,
        "Attr(reduction) can only be 'none'|'batchmean'|'sum'|'mean'.");

    if ("none" == reduction) {
      ctx->SetOutputDim("Loss", dim_x);
    } else {
      ctx->SetOutputDim("Loss", {1});
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class KLDivLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of KL divergence loss operator. "
             "This is a tensor with shape of [N, *], where N is the "
             "batch size, * means any number of additional dimensions.");
    AddInput("Target",
             "The  tensor of KL divergence loss operator. "
             "This is a tensor with shape of Input(X).");
    AddOutput(
        "Loss",
        "The output KL divergence loss tensor. if Attr(reduction) is "
        "'none', this tensor should be in same shape of of Input(X), else "
        "this tensor should be in shape of [1].");

    AddAttr<std::string>(
        "reduction",
        "The reduction type to apply to the output, available types "
        "are 'none' | 'batchmean' | 'mean' | 'sum', 'none' for no "
        "reduction, 'batchmean' for the sum of output divided by "
        "batch size, 'mean' for the average value of all output, "
        "'sum' for the sum of the output.")
        .SetDefault("mean");

    AddComment(R"DOC(
         This operator calculates the Kullback-Leibler divergence loss
         between Input(X) and Input(Target).

         KL divergence loss is calculated as follows:

         $$l(x, y) = y * (\log(y) - x)$$

         While :math:`x` is Input(X) and :math:`y` is Input(Target).

         While :attr:`reduction` is :attr:`none`, output loss is in
         the same shape as Input(X), loss in each point is calculated 
         seperately and no reduction is applied.
         
         While :attr:`reduction` is :attr:`mean`, output loss is in
         shape of [1] and loss value is the mean value of all losses.
         
         While :attr:`reduction` is :attr:`sum`, output loss is in
         shape of [1] and loss value is the sum value of all losses.
         
         While :attr:`reduction` is :attr:`batchmean`, output loss is 
         in shape of [1] and loss value is the sum value of all losses
         divided by batch size.
         
         )DOC");
  }
};

class KLDivLossOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Target"), "Input(Target) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Loss")),
                   "Input(Loss@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class KLDivLossOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType("kldiv_loss_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Target", Input("Target"));
    op->SetInput(framework::GradVarName("Loss"), OutputGrad("Loss"));

    op->SetAttrMap(Attrs());

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    return std::unique_ptr<framework::OpDesc>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(kldiv_loss, ops::KLDivLossOp, ops::KLDivLossOpMaker,
                  ops::KLDivLossOpGradMaker);
REGISTER_OPERATOR(kldiv_loss_grad, ops::KLDivLossOpGrad);
REGISTER_OP_CPU_KERNEL(
    kldiv_loss, ops::KLDivLossKernel<paddle::platform::CPUDeviceContext, float>,
    ops::KLDivLossKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    kldiv_loss_grad,
    ops::KLDivLossGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::KLDivLossGradKernel<paddle::platform::CPUDeviceContext, double>);
