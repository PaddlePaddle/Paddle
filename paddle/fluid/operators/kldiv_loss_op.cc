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

#include <memory>
#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class KLDivLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class KLDivLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of KL divergence loss operator. "
             "This is a tensor with shape of [N, *], where N is the "
             "batch size, * means any number of additional dimensions. "
             "The data type is float32 or flaot64");
    AddInput("Target",
             "The  tensor of KL divergence loss operator. "
             "This is a tensor with shape of Input(X). "
             "The data type is same as Input(X)");
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
         between Input(X) and Input(Target). Notes that Input(X) is the
         log-probability and Input(Target) is the probability.

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
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "KLDivLossGrad");
    OP_INOUT_CHECK(ctx->HasInput("Target"), "Input", "Target", "KLDivLossGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Loss")),
                   "Input",
                   "Loss@GRAD",
                   "KLDivLossGrad");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Loss")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class KLDivLossOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("kldiv_loss_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Target", this->Input("Target"));
    op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(KLDivLossGradNoNeedBufferVarInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(kldiv_loss,
                            KLDivInferShapeFunctor,
                            PD_INFER_META(phi::KLDivInferMeta));

REGISTER_OPERATOR(kldiv_loss,
                  ops::KLDivLossOp,
                  ops::KLDivLossOpMaker,
                  ops::KLDivLossOpGradMaker<paddle::framework::OpDesc>,
                  ops::KLDivLossOpGradMaker<paddle::imperative::OpBase>,
                  KLDivInferShapeFunctor);
REGISTER_OPERATOR(kldiv_loss_grad,
                  ops::KLDivLossOpGrad,
                  ops::KLDivLossGradNoNeedBufferVarInferer);
