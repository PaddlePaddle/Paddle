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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class MarginCrossEntropyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Logits"),
        ctx.device_context());
  }
};

class MarginCrossEntropyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("Logits",
             "(Tensor, default: Tensor<float>), The input tensor of unscaled "
             "log probabilities, whose dimension :attr:`axis` should be scaled "
             "by softmax.");
    AddInput(
        "Label",
        "(Tensor) The input tensor of groud truth label. Label is a "
        "Tensor<int64> in same shape with Input(Logits) except the shape in "
        "dimension :attr:`axis` as 1.");
    AddOutput(
        "Softmax",
        "(Tensor, default: Tensor<float>), A tensor in same shape with "
        "Input(Logits). "
        "The outputs value of softmax activation by given the input batch, "
        "which will be used in backward calculation.");
    AddOutput("Loss",
              "(Tensor, default: Tensor<float>), A tensor in same shape with "
              "Input(Logits) "
              "except the shape in dimension :attr:`axis` as 1. The cross "
              "entropy loss.");
    AddAttr<bool>("return_softmax",
                  "(bool default false) A flag to indicate "
                  "whether to return softmax.")
        .SetDefault(false);
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("rank", "(int default 0) rank id for MarginCrossEntropy.")
        .SetDefault(0);
    AddAttr<int>("nranks", "(int default 1) nranks id for MarginCrossEntropy.")
        .SetDefault(1);
    AddAttr<float>("margin1", "(float default 1.0) margin1 for MarginLoss.")
        .SetDefault(1.0);
    AddAttr<float>("margin2", "(float default 0.5) margin2 for MarginLoss.")
        .SetDefault(0.5);
    AddAttr<float>("margin3", "(float default 0.0) margin3 for MarginLoss.")
        .SetDefault(0.0);
    AddAttr<float>("scale", "(float default 64.0) scale for MarginLoss.")
        .SetDefault(64.0);
    AddComment(R"DOC(
MarginCrossEntropy Operator
.. math::

    L=-\frac{1}{N}\sum^N_{i=1}\log\frac{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}}{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}+\sum^n_{j=1,j\neq y_i} e^{scos\theta_{y_i}}}

where the :math: `\theta_{y_i}` is the angle between the feature :math: `x` and
the representation of class :math: `i`. The details of ArcFace loss
could be referred to https://arxiv.org/abs/1801.07698.

Note that the Op supports model parallel and single GPU. And Logits.shape[-1] can be different each rank.

)DOC");
  }
};

class MarginCrossEntropyOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Loss")),
                                   ctx.device_context());
  }
};

template <typename T>
class MarginCrossEntropyOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("margin_cross_entropy_grad");

    op->SetInput("Softmax", this->Output("Softmax"));
    op->SetInput("Logits", this->Input("Logits"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("Logits"), this->InputGrad("Logits"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(margin_cross_entropy,
                            MarginCrossEntropyInferShapeFunctor,
                            PD_INFER_META(phi::MarginCrossEntropyInferMeta));
REGISTER_OPERATOR(
    margin_cross_entropy,
    ops::MarginCrossEntropyOp,
    ops::MarginCrossEntropyOpMaker,
    ops::MarginCrossEntropyOpGradMaker<paddle::framework::OpDesc>,
    ops::MarginCrossEntropyOpGradMaker<paddle::imperative::OpBase>,
    MarginCrossEntropyInferShapeFunctor);
DECLARE_INFER_SHAPE_FUNCTOR(
    margin_cross_entropy_grad,
    MarginCrossEntropyGradInferShapeFunctor,
    PD_INFER_META(phi::MarginCrossEntropyGradInferMeta));
REGISTER_OPERATOR(margin_cross_entropy_grad,
                  ops::MarginCrossEntropyOpGrad,
                  MarginCrossEntropyGradInferShapeFunctor);
