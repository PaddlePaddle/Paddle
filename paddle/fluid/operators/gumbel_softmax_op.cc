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
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {
class GumbelSoftmaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class GumbelSoftmaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) An N-D Tensor, N >= 1,"
             "The first N - 1 dimensions index into a batch of independent "
             "distributions "
             "and the last dimension represents a vector of probabilities for "
             "each class.");
    AddOutput("Out", "The sampled tensor with the same shape as X.");
    AddAttr<float>("temperature",
                   "(float, default 1.0) non-negative scalar temperature.")
        .SetDefault(1.0);
    AddAttr<bool>(
        "hard",
        "(bool, default false) "
        "if True, the returned samples will be discretized as one-hot vectors, "
        "but will be differentiated as if it is the soft sample in autograd.")
        .SetDefault(false);
    AddAttr<int>("axis",
                 "(int, default -1)"
                 "The dimension index of Input(x) to perform gumbel_softmax.")
        .SetDefault(-1);
    AddComment(R"DOC(
GumbelSoftmax Operator.

Samples from the Gumbel-Softmax distribution and optionally discretizes.

)DOC");
  }
};

class GumbelSoftmaxGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class GumbelSoftmaxGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("gumbel_softmax_grad");
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(gumbel_softmax, GumbelSoftmaxInferShapeFunctor,
                            PD_INFER_META(phi::GumbelSoftmaxInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(gumbel_softmax_grad,
                            GumbelSoftmaxGradInferShapeFunctor,
                            PD_INFER_META(phi::GumbelSoftmaxGradInferMeta));

REGISTER_OPERATOR(gumbel_softmax, ops::GumbelSoftmaxOp,
                  ops::GumbelSoftmaxOpMaker,
                  ops::GumbelSoftmaxGradOpMaker<paddle::framework::OpDesc>,
                  ops::GumbelSoftmaxGradOpMaker<paddle::imperative::OpBase>,
                  GumbelSoftmaxInferShapeFunctor);
REGISTER_OPERATOR(gumbel_softmax_grad, ops::GumbelSoftmaxGradOp,
                  GumbelSoftmaxGradInferShapeFunctor);
