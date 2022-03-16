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

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class HuberLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename AttrType>
class HuberLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input value of huber loss op."
             "X is a N-D tensor with shape [N_1, N_2,..., N_n].");
    AddInput("Y",
             "The target value of huber loss op."
             "Y is a N-D tensor with shape [N_1, N_2,..., N_n].");
    AddOutput("Residual",
              "Intermediate tensor to cache residual value between Y and X."
              "The shape is same as Input(X) and will be reused in backward.")
        .AsIntermediate();
    AddOutput("Out",
              "The output N-D tensor with shape [N_1, N_2,..., N_n] "
              "which represents the huber loss.");
    AddAttr<AttrType>("delta", "Hyper parameter in huber loss.");
    AddComment(R"DOC(
HuberLoss Operator.

Huber loss is a loss function used in robust regression. We define X as the
input value and Y as the target value. Huber loss can evaluate the fitness of
X to Y. Different from MSE loss, Huber loss is more robust for outliers. If the
shape of X and Y are [batch_size, 1]. The equation is:

$$
Out_{\delta}(X, Y)_i =
\begin{cases}
0.5 * (Y_i - X_i)^2,
\quad |Y_i - X_i| \leq \delta \\
\delta * (|Y_i - X_i| - 0.5 * \delta),
\quad otherwise
\end{cases}
$$

In the above equation, $Out_\delta(X, Y)_i$, $X_i$ and $Y_i$ represent the ith
element of Out, X and Y.

)DOC");
  }
};

class HuberLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "HuberLossGrad");

    auto residual_dims = ctx->GetInputDim("Residual");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, residual_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, residual_dims);
    }
  }
};

template <typename T>
class HuberLossGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("huber_loss_grad");
    op->SetInput("Residual", this->Output("Residual"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(huber_loss, HuberLossInferShapeFunctor,
                            PD_INFER_META(phi::HuberLossInferMeta));

REGISTER_OPERATOR(huber_loss, ops::HuberLossOp, ops::HuberLossOpMaker<float>,
                  ops::HuberLossGradOpMaker<paddle::framework::OpDesc>,
                  ops::HuberLossGradOpMaker<paddle::imperative::OpBase>,
                  HuberLossInferShapeFunctor);
REGISTER_OPERATOR(huber_loss_grad, ops::HuberLossGradOp);
