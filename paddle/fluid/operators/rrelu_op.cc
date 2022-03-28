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
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class RReluOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class RReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of RRelu op.");
    AddOutput("Out", "The output of RRelu op.");
    AddOutput("Noise", "The random sampled RRelu noise.")
        .AsIntermediate()
        .AsExtra();

    float default_lower = 1. / 8.;
    AddAttr<float>("lower", "Lower bound of the uniform distribution.")
        .SetDefault(default_lower)
        .AddCustomChecker([](const float& lower) {
          PADDLE_ENFORCE_EQ(lower >= 0.0f && lower < 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'RRelu_lower' must be between 0.0 and 1.0."));
        });
    float defalut_upper = 1. / 3.;
    AddAttr<float>("upper", "Upper bound of the uniform distribution.")
        .SetDefault(defalut_upper)
        .AddCustomChecker([](const float& upper) {
          PADDLE_ENFORCE_EQ(upper > 0.0f && upper <= 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'RRelu_upper' must be between 0.0 and 1.0."));
        });

    AddComment(R"DOC(
RRelu Operator.

Applies the randomized leaky rectified liner unit function, element-wise,
as described in the paper:

`Empirical Evaluation of Rectified Activations in Convolutional Network`_.

The function is defined as:

.. math::
    \text{RReLU}(x) =
    \begin{cases}
        x & \text{if } x \geq 0 \\
        ax & \text{ otherwise }
    \end{cases}

where :math:`a` is randomly sampled from uniform distribution
:math:`\mathcal{U}(\text{lower}, \text{upper})`.

 See: https://arxiv.org/pdf/1505.00853.pdf

)DOC");
  }
};

class RReluOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Noise"), "Input", "Noise", "rrelu_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "rrelu_grad");

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class RReluGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("rrelu_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Noise", this->Output("Noise"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(rrelu, RReluInferShapeFunctor,
                            PD_INFER_META(phi::RReluInferMeta));

REGISTER_OPERATOR(rrelu, ops::RReluOp, ops::RReluOpMaker,
                  ops::RReluGradOpMaker<paddle::framework::OpDesc>,
                  ops::RReluGradOpMaker<paddle::imperative::OpBase>,
                  RReluInferShapeFunctor);
REGISTER_OPERATOR(rrelu_grad, ops::RReluOpGrad);
