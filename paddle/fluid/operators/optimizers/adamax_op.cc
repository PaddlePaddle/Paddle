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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class AdamaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Param"), ctx.GetPlace());
  }
};

class AdamaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("LearningRate", "(Tensor) Learning rate");
    AddInput("Moment", "(Tensor) First moment");
    AddInput("InfNorm",
             "(Tensor) "
             "Input exponentially weighted infinity norm");
    AddInput("Beta1Pow", "(Tensor) Input beta1 power accumulator");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("MomentOut", "(Tensor) Output first moment");
    AddOutput("InfNormOut",
              "(Tensor) "
              "Output exponentially weighted infinity norm");

    AddAttr<float>("beta1",
                   "(float, default 0.9) "
                   "Exponential decay rate for the "
                   "1st moment estimates.")
        .SetDefault(0.9f);
    AddAttr<float>("beta2",
                   "(float, default 0.999) "
                   "exponential decay rate for the weighted "
                   "infinity norm estimates.")
        .SetDefault(0.999f);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-8) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-8f);
    AddComment(R"DOC(
Adamax Optimizer.

We implement the Adamax optimizer from Section 7 of the Adam
paper: https://arxiv.org/abs/1412.6980. Adamax is a variant of the
Adam algorithm based on the infinity norm.

Adamax updates:

$$
moment\_out = \beta_1 * moment + (1 - \beta_1) * grad \\
inf\_norm\_out = max(\beta_2 * inf\_norm + \epsilon, |grad|) \\
learning\_rate = \frac{learning\_rate}{1 - \beta_{1\_pow}} \\
param\_out = param - learning\_rate * \frac{moment\_out}{inf\_norm\_out}
$$

The original paper does not have an epsilon attribute.
However, it is added here for numerical stability to prevent the
division by 0 error.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(adamax, AdamaxInferMetaFunctor,
                            PD_INFER_META(phi::AdamaxInferMeta));

REGISTER_OPERATOR(
    adamax, ops::AdamaxOp, ops::AdamaxOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    AdamaxInferMetaFunctor);
