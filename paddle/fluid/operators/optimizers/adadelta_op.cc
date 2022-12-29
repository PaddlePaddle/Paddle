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

class AdadeltaOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Param"), ctx.GetPlace());
  }
};

class AdadeltaOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("AvgSquaredGrad", "(Tensor) Input average of squared gradient");
    AddInput("AvgSquaredUpdate",
             "(Tensor) Input average of squared parameter updates");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("AvgSquaredGradOut",
              "(Tensor) Output average of squared gradient");
    AddOutput("AvgSquaredUpdateOut",
              "(Tensor) Output average of squared parameter updates");

    AddAttr<float>("rho",
                   "(float, default 0.95) Exponential decay rate "
                   "for squared gradients.")
        .SetDefault(0.95f);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-6) Constant for "
                   "numerical stability")
        .SetDefault(1.0e-6f);
    AddComment(R"DOC(
Adadelta Optimizer.

Adadelta optimizer is implemented as explained in:
https://arxiv.org/abs/1212.5701
Adadelta is a per-dimension adaptive learning rate method used
for gradient descent.

Adadelta updates are as follows:

$$
avg\_squared\_grad\_out = \rho * avg\_squared\_grad + (1 - \rho) * grad * grad \\
param\_update =  - \sqrt{\frac{avg\_squared\_update + \epsilon}{avg\_squared\_grad\_out + \epsilon}} * grad \\
avg\_squared\_update\_out = \rho * avg\_squared\_update + (1 - \rho) * {param\_update}^2 \\
param\_out = param + param\_update
$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(adadelta,
                            AdadeltaInferMetaFunctor,
                            PD_INFER_META(phi::AdadeltaInferMeta));
REGISTER_OPERATOR(
    adadelta,
    ops::AdadeltaOp,
    ops::AdadeltaOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    AdadeltaInferMetaFunctor);
