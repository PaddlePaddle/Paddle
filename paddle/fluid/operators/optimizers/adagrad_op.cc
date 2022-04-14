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

#include <cmath>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class AdagradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Param"), ctx.GetPlace());
  }
};

class AdagradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("Moment", "(Tensor) Second moment");
    AddInput("LearningRate", "(Tensor) Learning rate");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("MomentOut", "(Tensor) Output second moment");

    AddAttr<float>("epsilon",
                   "(float, default 1.0e-6) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-6f);
    AddComment(R"DOC(

Adaptive Gradient Algorithm (Adagrad).

The update is done as follows:

$$moment\_out = moment + grad * grad \\
param\_out = param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}
$$

The original paper(http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
does not have the epsilon attribute. It is added here in our implementation
as also proposed here: http://cs231n.github.io/neural-networks-3/#ada
for numerical stability to avoid the division by zero error.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(adagrad, AdagradInferShapeFunctor,
                            PD_INFER_META(phi::AdagradInferMeta));
REGISTER_OP_WITHOUT_GRADIENT(adagrad, ops::AdagradOp, ops::AdagradOpMaker,
                             AdagradInferShapeFunctor);
