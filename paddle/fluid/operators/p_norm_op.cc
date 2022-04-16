/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

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
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class PnormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) A tensor of rank >= axis.");
    AddAttr<float>("porder",
                   "(float, default 2) The porder is the p order vector norm "
                   "to calculate. Available for porder=0, inf, -inf and any "
                   "real number.")
        .SetDefault(2.0f);
    AddAttr<int>("axis",
                 "The axis on which to apply norm operation. If axis < 0, "
                 "the dimension to pnorm is rank(X) + axis. -1 is "
                 "the last dimension.")
        .SetDefault(-1);
    AddAttr<float>("epsilon",
                   "(float, default 1e-12) The epsilon value is used "
                   "to avoid division by zero.")
        .SetDefault(1.0e-12f);
    AddAttr<bool>(
        "keepdim",
        "(bool, default false) Whether to keep the dimensions as the input.")
        .SetDefault(false);

    AddAttr<bool>("asvector",
                  "(bool, default false) as vector norm when axis is None and "
                  "input is matrix, ")
        .SetDefault(false);
    AddOutput("Out", "(Tensor) Output result tensor of p-norm");
    AddComment(R"DOC(
Pnorm Operator.
Given a tensor X, compute Lp-norm of X.

When p = 0, defining $0^0 = 0$, the zero-norm of X is simply the number of non-zero elements of X.
$$
||X||_{0} = \lim_{p \rightarrow 0} \sum_i |x_i|^p
$$

When p = inf, the inf-norm of X is the maximum element of X.
$$
||X||_\infty = \max_i |x_i|
$$

When p = -inf, the negative-inf-norm of X is the minimum element of X.
$$
||X||_{-\infty} = \min_i |x_i|
$$

Otherwise, the p-norm of X follows the formula,
$$
||X||_{p} = (\sum_i |x_i|^p)^{1/p}
$$
where, $\sum_i $ is calculated along the `axis` dimension.

)DOC");
  }
};

class PnormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class PnormOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class PnormOpGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("p_norm_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("X", this->Input("X"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

DECLARE_INFER_SHAPE_FUNCTOR(p_norm, PNormInferShapeFunctor,
                            PD_INFER_META(phi::PNormInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(p_norm_grad, PNormGradInferShapeFunctor,
                            PD_INFER_META(phi::GeneralUnaryGradInferMeta));

REGISTER_OPERATOR(p_norm, ops::PnormOp, ops::PnormOpMaker,
                  ops::PnormOpGradOpMaker<paddle::framework::OpDesc>,
                  ops::PnormOpGradOpMaker<paddle::imperative::OpBase>,
                  PNormInferShapeFunctor);
REGISTER_OPERATOR(p_norm_grad, ops::PnormOpGrad, PNormGradInferShapeFunctor);

REGISTER_OP_VERSION(p_norm)
    .AddCheckpoint(
        R"ROC(
        Upgrade p_norm, add 1 attribute [asvector].
      )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "asvector",
            "Compute as vector when axis is None and input is matrix", false));
