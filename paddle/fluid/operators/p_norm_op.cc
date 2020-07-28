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

#include "paddle/fluid/operators/p_norm_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class PnormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) A tensor of rank >= axis.");
    AddAttr<float>("porder",
                   "The porder is the p order vector norm to calculate.")
        .SetDefault(2.0f);
    AddAttr<int>("axis",
                 "The axis on which to apply normalization. If axis < 0, "
                 "the dimension to pnorm is rank(X) + axis. -1 is "
                 "the last dimension.")
        .SetDefault(-1);
    AddAttr<float>("epsilon",
                   "(float, default 1e-10) The epsilon value is used "
                   "to avoid division by zero.")
        .SetDefault(1.0e-12f);
    AddAttr<bool>(
        "keepdim",
        "(bool, default false) Whether to keep the dimensions as the input")
        .SetDefault(false);
    AddOutput(
        "Out",
        "(Tensor) Output tensor for the `(sum(x.pow(p)) + epsion).pow(1/p)`");
    AddComment(R"DOC(

Given a tensor, apply 2-normalization along the provided axis.

$$
pnorm = \(\sum_i {abs\(x_i\)^p}  \)^{1/p}
$$

where, $\sum_i{x_i^p}$ is calculated along the `axis` dimension.
        
)DOC");
  }
};

class PnormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "p_norm");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "p_norm");
    auto porder = ctx->Attrs().Get<float>("porder");
    PADDLE_ENFORCE_NE(porder, INFINITY,
                      platform::errors::Unimplemented(
                          "The input porder of p_norm is not support for "
                          "porder == 0, INFINITY, -INFINITY now."));
    PADDLE_ENFORCE_NE(porder, -INFINITY,
                      platform::errors::Unimplemented(
                          "The input porder of p_norm is not support for "
                          "porder == 0, INFINITY, -INFINITY now."));
    PADDLE_ENFORCE_GT(porder, 0.0f,
                      platform::errors::InvalidArgument(
                          "The input porder of p_norm is not support for "
                          "porder <= 0, But received porder=%f.",
                          porder));
    auto xdim = ctx->GetInputDim("X");
    int axis = ctx->Attrs().Get<int>("axis");
    bool keepdim = ctx->Attrs().Get<bool>("keepdim");
    if (axis < 0) axis = xdim.size() + axis;
    std::vector<int> reduce_dims;
    for (int i = 0; i < xdim.size(); ++i) {
      if (i != axis) reduce_dims.emplace_back(xdim[i]);
    }
    xdim[axis] = 1;
    if (keepdim) {
      ctx->SetOutputDim("Out", xdim);
    } else {
      ctx->SetOutputDim("Out", framework::make_ddim(reduce_dims));
    }
  }
};

class PnormOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "p_norm");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "p_norm");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "p_norm");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@GRAD", "p_norm");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
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

REGISTER_OPERATOR(p_norm, ops::PnormOp, ops::PnormOpMaker,
                  ops::PnormOpGradOpMaker<paddle::framework::OpDesc>,
                  ops::PnormOpGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(p_norm_grad, ops::PnormOpGrad);
REGISTER_OP_CPU_KERNEL(p_norm, ops::PnormKernel<CPU, float>,
                       ops::PnormKernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(p_norm_grad, ops::PnormGradKernel<CPU, float>,
                       ops::PnormGradKernel<CPU, double>);
