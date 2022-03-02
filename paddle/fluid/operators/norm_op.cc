/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class NormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) A tensor of rank >= axis.");
    AddAttr<int>("axis",
                 "The axis on which to apply normalization. If axis < 0, "
                 "the dimension to normalization is rank(X) + axis. -1 is "
                 "the last dimension.");
    AddAttr<float>("epsilon",
                   "(float, default 1e-10) The epsilon value is used "
                   "to avoid division by zero.")
        .SetDefault(1.0e-10f);
    AddOutput("Norm",
              "(Tensor) A tensor saved the `sqrt(sum(x) + epsion)` will "
              "be used in backward kernel.")
        .AsIntermediate()
        .AsExtra();
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training.")
        .SetDefault(false);
    AddOutput("Out", "(Tensor) A tensor of the same shape as X.");
    AddComment(R"DOC(

Given a tensor, apply 2-normalization along the provided axis.

$$
y = \frac{x}{ \sqrt{\sum {x^2} + epsion }}
$$

where, $\sum {x^2}$ is calculated along the `axis` dimension.
        
)DOC");
  }
};

class NormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "NormOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "NormOp");
    auto xdim = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", xdim);

    if (ctx->Attrs().Get<bool>("is_test") == false) {
      int axis = ctx->Attrs().Get<int>("axis");
      if (axis < 0) axis = xdim.size() + axis;
      xdim[axis] = 1;
      ctx->SetOutputDim("Norm", xdim);
    }
  }
};

class NormOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "NormOpGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Input",
                   "X@GRAD", "NormOpGrad");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

template <typename T>
class NormOpGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("norm_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
#ifndef PADDLE_WITH_ASCEND_CL
    op->SetInput("Norm", this->Output("Norm"));
#else
    op->SetInput("Out", this->Output("Out"));
#endif
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(norm, ops::NormOp, ops::NormOpMaker,
                  ops::NormOpGradOpMaker<paddle::framework::OpDesc>,
                  ops::NormOpGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(norm_grad, ops::NormOpGrad);
