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

#include "paddle/fluid/operators/clip_op.h"
#include <memory>

namespace paddle {
namespace operators {

class ClipOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "clip");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "clip");
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

template <typename AttrType>
class ClipOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Tensor, the input of clip op, data type should be float32 or "
             "float64.");
    AddInput("Min",
             "Tensor, the lower bound, data type should be float32 "
             "or float64.")
        .AsDispensable();
    AddInput("Max",
             "Tensor, the upper bound, data type should be float32 "
             "or float64.")
        .AsDispensable();
    AddOutput(
        "Out",
        "Tensor, the clipped tensor, with the same shape and data type as "
        "input(x)");
    AddAttr<AttrType>("min", "float number, the minimum value to clip by.");
    AddAttr<AttrType>("max", "float number, the maximum value to clip by.");
    AddComment(R"DOC(
Clip Operator.

The clip operator limits the value of given input within an interval [min, max],
just as the following equation,

$$
Out = \MIN(\MAX(x, min), max)
$$

)DOC");
  }
};

class ClipOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "clip_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "clip_grad");
    auto x_dims = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
  }
};

template <typename T>
class ClipGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("clip_grad");
    op->SetInput("X", this->Input("X"));
    if (this->HasInput("Min")) {
      op->SetInput("Min", this->Input("Min"));
    }
    if (this->HasInput("Max")) {
      op->SetInput("Max", this->Input("Max"));
    }
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(ClipInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(ClipGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(clip, ops::ClipOp, ops::ClipOpMaker<float>,
                  ops::ClipGradOpMaker<paddle::framework::OpDesc>,
                  ops::ClipGradOpMaker<paddle::imperative::OpBase>,
                  ops::ClipInplaceInferer);
REGISTER_OPERATOR(clip_grad, ops::ClipOpGrad, ops::ClipGradInplaceInferer);
REGISTER_OP_CPU_KERNEL(
    clip, ops::ClipKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ClipKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    clip_grad, ops::ClipGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ClipGradKernel<paddle::platform::CPUDeviceContext, double>);
