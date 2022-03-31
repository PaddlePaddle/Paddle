// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/angle_op.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

class AngleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "angle");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "angle");

    auto in_dims = ctx->GetInputDim("X");

    ctx->SetOutputDim("Out", in_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class AngleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of angle op.");
    AddOutput("Out", "(Tensor), The output tensor of angle op.");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
    AddAttr<bool>("use_cudnn",
                  "(bool, default false) Only used in cudnn kernel, need "
                  "install cudnn")
        .SetDefault(false)
        .AsExtra();
    AddComment(R"DOC(
Angle Operator.

This operator is used to perform elementwise angle for input $X$.
$$out = angle(x)$$

)DOC");
  }
};

class AngleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@Grad", "angle_grad");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "Out@Grad", "angle_grad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@Grad", "angle_grad");

    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), dout_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

template <typename T>
class AngleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("angle_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(angle, ops::AngleOp, ops::AngleOpMaker,
                  ops::AngleGradMaker<paddle::framework::OpDesc>,
                  ops::AngleGradMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    angle, ops::AngleKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AngleKernel<paddle::platform::CPUDeviceContext, double>,
    ops::AngleKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex<float>>,
    ops::AngleKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex<double>>);

REGISTER_OPERATOR(angle_grad, ops::AngleGradOp);

REGISTER_OP_CPU_KERNEL(
    angle_grad, ops::AngleGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AngleGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::AngleGradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex<float>>,
    ops::AngleGradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex<double>>);
