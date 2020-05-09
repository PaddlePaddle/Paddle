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

#include "paddle/fluid/operators/fill_zeros_like_op.h"

namespace paddle {
namespace operators {

class FillZerosLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "fill_zeros_like");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "fill_zeros_like");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class FillZerosLikeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of fill-zeros-like op.");
    AddOutput("Out", "The variable will be filled up with zeros.");
    ExtraMake();
    AddComment(R"DOC(
FillZerosLike Operator.

Fill up a variable with zeros.
The output will have the same size as the input.

)DOC");
  }

 protected:
  virtual void ExtraMake() {}
};

class FillZerosLikeOp2 : public FillZerosLikeOp {
 public:
  using FillZerosLikeOp::FillZerosLikeOp;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class FillZerosLikeOp2Maker : public FillZerosLikeOpMaker {
 protected:
  void ExtraMake() override {
    this->AddAttr<int>("dtype",
                       "(int, default 5(FP32)) "
                       "Output data type.")
        .SetDefault(framework::proto::VarType::FP32);
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(FillZerosLikeOp2NoNeedBufferVarsInference,
                                    "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fill_zeros_like, ops::FillZerosLikeOp,
                             ops::FillZerosLikeOpMaker);

REGISTER_OPERATOR(
    fill_zeros_like2, ops::FillZerosLikeOp2, ops::FillZerosLikeOp2Maker,
    ops::FillZerosLikeOp2NoNeedBufferVarsInference,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    fill_zeros_like,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, bool>);

REGISTER_OP_CPU_KERNEL(
    fill_zeros_like2,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, bool>);
