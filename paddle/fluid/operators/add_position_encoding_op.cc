/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/add_position_encoding_op.h"

#include <memory>

namespace paddle {
namespace operators {

class AddPositionEncodingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "AddPositionEncoding");
    OP_INOUT_CHECK(
        ctx->HasOutput("Out"), "Output", "Out", "AddPositionEncoding");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        platform::CPUPlace());
  }
};

class AddPositionEncodingOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
      ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   platform::CPUPlace());
  }
};

class AddPositionEncodingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of AddPositionEncoding operator");
    AddOutput("Out", "Output of AddPositionEncoding operator");
    AddAttr<float>("alpha", "The scale of Original Embedding.")
        .SetDefault(1.0f)
        .AddCustomChecker([](const float& alpha) {
          PADDLE_ENFORCE_GE(
              alpha,
              0.0f,
              platform::errors::InvalidArgument(
                  "Attribute 'alpha' must be greater than or equal to 0.0."));
        });
    AddAttr<float>("beta", "The scale of Position Embedding.")
        .SetDefault(1.0f)
        .AddCustomChecker([](const float& beta) {
          PADDLE_ENFORCE_GE(
              beta,
              0.0f,
              platform::errors::InvalidArgument(
                  "Attribute 'beta' must be greater than or equal to 0.0."));
        });
    AddComment(R"DOC(
    Add Position Encoding Operator.

    The add position encoding calculates the output based on the input, alpha, beta.
    The size of each dimension of the parameters checked in the infer-shape.
  )DOC");
  }
};

template <typename T>
class AddPositionEncodingGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("add_position_encoding_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;

REGISTER_OPERATOR(
    add_position_encoding,
    ops::AddPositionEncodingOp,
    ops::AddPositionEncodingOpMaker,
    ops::AddPositionEncodingGradOpMaker<paddle::framework::OpDesc>,
    ops::AddPositionEncodingGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(add_position_encoding_grad, ops::AddPositionEncodingOpGrad);

REGISTER_OP_CPU_KERNEL(add_position_encoding,
                       ops::AddPositionEncodingKernel<phi::CPUContext, float>,
                       ops::AddPositionEncodingKernel<phi::CPUContext, double>);

REGISTER_OP_CPU_KERNEL(
    add_position_encoding_grad,
    ops::AddPositionEncodingGradKernel<phi::CPUContext, float>,
    ops::AddPositionEncodingGradKernel<phi::CPUContext, double>);
