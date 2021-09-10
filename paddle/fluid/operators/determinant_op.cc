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

#include "paddle/fluid/operators/determinant_op.h"

namespace paddle {
namespace operators {

class DeterminantOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "determinant");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "determinant");
  }
};

class DeterminantOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor) The input tensor of determinant.");
    AddOutput("Out",
              "(Tensor) The output Tensor cotaining the determinant"
              "value of a square matrix or batches of square matrices ");

    AddComment(R"DOC(
Determinant Operator.)DOC");
  }
};

class DeterminantGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input",
                   "DeterminantGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Input")), "Output",
                   framework::GradVarName("Input"), "DeterminantGradOp");

    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class DeterminantGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("determinant_grad");
    grad_op->SetInput("Input", this->Input("Input"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("Input"),
                       this->InputGrad("Input"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(DeterminantGradNoNeedBufferVarsInferer,
                                    "Input");

class SlogDeterminantOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "determinant");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "determinant");
  }
};

class SlogDeterminantOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor) The input tensor of SlogDeterminant.");
    AddOutput("Out",
              "(Tensor) The output tensor containing the sign of the"
              "determinant and the natural logarithm"
              "of the absolute value of determinant,");

    AddComment(R"DOC(
SlogDeterminant Operator.)DOC");
  }
};

class SlogDeterminantGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input",
                   "SlogDeterminantGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Input")), "Output",
                   framework::GradVarName("Input"), "SlogDeterminantGradOp");

    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class SlogDeterminantGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("slogdeterminant_grad");
    grad_op->SetInput("Input", this->Input("Input"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("Input"),
                       this->InputGrad("Input"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SlogDeterminantGradNoNeedBufferVarsInferer,
                                    "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(determinant, ops::DeterminantOp, ops::DeterminantOpMaker,
                  ops::DeterminantGradOpMaker<paddle::framework::OpDesc>,
                  ops::DeterminantGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(determinant_grad, ops::DeterminantGradOp)

REGISTER_OP_CPU_KERNEL(determinant, ops::DeterminantKernel<int>,
                       ops::DeterminantKernel<int64_t>,
                       ops::DeterminantKernel<float>,
                       ops::DeterminantKernel<double>,
                       ops::DeterminantKernel<bool>);

REGISTER_OP_CPU_KERNEL(determinant_grad, ops::DeterminantGradKernel<int>,
                       ops::DeterminantGradKernel<int64_t>,
                       ops::DeterminantGradKernel<float>,
                       ops::DeterminantGradKernel<double>,
                       ops::DeterminantGradKernel<bool>);

REGISTER_OPERATOR(slogdeterminant, ops::SlogDeterminantOp,
                  ops::SlogDeterminantOpMaker,
                  ops::SlogDeterminantGradOpMaker<paddle::framework::OpDesc>,
                  ops::SlogDeterminantGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(slogdeterminant_grad,
                  ops::DeterminantGradOp)  // reuse det grad op

REGISTER_OP_CPU_KERNEL(slogdeterminant, ops::SlogDeterminantKernel<int>,
                       ops::SlogDeterminantKernel<int64_t>,
                       ops::SlogDeterminantKernel<float>,
                       ops::SlogDeterminantKernel<double>,
                       ops::SlogDeterminantKernel<bool>);

REGISTER_OP_CPU_KERNEL(slogdeterminant_grad, ops::DeterminantGradKernel<int>,
                       ops::DeterminantGradKernel<int64_t>,
                       ops::DeterminantGradKernel<float>,
                       ops::DeterminantGradKernel<double>,
                       ops::DeterminantGradKernel<bool>);
