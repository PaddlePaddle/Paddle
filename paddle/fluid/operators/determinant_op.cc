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
    AddInput(
        "Input",
        "(Tensor) The input tensor, from which the determinant are taken.");
    AddOutput("Out",
              "(Tensor) The partial view of input with the its determinant "
              "elements.");

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
    AddInput(
        "Input",
        "(Tensor) The input tensor, from which the determinant are taken.");
    AddOutput("Out",
              "(Tensor) The partial view of input with the its slogdeterminant "
              "elements.");

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

REGISTER_OP_CPU_KERNEL(determinant, ops::DeterminantKernel<int>,
                       ops::DeterminantKernel<int64_t>,
                       ops::DeterminantKernel<float>,
                       ops::DeterminantKernel<double>,
                       ops::DeterminantKernel<bool>);

REGISTER_OPERATOR(slogdeterminant, ops::SlogDeterminantOp,
                  ops::SlogDeterminantOpMaker,
                  ops::SlogDeterminantGradOpMaker<paddle::framework::OpDesc>,
                  ops::SlogDeterminantGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(slogdeterminant, ops::SlogDeterminantKernel<int>,
                       ops::SlogDeterminantKernel<int64_t>,
                       ops::SlogDeterminantKernel<float>,
                       ops::SlogDeterminantKernel<double>,
                       ops::SlogDeterminantKernel<bool>);
