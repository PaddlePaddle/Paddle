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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class CumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class CumsumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of cumsum operator");
    AddOutput("Out", "Output of cumsum operator");
    AddAttr<int>("axis",
                 "The dimension to accumulate along. -1 means the last "
                 "dimension [default -1].")
        .SetDefault(-1)
        .SupportTensor();
    AddAttr<bool>("flatten",
                  "Whether to compute the cumsum over the flattened array. "
                  "[default false].")
        .SetDefault(false);
    AddAttr<bool>("exclusive",
                  "Whether to perform exclusive cumsum. [default false].")
        .SetDefault(false);
    AddAttr<bool>("reverse",
                  "If true, the cumsum is performed in the reversed direction. "
                  "[default false].")
        .SetDefault(false);
    AddComment(R"DOC(
The cumulative sum of the elements along a given axis.
By default, the first element of the result is the same of the first element of
the input. If exclusive is true, the first element of the result is 0.
)DOC");
  }
};

template <typename T>
class CumsumGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("cumsum");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    grad_op->SetAttr("reverse",
                     !PADDLE_GET_CONST(bool, this->GetAttr("reverse")));
  }
};

class LogcumsumexpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of logcumsumexp operator");
    AddOutput("Out", "Output of logcumsumexp operator");
    AddAttr<int>("axis",
                 "The dimension to accumulate along. -1 means the last "
                 "dimension [default -1].")
        .SetDefault(-1);
    AddAttr<bool>(
        "flatten",
        "Whether to compute the logcumsumexp over the flattened array. "
        "[default false].")
        .SetDefault(false);
    AddAttr<bool>("exclusive",
                  "Whether to perform exclusive logcumsumexp. [default false].")
        .SetDefault(false);
    AddAttr<bool>(
        "reverse",
        "If true, the logcumsumexp is performed in the reversed direction. "
        "[default false].")
        .SetDefault(false);
    AddComment(R"DOC(
Returns the logarithm of the cumulative summation of the exponentiation of elements of input along the given axis.
By default, the first element of the result is the same of the first element of
the input. If exclusive is true, the first element of the result is the lowest finite value of the dtype of output tensor.
)DOC");
  }
};

class LogcumsumexpGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "logcumsumexp");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "logcumsumexp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "logcumsumexp");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

template <typename T>
class LogcumsumexpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("logcumsumexp_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Out", this->Output("Out"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttr("axis", PADDLE_GET_CONST(int, this->GetAttr("axis")));
    grad_op->SetAttr("flatten",
                     PADDLE_GET_CONST(bool, this->GetAttr("flatten")));
    grad_op->SetAttr("exclusive",
                     PADDLE_GET_CONST(bool, this->GetAttr("exclusive")));
    grad_op->SetAttr("reverse",
                     PADDLE_GET_CONST(bool, this->GetAttr("reverse")));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = phi::CPUContext;
DECLARE_INFER_SHAPE_FUNCTOR(cumsum,
                            CumsumInferShapeFunctor,
                            PD_INFER_META(phi::CumScalarAxisInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(logcumsumexp,
                            LogcumsumexpInferShapeFunctor,
                            PD_INFER_META(phi::CumInferMeta));
REGISTER_OPERATOR(cumsum,
                  ops::CumOp,
                  ops::CumsumOpMaker,
                  ops::CumsumGradMaker<paddle::framework::OpDesc>,
                  ops::CumsumGradMaker<paddle::imperative::OpBase>,
                  CumsumInferShapeFunctor);
REGISTER_OPERATOR(logcumsumexp,
                  ops::CumOp,
                  ops::LogcumsumexpOpMaker,
                  ops::LogcumsumexpGradMaker<paddle::framework::OpDesc>,
                  ops::LogcumsumexpGradMaker<paddle::imperative::OpBase>,
                  LogcumsumexpInferShapeFunctor);
REGISTER_OPERATOR(logcumsumexp_grad, ops::LogcumsumexpGradOp);

REGISTER_OP_VERSION(cumsum).AddCheckpoint(
    R"ROC(
      Upgrade cumsum add a new attribute [flatten].
    )ROC",
    paddle::framework::compatible::OpVersionDesc().NewAttr(
        "flatten",
        "In order to compute the cumsum over the flattened array when the "
        "argument `axis` in python API is None.",
        false));
