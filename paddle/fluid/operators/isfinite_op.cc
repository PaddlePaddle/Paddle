// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/isfinite_op.h"

#include <string>

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

class OverflowOp : public framework::OperatorWithKernel {
 public:
  OverflowOp(const std::string &type, const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "isfinite");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "isfinite");

    ctx->SetOutputDim("Out", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    int dtype = -1;
    auto *x_var = ctx.InputVar("X");
    if (x_var->IsType<framework::LoDTensor>()) {
      dtype = framework::TransToProtoVarType(
          x_var->Get<framework::LoDTensor>().type());
    } else if (x_var->IsType<pten::SelectedRows>()) {
      dtype = framework::TransToProtoVarType(
          x_var->Get<pten::SelectedRows>().value().type());
    } else {
      PADDLE_ENFORCE_EQ(
          true, false,
          platform::errors::InvalidArgument(
              "The input type mismatch, the type of Input(X) must be Tensor or "
              "SelectedRows, please check your input."));
    }
    return framework::OpKernelType(framework::proto::VarType::Type(dtype),
                                   ctx.GetPlace());
  }
};

class OverflowOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensors of overflow operator.");
    AddOutput("Out",
              "(Tensor) 1-dim tensor, contains a bool scalar. The output "
              "tensor of overflow operator.");
    AddComment(string::Sprintf(R"DOC(
Overflow %s operator.

$$Out = any(X)$$

If any X contains Inf or Nan, the Out will generate a indicator.
Out = Inf if any X contains Inf,
Out = Nan if any X contains Nan,
Out = 0 if no Inf/Nan detected.
If X contains both Inf/Nan, it will return the first indicator it meeted.

%s
)DOC",
                               GetName(), GetComments()));
  }

 protected:
  virtual std::string GetName() const = 0;
  virtual std::string GetComments() const = 0;
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_OP_MAKER(op_type, comment)                           \
  namespace paddle {                                                  \
  namespace operators {                                               \
  class _##op_type##OverflowOpMaker                                   \
      : public ::paddle::operators::OverflowOpMaker {                 \
   protected:                                                         \
    std::string GetName() const { return #op_type; }                  \
    std::string GetComments() const { return comment; }               \
  };                                                                  \
  }                                                                   \
  }                                                                   \
  REGISTER_OPERATOR(                                                  \
      op_type, ops::OverflowOp, ops::_##op_type##OverflowOpMaker,     \
      paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>, \
      paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

#define REGISTER_OVERFLOW_CPU_KERNEL(op_type, functor)                      \
  REGISTER_OP_CPU_KERNEL(                                                   \
      op_type, ops::OverflowKernel<paddle::platform::CPUDeviceContext, int, \
                                   ops::functor>,                           \
      ops::OverflowKernel<paddle::platform::CPUDeviceContext, int64_t,      \
                          ops::functor>,                                    \
      ops::OverflowKernel<paddle::platform::CPUDeviceContext, float,        \
                          ops::functor>,                                    \
      ops::OverflowKernel<paddle::platform::CPUDeviceContext, double,       \
                          ops::functor>);

REGISTER_OP_MAKER(isinf, "isinf(X)");
REGISTER_OP_MAKER(isnan, "isnan(X)");
REGISTER_OP_MAKER(isfinite, "isfinite(X)");

REGISTER_OP_CPU_KERNEL(isinf,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           int, ops::InfinityFunctor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           int64_t, ops::InfinityFunctor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           float, ops::InfinityFunctor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           double, ops::InfinityFunctor>);

REGISTER_OP_CPU_KERNEL(isnan,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           int, ops::NANFunctor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           int64_t, ops::NANFunctor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           float, ops::NANFunctor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           double, ops::NANFunctor>);

REGISTER_OP_CPU_KERNEL(isfinite,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           int, ops::IsfiniteFunctor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           int64_t, ops::IsfiniteFunctor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           float, ops::IsfiniteFunctor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           double, ops::IsfiniteFunctor>);
