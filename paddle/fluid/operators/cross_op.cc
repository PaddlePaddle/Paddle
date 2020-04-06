// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/cross_op.h"
#include <memory>

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::DDim;

class CrossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of CrossOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true,
                      platform::errors::InvalidArgument(
                          "Input(Index) of CrossOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of CrossOp should not be null."));

    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("Y");
    auto dim = ctx->Attrs().Get<int>("dim");

    bool dims_match = CheckDims(x_dim, y_dim);
    PADDLE_ENFORCE_EQ(dims_match, true,
                      platform::errors::InvalidArgument(
                          "The 'shape' of Input(X) should be equal to "
                          "the 'shape' of Input(Y). But received "
                          "Input(X).dimensions = [%s], "
                          "Input(Y).dimensions = [%s]",
                          x_dim, y_dim));

    if (dim != kDefaultDim) {
      PADDLE_ENFORCE_EQ(
          dim < x_dim.size() && dim >= (0 - x_dim.size()), true,
          platform::errors::OutOfRange(
              "Attr(dim) is out of range, It's expected "
              "to be in range of [-%d, %d]. But received Attr(dim) = %d.",
              x_dim.size(), x_dim.size() - 1, dim));
      if (dim < 0) {
        dim += x_dim.size();
      }
      PADDLE_ENFORCE_EQ(x_dim[dim] == 3 && y_dim[dim] == 3, true,
                        platform::errors::InvalidArgument(
                            "Input(X/Y).dims()[dim] should be equal to 3."
                            "But received Input(X/Y).dims()[dim] = %d.",
                            x_dim[dim]));
    }

    ctx->SetOutputDim("Out", x_dim);
    auto type = ctx->GetInputsVarType("X")[0];
    if (type == framework::proto::VarType::LOD_TENSOR) {
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class CrossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should be not null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::InvalidArgument("Input(Y) should be not null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Input(Out@GRAD) should be not null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")), true,
                      platform::errors::InvalidArgument(
                          "Output(X@GRAD) should be not null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Y")), true,
                      platform::errors::InvalidArgument(
                          "Output(Y@GRAD) should be not null."));

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->SetOutputDim(framework::GradVarName("Y"), ctx->GetInputDim("Y"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class CrossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) the input tensor.");
    AddInput("Y", "(Tensor) the second input tensor.");
    AddOutput("Out", "(Tensor), the output tensor.");
    AddAttr<int>("dim", "the dimension to take the cross-product in.")
        .SetDefault(kDefaultDim);
    AddComment(R"DOC(
    Returns the cross product of vectors in dimension dim of
    input and other. Input and other must have the same size,
    and the size of their dim dimension should be 3.
    If dim is not given, it defaults to the first dimension
    found with the size 3.
    )DOC");
  }
};

template <typename T>
class CrossGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("cross_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(cross, ops::CrossOp, ops::CrossOpMaker,
                  ops::CrossGradMaker<paddle::framework::OpDesc>,
                  ops::CrossGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(cross_grad, ops::CrossGradOp);
REGISTER_OP_CPU_KERNEL(
    cross, ops::CrossKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CrossKernel<paddle::platform::CPUDeviceContext, double>,
    ops::CrossKernel<paddle::platform::CPUDeviceContext, int>,
    ops::CrossKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    cross_grad, ops::CrossGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CrossGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::CrossGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::CrossGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
