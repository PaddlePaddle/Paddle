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

#include "paddle/fluid/operators/cross_op.h"
#include <memory>

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::DDim;

inline bool CheckDims(const DDim& dims_x, const DDim& dims_y) {
  if (dims_x.size() != dims_y.size()) {
    return false;
  }
  for (int i = 0; i < dims_x.size(); i++) {
    if (dims_x[i] != dims_y[i]) {
      return false;
    }
  }
  return true;
}

class CrossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) of CrossOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true,
                      "Input(Index) of CrossOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of CrossOp should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("Y");
    auto dim = ctx->Attrs().Get<int>("dim");

    bool check_dim = CheckDims(x_dim, y_dim);
    PADDLE_ENFORCE_EQ(check_dim, true,
                      "ShapeError: Input(X).dims() should be equal to "
                      "Input(Y).dims(). But received Input(X).dimensions"
                      " = [%s], Input(Y).dimensions = [%s]",
                      x_dim, y_dim);

    if (dim != -1) {
      PADDLE_ENFORCE_GT(
          x_dim.size(), dim,
          "ShapeError: Input(X).dims().size() should be greater than Attr(dim)."
          "But received Input(X).dimensions = [%s], Attr(dim) = %d.",
          x_dim, dim);
      PADDLE_ENFORCE_GT(
          y_dim.size(), dim,
          "ShapeError: Input(Y).dims().size() should be greater than Attr(dim)."
          "But received Input(Y).dimensions = [%s], Attr(dim) = %d.",
          y_dim, dim);
      PADDLE_ENFORCE_EQ(x_dim[dim], 3,
                        "ShapeError: Input(X).dims()[dim] should be equal to 3."
                        "But received Input(X).dims()[dim] = %d.",
                        x_dim[dim]);
      PADDLE_ENFORCE_EQ(y_dim[dim], 3,
                        "ShapeError: Input(Y).dims()[dim] should be equal to 3."
                        "But received Input(Y).dims()[dim] = %d.",
                        y_dim[dim]);
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
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true, "Input(X) should be not null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true, "Input(Y) should be not null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      "Input(Out@GRAD) should be not null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")), true,
                      "Output(X@GRAD) should be not null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Y")), true,
                      "Output(Y@GRAD) should be not null.");

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
        .SetDefault(-1);
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
    cross, ops::CrossKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    cross_grad,
    ops::CrossGradKernel<paddle::platform::CPUDeviceContext, float>);
