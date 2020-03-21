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

#include "paddle/fluid/operators/index_select_op.h"
#include <memory>

namespace paddle {
namespace operators {

using framework::Tensor;

class IndexSelectOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) of IndexSelectOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Index"), true,
                      "Input(Index) of IndexSelectOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of IndexSelectOp should not be null.");

    auto input_dim = ctx->GetInputDim("X");
    auto index_dim = ctx->GetInputDim("Index");
    auto dim = ctx->Attrs().Get<int>("dim");

    PADDLE_ENFORCE_LT(dim, input_dim.size(),
                      "ShapeError: dim must be less than the dimensions "
                      "of the input(X). But received select dim = %d, "
                      "input(X)'s dimensions = [%s].",
                      dim, input_dim.size());
    PADDLE_ENFORCE_GE(dim, 0,
                      "ShapeError: dim must be greater or equal to 0. "
                      "But received select dim = %d.",
                      dim);
    PADDLE_ENFORCE_EQ(
        index_dim.size() == 1 || (index_dim.size() == 2 && index_dim[1] == 1),
        true,
        "ShapeError: index must be 1-D tensor, "
        "But received: the shape of index is [%s],the dimension "
        "of index is [%d]",
        index_dim, index_dim.size());

    auto output_dim = framework::vectorize(input_dim);
    output_dim[dim] = index_dim[0];
    ctx->SetOutputDim("Out", framework::make_ddim(output_dim));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class IndexSelectGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Index"), "Input(Index) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out"));
  };
}

class IndexSelectOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) the input tensor.");
    AddInput("Index", "the 1-D tensor containing the indices to index.");
    AddOutput("Out", "the output tensor.");
    AddAttr<int>("dim", "the dimension in which we index.").SetDefault(0);
    AddComment(R"DOC(
    Returns a new tensor which indexes the input tensor
    along dimension dim using the entries in index which
    is a Tensor.

    The returned tensor has the same number of dimensions
    as the original tensor (input). The dimth dimension
    has the same size as the length of index; other dimensions
    have the same size as in the original tensor.
    )DOC");
  }
};

template <typename T>
class IndexSelectGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("index_select_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("Index", this->Input("Index"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
}
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(index_select, ops::IndexSelectOp, ops::IndexSelectOpMaker,
                  ops::IndexSelectGradMaker<paddle::framework::OpDesc>,
                  ops::IndexSelectGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(index_select_grad, ops::IndexSelectGradOp);
REGISTER_OP_CPU_KERNEL(
    index_select,
    ops::IndexSelectKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    index_select_grad,
    ops::IndexSelectGradKernel<paddle::platform::CPUDeviceContext, float>);
