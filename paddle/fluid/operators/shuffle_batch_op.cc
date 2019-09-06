// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/shuffle_batch_op.h"
#include <memory>
#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {
class ShuffleBatchOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true, "Input(X) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("ShuffleIdx"), true,
                      "Output(ShuffleIdx) should not be null.");

    ctx->ShareDim("X", "Out");
    ctx->ShareLoD("X", "Out");
    ctx->SetOutputDim("ShuffleIdx", framework::make_ddim({-1}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("X"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class ShuffleBatchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) The input tensor of shuffle_batch op.");
    AddAttr<std::vector<int64_t>>(
        "ShuffleOrder",
        "(Tensor) Predefined shuffle order. This attr is used for op test. "
        "If not set(Default), shuffle order will be generated randomly");
    AddOutput("Out", "(LoDTensor) The output tensor of shuffle_batch op.");
    AddOutput("ShuffleIdx", "(Tensor) Record forword shuffle order");
    AddComment(R"DOC(
Shuffle Batch Operator.

This operator is used to shuffle input $X$'s elements.

There is 1 input. The product of input dims (except last dim) numbers of elements will be shuffled.

There are 2 outputs. $Out$ is shuffled tensor of input. $ShuffleIdx$ is the tensor used to record shuffle order.
)DOC");
  }
};

class ShuffleBatchOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("ShuffleIdx"), true,
                      "Input(ShuffleIdx) should not be null");
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      "Grad Input(Out) should not be null)");
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")), true,
                      "Grad Output(X) should not be null");

    ctx->ShareDim(framework::GradVarName("Out"), framework::GradVarName("X"));
    ctx->ShareLoD(framework::GradVarName("Out"), framework::GradVarName("X"));
  }

 protected:
 public:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(
        ctx.InputVar(framework::GradVarName("Out")));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class ShuffleBatchOpGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("shuffle_batch_grad");
    op->SetInput("ShuffleIdx", Output("ShuffleIdx"));
    op->SetAttrMap(Attrs());
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(shuffle_batch, ops::ShuffleBatchOp, ops::ShuffleBatchOpMaker,
                  ops::ShuffleBatchOpGradDescMaker);
REGISTER_OPERATOR(shuffle_batch_grad, ops::ShuffleBatchOpGrad);

REGISTER_OP_CPU_KERNEL(shuffle_batch, ops::ShuffleBatchKernel<float>,
                       ops::ShuffleBatchKernel<double>,
                       ops::ShuffleBatchKernel<int32_t>,
                       ops::ShuffleBatchKernel<int64_t>);

REGISTER_OP_CPU_KERNEL(shuffle_batch_grad, ops::ShuffleBatchGradKernel<float>,
                       ops::ShuffleBatchGradKernel<double>,
                       ops::ShuffleBatchGradKernel<int32_t>,
                       ops::ShuffleBatchGradKernel<int64_t>);
