/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/unique_op.h"

namespace paddle {
namespace operators {

class UniqueOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of UniqueOp shold not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of UniqueOp shold not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Index"),
                   "Output(Index) of UniqueOp shold not be null.");

    auto in_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(in_dims.size() == 1, "Input(X) shold be a vector.");

    ctx->SetOutputDim("Out", {-1});
    ctx->SetOutputDim("Index", in_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        platform::CPUPlace());
  }
};

class UniqueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input tensor. It shold be a 1-D tensor.");
    AddAttr<int>("dtype", "data type for output indice");
    AddOutput("Out", "A unique subsequence for input tensor.");
    AddOutput("Index",
              "An indice tensor pointing to unique subsequence, which has "
              "identical shape with input tensor and int64 dtype.");
    AddComment(R"DOC(
    Return a unique subsequence for 1-D input tensor, and an indice tensor pointing to this unique subsequence
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(unique, ops::UniqueOp, ops::UniqueOpMaker);
REGISTER_OP_CPU_KERNEL(unique, ops::UniqueKernel<float>,
                       ops::UniqueKernel<double>, ops::UniqueKernel<int32_t>,
                       ops::UniqueKernel<int64_t>);
