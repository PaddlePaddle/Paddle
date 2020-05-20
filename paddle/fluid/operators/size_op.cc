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

#include "paddle/fluid/operators/size_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class SizeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "Size");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Size");

    ctx->SetOutputDim("Out", {1});
  }
};

class SizeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input tensor.");
    AddOutput("Out",
              "The returned tensor, the data type "
              "is int64_t, will be on the same device with the input Tensor.");
    AddComment(R"DOC(
Size Operator.

Return the number of elements in the input.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    size, ops::SizeOp, ops::SizeOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(size, ops::SizeKernel<int>, ops::SizeKernel<int32_t>,
                       ops::SizeKernel<float>, ops::SizeKernel<double>,
                       ops::SizeKernel<bool>);
