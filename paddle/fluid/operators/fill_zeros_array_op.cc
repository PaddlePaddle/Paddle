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

#include "paddle/fluid/operators/fill_zeros_array_op.h"

namespace paddle {
namespace operators {

class FillZerosArrayOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {}
};

class FillZerosArrayOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of fill-zeros-like op.");
    AddOutput("Out", "The variable will be filled up with zeros.");
    AddComment(R"DOC(
                 FillZerosArray Operator.
                 Fill up an LoDTensorArray with zeros.
                 The output will have the same size as the input.
                 )DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fill_zeros_array, ops::FillZerosArrayOp,
                             ops::FillZerosArrayOpMaker);
REGISTER_OP_CPU_KERNEL(
    fill_zeros_array,
    ops::FillZerosArrayKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FillZerosArrayKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::FillZerosArrayKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FillZerosArrayKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FillZerosArrayKernel<paddle::platform::CPUDeviceContext, bool>);
