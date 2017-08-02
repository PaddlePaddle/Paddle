/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/fill_zeros_like_op.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {

class FillZerosLikeOp : public framework::OperatorWithKernel {
protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(ctx.InputSize() == 1UL,
                   "Input size of FillZerosLikeOp must be one.");
    PADDLE_ENFORCE(ctx.OutputSize() == 1UL,
                   "Output size of AddOp must be one.");
    PADDLE_ENFORCE(ctx.InputVar(0) != nullptr,
                   "Input of FillZerosLikeOp must be set.");
    PADDLE_ENFORCE(ctx.OutputVar(0) != nullptr,
                   "Output of FillZerosLikeOp must be set.");
    ctx.Output<framework::Tensor>(0)->Resize(
        ctx.Input<framework::Tensor>(0)->dims());
  }
};

class FillZerosLikeOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  FillZerosLikeOpMaker(framework::OpProto *proto,
                       framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Src", "The input of fill-zeros-like op.");
    AddOutput("Dst", "The varibale will be filled up with zeros.");
    AddComment(R"DOC(
Fill up a vriable with zeros.

The output will have the same size with input.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP(fill_zeros_like,
            paddle::operators::FillZerosLikeOp,
            paddle::operators::FillZerosLikeOpMaker);
REGISTER_OP_CPU_KERNEL(
    fill_zeros_like,
    paddle::operators::FillZerosLikeKernel<paddle::platform::CPUPlace, float>);
