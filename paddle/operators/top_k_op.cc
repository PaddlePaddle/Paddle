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

#include "paddle/operators/mean_op.h"

namespace paddle {
namespace operators {

class TopkOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input of TopkOP must be initialized.");
    const int beam = static_cast<T>(context.op_.GetAttr<AttrType>("k"));
    ctx.Output<Tensor>("Out")->Resize({beam});
  }
};

class TopkOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TopkOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of Topk op");
    AddOutput("Out", "The output of Topk op");
    AddComment("Get top k elements of 2d tensor(matrix) for each row.");
    AddAttr<AttrType>("k", "The k of top k elements.").SetDefault(1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(topk, ops::TopkOp, ops::TopkOpMaker);
REGISTER_OP_CPU_KERNEL(topk,
                       ops::TopkKernel<paddle::platform::CPUPlace, float>);
