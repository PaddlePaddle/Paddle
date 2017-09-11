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

#include "paddle/operators/split_op.h"

namespace paddle {
namespace operators {
using framework::Tensor;

class SplitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    // infershape
    auto *in = ctx.Input<framework::Tensor>("X");
    auto *out = ctx.Output<framework::Tensor>("Out");
    out->Resize(in->dims());
  }
};

class SplitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SplitOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "the input tensor of split operator.");
    AddOutput("Out", "the output tensors of split operator.");
    AddComment(R"DOC(
      Split the input tensor into multiple sub-tensors.
      Example:
        Input = [[1,2],
                 [3,4],
                 [5,6]]
        indices = 1
        axis = 0
        Output[0] = [[1,2]]
        Output[1] = [[3,4]]
        Output[2] = [[5,6]] 
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(split, ops::SplitOp, ops::SplitOpMaker)
REGISTER_OP_CPU_KERNEL(split,
                       ops::SplitKernel<paddle::platform::CPUPlace, float>)
