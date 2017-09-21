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

#include "paddle/operators/sequence_softmax_op.h"

namespace paddle {
namespace operators {

class SequenceSoftmaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(
        ctx.InputVar("X"), "Input(X) of SequenceSoftmaxOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(
        ctx.OutputVar("Out"),
        "Output(Out) of SequenceSoftmaxOp should not be null.");

    auto *x = ctx.Input<framework::LoDTensor>("X");
    auto dims = x->dims();
    auto lod = x->lod();
    PADDLE_ENFORCE_EQ(lod.size(), 1UL, "Only support one level sequence now.");
    PADDLE_ENFORCE_GE(
        dims[0],
        /* batch_size */ static_cast<int64_t>(lod[0].size() - 1),
        "The first dimension of Input(X) should be larger than batch size.");
    PADDLE_ENFORCE_EQ(x->numel(), static_cast<int64_t>(lod[0].size() - 1),
                      "The width of each timestep in Input(X) of "
                      "SequenceSoftmaxOp should be 1.");

    dims[0] = lod[0].size() - 1;
    ctx.Output<framework::LoDTensor>("Out")->Resize({dims});
  }
};

class SequenceSoftmaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequenceSoftmaxOpMaker(framework::OpProto *proto,
                         framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(LoDTensor)");
    AddOutput("Out", "(LoDTensor)");
    AddComment(R"DOC(
Softmax of Sequence.
)DOC");
  }
};

class SequenceSoftmaxGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sequence_softmax, ops::SequenceSoftmaxOp,
            ops::SequenceSoftmaxOpMaker, sequence_softmax_grad,
            ops::SequenceSoftmaxGradOp);
REGISTER_OP_CPU_KERNEL(
    sequence_softmax,
    ops::SequenceSoftmaxKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    sequence_softmax_grad,
    ops::SequenceSoftmaxGradKernel<paddle::platform::GPUPlace, float>);
