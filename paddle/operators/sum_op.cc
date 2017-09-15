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

#include "paddle/operators/sum_op.h"
#include <vector>

namespace paddle {
namespace operators {
using framework::Tensor;

class SumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE(!ctx.MultiInputVar("X").empty(),
                   "Input(X) of SumOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Output(Out) of SumOp should not be null.");

    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto *out = ctx.Output<framework::LoDTensor>("Out");
    int N = ins.size();

    auto in_dim = ins[0]->dims();

    PADDLE_ENFORCE_GT(N, 1, "Input tensors count should > 1.");
    for (int i = 1; i < N; i++) {
      auto dim = ins[i]->dims();
      PADDLE_ENFORCE(in_dim == dim, "Input tensors must have same shape");
    }
    out->Resize(in_dim);
  }
};

class SumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SumOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "the input tensors of sum operator.").AsDuplicable();
    AddOutput("Out", "the output tensor of sum operator.");
    AddComment(R"DOC(
            Sum the input tensors.
        )DOC");
  }
};

class SumGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto outputs =
        ctx.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));
    auto dims = ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();
    for (auto output : outputs) {
      output->Resize(dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sum, ops::SumOp, ops::SumOpMaker, sum_grad, ops::SumGradOp);
REGISTER_OP_CPU_KERNEL(sum, ops::SumKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(sum_grad,
                       ops::SumGradKernel<paddle::platform::CPUPlace, float>);
