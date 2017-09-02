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
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto *out = ctx.Output<framework::Tensor>("Out");
    int N = ins.size();

    PADDLE_ENFORCE_GT(N, 1, "Input tensors count should > 1.");

    auto dim_zero = ins[0]->dims();
    out->Resize(dim_zero);
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(sum, ops::SumOp, ops::SumOpMaker)
REGISTER_OP_CPU_KERNEL(sum, ops::SumKernel<paddle::platform::CPUPlace, float>)
