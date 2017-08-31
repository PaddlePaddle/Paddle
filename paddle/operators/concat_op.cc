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

#include "paddle/operators/concat_op.h"
#include <vector>

namespace paddle {
namespace operators {
using framework::Tensor;

class ConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto *out = ctx.Output<framework::Tensor>("Out");
    const int axis = static_cast<int>(ctx.op_.GetAttr<int>("axis"));
    int N = ins.size();

    PADDLE_ENFORCE_GT(N, 1, "Input tensors count should >= 1.");

    auto dim_zero = ins[0]->dims();
    auto dim_zero_size = dim_zero.size();
    auto concat_dim = dim_zero;
    for (int i = 1; i < N; i++) {
      PADDLE_ENFORCE_EQ(dim_zero_size, ins[i]->dims().size(),
                        "input tensors should have the same dim size.");
      for (int j = 0; j < dim_zero_size; j++) {
        if (j == axis) {
          concat_dim[axis] += ins[i]->dims()[j];
          continue;
        }
        PADDLE_ENFORCE_EQ(dim_zero[j], ins[i]->dims()[j],
                          "Input tensors should have the same "
                          "elements except the specify axis.")
      }
    }
    out->Resize(concat_dim);
  }
};

class ConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ConcatOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "the input tensors of concat operator.");
    AddOutput("Out", "the output tensor of concat operator.");
    AddComment(R"DOC(
            Join the input tensors alone the with the axis.
        )DOC");
    AddAttr<int>("axis", "The axis alone which the inputs will be joined")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(concat, ops::ConcatOp, ops::ConcatOpMaker)
REGISTER_OP_CPU_KERNEL(concat,
                       ops::ConcatKernel<paddle::platform::CPUPlace, float>)
