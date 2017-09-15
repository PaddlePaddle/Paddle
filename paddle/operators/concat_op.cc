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
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Output(Out) of ConcatOp should not be null.");

    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto *out = ctx.Output<framework::LoDTensor>("Out");
    size_t axis = static_cast<size_t>(ctx.Attr<int>("axis"));
    size_t n = ins.size();

    PADDLE_ENFORCE_GT(n, 1, "Input tensors count should > 1.");

    auto out_dims = ins[0]->dims();
    size_t in_zero_dims_size = out_dims.size();
    for (size_t i = 1; i < n; i++) {
      for (size_t j = 0; j < in_zero_dims_size; j++) {
        if (j == axis) {
          out_dims[axis] += ins[i]->dims()[j];
          continue;
        }
        PADDLE_ENFORCE_EQ(out_dims[j], ins[i]->dims()[j],
                          "Input tensors should have the same "
                          "elements except the specify axis.")
      }
    }
    out->Resize(out_dims);
  }
};

class ConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ConcatOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "the input tensors of concat operator.").AsDuplicable();
    AddOutput("Out", "the output tensor of concat operator.");
    AddComment(R"DOC(
            Join the input tensors along with the axis.
            Examples:
              Input[0] = [[1,2],[3,4]]
              Input[1] = [[5,6]]
              axis = 0
              Output = [[1,2],
                        [3,4],
                        [5,6]]
        )DOC");
    AddAttr<int>("axis", "The axis which the inputs will be joined with.")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(concat, ops::ConcatOp, ops::ConcatOpMaker)
REGISTER_OP_CPU_KERNEL(concat,
                       ops::ConcatKernel<paddle::platform::CPUPlace, float>)
