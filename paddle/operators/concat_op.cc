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

namespace paddle {
namespace operators {
using framework::Tensor;

class ConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    // TODO(Yancey1989)
  }
};

class ConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ConcatOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    // TODO(Yancey1989)
    AddComment(R"DOC(
            Concat the input tensor
        )DOC");
  }
};

class ConcatOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    // TODO(Yancey1989)
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(concat, ops::ConcatOp, ops::ConcatOpMaker, concat_grad,
            ops::ConcatOpGrad)
REGISTER_OP_CPU_KERNEL(concat,
                       ops::ConcatKernel<paddle::platform::CPUPlace, float>)
