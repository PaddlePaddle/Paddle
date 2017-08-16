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

#include "paddle/operators/rowwise_add_op.h"

namespace paddle {
namespace operators {

class RowWiseAddOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto dim0 = ctx.Input<Tensor>("X")->dims();
    auto dim1 = ctx.Input<Tensor>("b")->dims();

    PADDLE_ENFORCE(dim0.size() == 2, "Input 0 must be matrix");
    PADDLE_ENFORCE(dim1.size() == 1, "The second input must be vector");
    PADDLE_ENFORCE(dim0[1] == dim1[0], "The width of two input must be same");
    PADDLE_ENFORCE(ctx.OutputSize("Out") == 1, "The output size must be 1");
    ctx.Output<Tensor>("Out")->Resize(ctx.Input<Tensor>("X")->dims());
  }
};

class RowWiseAddOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RowWiseAddOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The left input of row-wise add op, must be matrix");
    AddInput("b", "The right input of row-wise add op, must be vector");
    AddOutput("Out", "The output of row-wise add op");
    AddComment(R"DOC(Row-wise Add operator

for i in xrange(X.shape[0]):
  Out = X[i] + b
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(rowwise_add, ops::RowWiseAddOp,
                             ops::RowWiseAddOpMaker);
REGISTER_OP_CPU_KERNEL(
    rowwise_add, ops::RowWiseAddKernel<paddle::platform::CPUPlace, float>);
