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

class RowWiseAddOp : public OperatorWithKernel {
protected:
  void InferShape(const std::vector<const Tensor *> &inputs,
                  const std::vector<Tensor *> &outputs) const override {
    PADDLE_ENFORCE(inputs.size() == 2UL, "Two inputs is needed by rowwise add");
    auto dim0 = inputs[0]->dims();
    auto dim1 = inputs[1]->dims();

    PADDLE_ENFORCE(dim0.size() == 2, "Input 0 must be matrix");
    PADDLE_ENFORCE(dim1.size() == 1, "The second input must be vector");
    PADDLE_ENFORCE(dim0[1] == dim1[0], "The width of two input must be same");
    PADDLE_ENFORCE(outputs.size() == 1, "The output size must be 1");
    outputs[0]->Resize(inputs[0]->dims());
  }
};

class RowWiseAddOpMaker : public OpProtoAndCheckerMaker {
public:
  RowWiseAddOpMaker(OpProto *proto, OpAttrChecker *op_checker)
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

REGISTER_OP(rowwise_add, ops::RowWiseAddOp, ops::RowWiseAddOpMaker);
REGISTER_OP_CPU_KERNEL(rowwise_add,
                       ops::RowWiseAddKernel<ops::CPUPlace, float>);
