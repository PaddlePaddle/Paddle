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

#include "paddle/operators/mul_op.h"

namespace paddle {
namespace operators {

class MulOp : public OperatorWithKernel {
protected:
  void InferShape(const std::vector<const Tensor *> &inputs,
                  const std::vector<Tensor *> &outputs) const override {
    PADDLE_ENFORCE(inputs.size() == 2, "The mul op must take two inputs");
    auto dim0 = inputs[0]->dims();
    auto dim1 = inputs[1]->dims();
    PADDLE_ENFORCE(dim0.size() == 2 && dim1.size() == 2,
                   "The input of mul op must be matrix");
    PADDLE_ENFORCE(
        dim0[1] == dim1[0],
        "First matrix's width must be equal with second matrix's height.");
    PADDLE_ENFORCE(outputs.size() == 1, "The mul op must take one output");
    outputs[0]->Resize({dim0[0], dim1[1]});
  }
};

class MulOpMaker : public OpProtoAndCheckerMaker {
public:
  MulOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of mul op");
    AddInput("Y", "The second input of mul op");
    AddOutput("Out", "The output of mul op");
    AddComment(R"DOC(
Two Element Mul Operator.

The equation is: Out = X * Y
)DOC");
  }
};

class MulOpGrad : public OperatorWithKernel {
protected:
  void InferShape(const std::vector<const Tensor *> &inputs,
                  const std::vector<Tensor *> &outputs) const override {}
  std::string DebugString() const override {
    LOG(INFO) << "MulGrad";
    return "";
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP(mul, ops::MulOp, ops::MulOpMaker);
REGISTER_GRADIENT_OP(mul, mul_grad, ops::MulOpGrad);

REGISTER_OP_CPU_KERNEL(mul, ops::MulKernel<ops::CPUPlace, float>);
