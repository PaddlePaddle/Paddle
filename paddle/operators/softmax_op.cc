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
#include "paddle/operators/softmax_op.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

class SoftmaxOp : public framework::OperatorWithKernel {
protected:
  void InferShape(
      const std::vector<const framework::Tensor *> &inputs,
      const std::vector<framework::Tensor *> &outputs) const override {
    PADDLE_ENFORCE(inputs.size() == 1, "Only one input is need for softmax");
    PADDLE_ENFORCE(inputs[0]->dims().size() == 2,
                   "The input of softmax op must be matrix");
    PADDLE_ENFORCE(outputs.size() == 1, "Only one output is need for softmax");

    outputs[0]->Resize(inputs[0]->dims());
  }
};

class SoftmaxOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  SoftmaxOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "input of softmax");
    AddOutput("Y", "output of softmax");
    AddComment("Softmax Op");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(softmax, ops::SoftmaxOp, ops::SoftmaxOpMaker);
REGISTER_OP_CPU_KERNEL(softmax,
                       ops::SoftmaxKernel<paddle::platform::CPUPlace, float>);
