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

#include "paddle/operators/sgd_op.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {

class SGDOp : public framework::OperatorWithKernel {
protected:
  void InferShape(
      const std::vector<const framework::Tensor *> &inputs,
      const std::vector<framework::Tensor *> &outputs) const override {
    PADDLE_ENFORCE(inputs.size() == 2, "Input size of SGDOp must be two");
    PADDLE_ENFORCE(outputs.size() == 1, "Output size of SGDOp must be one");
    PADDLE_ENFORCE(inputs[0] != nullptr, "inputs[0] mast be set");
    PADDLE_ENFORCE(inputs[1] != nullptr, "inputs[1] mast be set");
    PADDLE_ENFORCE(outputs[0] != nullptr, "outputs[0] mast be set");
    PADDLE_ENFORCE(inputs[0]->dims() == inputs[1]->dims(),
                   "Two input of SGD Op's dimension must be same.");
    outputs[0]->Resize(inputs[0]->dims());
  }
};

class SGDOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  SGDOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("param", "input parameter");
    AddInput("grad", "input gradient");
    AddOutput("param_out", "output parameter");
    AddAttr<float>("learning_rate", "learning rate of sgd");
    AddComment(R"DOC(

Simplest sgd algorithm.

param_out = param - learning_rate * grad;

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP(sgd, paddle::operators::SGDOp, paddle::operators::SGDOpMaker);
typedef paddle::operators::SGDOpKernel<::paddle::platform::CPUPlace, float>
    SGDOpKernel_CPU_float;
REGISTER_OP_CPU_KERNEL(sgd, SGDOpKernel_CPU_float);
