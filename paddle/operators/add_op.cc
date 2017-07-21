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

#include "paddle/operators/add_op.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {

class AddOp : public framework::OperatorWithKernel {
protected:
  void InferShape(
      const std::vector<const framework::Tensor *> &inputs,
      const std::vector<framework::Tensor *> &outputs) const override {
    PADDLE_ENFORCE(inputs.size() == 2, "Input size of AddOp must be two");
    PADDLE_ENFORCE(outputs.size() == 1, "Output size of AddOp must be one");
    PADDLE_ENFORCE(
        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
        "Inputs/Outputs of AddOp must all be set");
    PADDLE_ENFORCE(inputs[0]->dims() == inputs[1]->dims(),
                   "Two input of Add Op's dimension must be same.");
    outputs[0]->Resize(inputs[0]->dims());
  }
};

class AddOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  AddOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of add op");
    AddInput("Y", "The second input of add op");
    AddOutput("Out", "The output of add op");
    AddComment(R"DOC(
Two Element Add Operator.

The equation is: Out = X + Y
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP(add_two, paddle::operators::AddOp, paddle::operators::AddOpMaker);
REGISTER_OP_CPU_KERNEL(
    add_two, paddle::operators::AddKernel<paddle::platform::CPUPlace, float>);
