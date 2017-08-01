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

#include "paddle/operators/mean_op.h"

namespace paddle {
namespace operators {

class MeanOp : public OperatorWithKernel {
protected:
  void InferShape(const std::vector<const Tensor *> &inputs,
                  const std::vector<Tensor *> &outputs) const override {
    PADDLE_ENFORCE(inputs.size() == 1, "Input size of AddOp must be one");
    PADDLE_ENFORCE(outputs.size() == 1, "Output size of AddOp must be one");
    PADDLE_ENFORCE(inputs[0] != nullptr && outputs[0] != nullptr,
                   "Input/Output of MeanOp must be initialized.");
    outputs[0]->Resize(framework::make_ddim({1}));
  }
};

class MeanOpMaker : public OpProtoAndCheckerMaker {
public:
  MeanOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of mean op");
    AddOutput("Out", "The output of mean op");
    AddComment("Mean Operator");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP(mean, ops::MeanOp, ops::MeanOpMaker);
REGISTER_OP_CPU_KERNEL(mean, ops::MeanKernel<ops::CPUPlace, float>);
