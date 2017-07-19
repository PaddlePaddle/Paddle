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

#include "paddle/operators/cross_entropy_op.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {

class CrossEntropyOp : public framework::OperatorWithKernel {
protected:
  void InferShape(
      const std::vector<const framework::Tensor *> &inputs,
      const std::vector<framework::Tensor *> &outputs) const override {
    PADDLE_ENFORCE(inputs.size() == 2,
                   "Input size of CrossEntropyOp must be two");
    PADDLE_ENFORCE(outputs.size() == 1,
                   "Output size of CrossEntropyOp must be one");
    PADDLE_ENFORCE(
        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
        "Inputs/Outputs of CrossEntropyOp must all be set");
    PADDLE_ENFORCE(inputs[0]->dims() == inputs[1]->dims(),
                   "Two input of CrossEntropyOp's dimension must be same.");
    // set dim
  }
};

class CrossEntropyOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  CrossEntropyOpMaker(framework::OpProto *proto,
                      framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of CrossEntropyOp");
    AddInput("label", "The second input of CrossEntropyOp");
    AddOutput("Y", "The output of CrossEntropyOp");
    AddComment(R"DOC(
Two Element CrossEntropy Operator.

Operator computes the cross entropy between the input and the label set. In
 practice, it is most commonly used at the end of models, after the SoftMax
 operator and before the AveragedLoss operator. Note that CrossEntropy
 assumes that the soft labels provided is a 2D array of size N x D
 (batch size x number of classes). Each entry in the 2D label corresponds to
 the soft label for the input, where each element represents the correct
 probability of the class being selected. As such, each element must be between
 0 and 1, and all elements in an entry must sum to 1. The formula used is:

                Y[i] = sum_j (label[i][j] * log(X[i][j]))

 where (i, j) is the classifier's prediction of the jth class (the correct one),
 and i is the batch size. Each log has a lower limit for numerical stability.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP(cross_entropy,
            paddle::operators::CrossEntropyOp,
            paddle::operators::CrossEntropyOpMaker);
typedef paddle::operators::CrossEntropyOpKernel<::paddle::platform::CPUPlace,
                                                float>
    CrossEntropyOpKernel_CPU_float;
REGISTER_OP_CPU_KERNEL(cross_entropy, CrossEntropyOpKernel_CPU_float);
