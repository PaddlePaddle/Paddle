/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/cross_entropy_op.h"
#include <string>
#include "paddle/fluid/operators/cross_entropy_op_base.h"

namespace paddle {
namespace operators {

class CrossEntropyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), a tensor whose last dimension "
             "size is equal to the number of classes. This input is a "
             "probability computed by the previous operator, which is almost "
             "always the result of a softmax operator.");
    AddInput(
        "Label",
        "(Tensor), the tensor which represents the ground truth. It has the "
        "same shape with 'X' except the last dimension. When soft_label is set "
        "to false, the last dimension size is 1; when soft_label is set to "
        "true, the last dimension size is equal to the number of classes.");
    AddOutput("Y",
              "(Tensor, default Tensor<float>), a tensor whose shape is same "
              "with 'X' except that the last dimension size is 1. It "
              "represents the cross entropy loss.");
    AddAttr<bool>("soft_label",
                  "(bool, default false), a flag indicating whether to "
                  "interpretate the given labels as soft labels.")
        .SetDefault(false);
    AddAttr<int>("ignore_index",
                 "(int, default -100), Specifies a target value that is"
                 "ignored and does not contribute to the input gradient."
                 "Only valid if soft_label is set to False")
        .SetDefault(-100);
    AddComment(R"DOC(
CrossEntropy Operator.

The input 'X' and 'Label' will first be logically flattened to 2-D matrixs. 
The matrix's second dimension(row length) is as same as the original last 
dimension, and the first dimension(column length) is the product of all other 
original dimensions. Then the softmax computation will take palce on each raw 
of flattened matrixs.

It supports both standard cross-entropy and soft-label cross-entropy loss
computation.
1) One-hot cross-entropy:
    soft_label = false, Label[i, 0] indicates the class index for sample i:

                $Y[i] = -\log(X[i, Label[i]])$

2) Soft-label cross-entropy:
    soft_label = true, Label[i, j] indicates the soft label of class j
    for sample i:

                $Y[i] = \sum_j{-Label[i, j] * log(X[i, j])}$

   Please make sure that in this case the summuation of each row of Label
   equals one.

3) One-hot cross-entropy with vecterized Input(Label):
     As a special case of 2), when each row of Input(Label) has only one
     non-zero element (equals 1), soft-label cross-entropy degenerates to a
     one-hot cross-entropy with one-hot label representation.

Both the input X and Label can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input X.

)DOC");
  }
};

class CrossEntropyGradientOp : public CrossEntropyGradientOpBase {
 public:
  using CrossEntropyGradientOpBase::CrossEntropyGradientOpBase;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    CrossEntropyGradientOpBase::InferShape(ctx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPUCtx = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(cross_entropy, ops::CrossEntropyOpBase,
                  ops::CrossEntropyOpMaker, ops::CrossEntropyOpInferVarType,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(cross_entropy_grad, ops::CrossEntropyGradientOp);
REGISTER_OP_CPU_KERNEL(cross_entropy, ops::CrossEntropyOpKernel<CPUCtx, float>,
                       ops::CrossEntropyOpKernel<CPUCtx, double>);
REGISTER_OP_CPU_KERNEL(cross_entropy_grad,
                       ops::CrossEntropyGradientOpKernel<CPUCtx, float>,
                       ops::CrossEntropyGradientOpKernel<CPUCtx, double>);
