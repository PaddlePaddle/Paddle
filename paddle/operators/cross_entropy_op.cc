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

class OnehotCrossEntropyOp : public framework::OperatorWithKernel {
protected:
  void InferShapeImpl(const framework::InferContext &ctx) const override {
    PADDLE_ENFORCE(ctx.InputSize() == 2,
                   "Input size of OnehotCrossEntropyOp must be two");
    PADDLE_ENFORCE(ctx.OutputSize() == 1,
                   "Output size of OnehotCrossEntropyOp must be one");
    PADDLE_ENFORCE(ctx.InputVar(0) != nullptr && ctx.InputVar(1) != nullptr,
                   "Inputs of OnehotCrossEntropyOp must all be set");
    PADDLE_ENFORCE(ctx.OutputVar(0) != nullptr,
                   "Outputs of OnehotCrossEntropyOp must all be set");
    PADDLE_ENFORCE(ctx.Input(0).dims().size() == 2, "X's dimension must be 2.");
    PADDLE_ENFORCE(ctx.Output(0)->dims().size() == 1,
                   "label's dimension must be 1.");
    ctx.Output(0)->Resize(framework::make_ddim({ctx.Input(0).dims()[0]}));
  }
};

class OnehotCrossEntropyOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  OnehotCrossEntropyOpMaker(framework::OpProto *proto,
                            framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of OnehotCrossEntropyOp");
    AddInput("label", "The second input of OnehotCrossEntropyOp");
    AddOutput("Y", "The output of OnehotCrossEntropyOp");
    AddComment(R"DOC(
OnehotCrossEntropy Operator.

                Y[i] = -log(X[i][j])

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP(onehot_cross_entropy,
            paddle::operators::OnehotCrossEntropyOp,
            paddle::operators::OnehotCrossEntropyOpMaker);
REGISTER_OP_CPU_KERNEL(
    onehot_cross_entropy,
    paddle::operators::OnehotCrossEntropyOpKernel<::paddle::platform::CPUPlace,
                                                  float>);
