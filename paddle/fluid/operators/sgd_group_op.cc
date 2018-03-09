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

#include "paddle/fluid/operators/sgd_group_op.h"

namespace paddle {
namespace operators {

class SGDGroupOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("Params"),
                   "Inputs(Param) of SGDGroupOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInputs("Grads"),
                   "Inputs(Grad) of SGDGroupOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInputs("LearningRates"),
                   "Inputs(LearningRates) of SGDGroupOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutputs("ParamOuts"),
                   "Outputs(ParamOut) of SGDGroupOp should not be null.");

    auto params = ctx->GetInputsDim("Params");
    auto grads = ctx->GetInputsDim("Grads");
    auto learning_rates = ctx->GetInputsDim("LearningRates");

    auto param_num = params.size();

    PADDLE_ENFORCE_EQ(param_num, grads.size(),
                      "The number of param and grads should be equal.");
    PADDLE_ENFORCE_EQ(
        param_num, learning_rates.size(),
        "The number of param and learning_rates should be equal.");

    for (size_t i = 0; i < param_num; ++i) {
      PADDLE_ENFORCE_EQ(framework::product(learning_rates[i]), 1,
                        "Learning rate should have 1 element");
    }

    auto param_dims = ctx->GetInputsDim("Params");
    // TODO(qijun): check dimensions of Param and Grad at complie
    // and run time.
    ctx->SetOutputsDim("ParamOuts", param_dims);
  }
};

class SGDGroupOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SGDGroupOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Params", "(vector<Tensor>) Input parameter").AsDuplicable();
    AddInput("LearningRates", "(vector<Tensor>) Learning rate of SGD")
        .AsDuplicable();
    AddInput("Grads", "(vector<Tensor>) Input gradient").AsDuplicable();
    AddOutput("ParamOuts", "(vector<Tensor>) Output parameter").AsDuplicable();
    AddComment(R"DOC(
SGDGroup operator

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(sgd_group, ops::SGDGroupOp, ops::SGDGroupOpMaker);
REGISTER_OP_CPU_KERNEL(sgd_group, ops::SGDGroupOpKernel<float>,
                       ops::SGDGroupOpKernel<double>);
