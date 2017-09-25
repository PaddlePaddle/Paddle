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

#include "paddle/operators/sequence_avg_pool_op.h"

namespace paddle {
namespace operators {

class SequenceAvgPoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContextBase& ctx) const override {
    PADDLE_ENFORCE(ctx.HasInput("X"),
                   "Input(X) of SequenceAvgPoolOp should not be null.");
    PADDLE_ENFORCE(ctx.HasOutput("Out"),
                   "Output(Out) of SequenceAvgPoolOp should not be null.");
    ctx.SetOutputDim("Out", ctx.GetInputDim("X"));
  }
};

class SequenceAvgPoolOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequenceAvgPoolOpMaker(framework::OpProto* proto,
                         framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input of SequenceAvgPoolOp.");
    AddOutput("Out", "The output of SequenceAvgPoolOp.");
    AddComment(R"DOC(
    SequenceAvgPoolOp averages features of all time-steps of each instance.
    More detailed comments will be added later.
    )DOC");
  }
};

class SequenceAvgPoolGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContextBase& ctx) const override {
    PADDLE_ENFORCE(ctx.HasInput(framework::GradVarName("Out")),
                   "Gradient of Out should not be null.");
    PADDLE_ENFORCE(ctx.HasInput("X"), "The input X should not be null.");
    auto og_dims = ctx.GetInputDim(framework::GradVarName("Out"));
    auto x_dims = ctx.GetInputDim("X");
    PADDLE_ENFORCE_EQ(og_dims.size(), x_dims.size(),
                      "The rank of output grad must equal to Input(X).");
    for (int64_t i = 1; i < og_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(og_dims[i], x_dims[i], "The dimension mismatch.");
    }
    ctx.SetOutputDim(framework::GradVarName("X"), x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sequence_avg_pool, ops::SequenceAvgPoolOp,
            ops::SequenceAvgPoolOpMaker, sequence_avg_pool_grad,
            ops::SequenceAvgPoolGradOp);
REGISTER_OP_CPU_KERNEL(
    sequence_avg_pool,
    ops::SequenceAvgPoolKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    sequence_avg_pool_grad,
    ops::SequenceAvgPoolGradKernel<paddle::platform::CPUPlace, float>);
