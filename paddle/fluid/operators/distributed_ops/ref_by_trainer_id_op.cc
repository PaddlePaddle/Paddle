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

#include "paddle/fluid/operators/distributed_ops/ref_by_trainer_id_op.h"
#include <string>

namespace paddle {
namespace operators {

class RefByTrainerIdOp : public framework::OperatorWithKernel {
 public:
  RefByTrainerIdOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInputs("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of RefByTrainerIdOp should not be null."));

    PADDLE_ENFORCE_EQ(
        ctx->HasInput("TrainerId"), true,
        platform::errors::InvalidArgument(
            "Input(TrainerId) of RefByTrainerIdOp should not be null."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of RefByTrainerIdOp should not be null."));

    PADDLE_ENFORCE_EQ(
        ctx->GetInputDim("TrainerId").size(), 1,
        platform::errors::InvalidArgument("TrainerId should be a scalar."));
    // Out's shape is determined at runtime.
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class RefByTrainerIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor list.").AsDuplicable();
    AddInput("TrainerId", "(Tensor) Scalar int, the trainer id runtime value.");
    AddOutput("Out", "(Tensor) Return one tensor reference of X[trainer_id]");
    AddComment(R"DOC(
**RefByTrainerId operator**

Return a reference of a tensor, using trainer_id as the index to find from the input.

$$Out = X[TrainerId]$$
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(ref_by_trainer_id, ops::RefByTrainerIdOp,
                             ops::RefByTrainerIdOpMaker);
REGISTER_OP_CPU_KERNEL(
    ref_by_trainer_id,
    ops::RefByTrainerIdKernel<paddle::platform::CPUDeviceContext, float>,
    ops::RefByTrainerIdKernel<paddle::platform::CPUDeviceContext, double>,
    ops::RefByTrainerIdKernel<paddle::platform::CPUDeviceContext, int>,
    ops::RefByTrainerIdKernel<paddle::platform::CPUDeviceContext, int64_t>);
