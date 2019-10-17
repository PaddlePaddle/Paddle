// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/sync_fused_tensor_op.h"

namespace paddle {
namespace operators {

class SyncFusedTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInputs("Input"), true,
                      "Input(Input) of SyncFusedTensorOp should not be null.");
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("FusedInput"), true,
        "Input(FusedInput) of SyncFusedTensorOp should not be null.");
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("FusedOutput"), true,
        "Output(FusedOutput) of SyncFusedTensorOp should not be null.");
  }
};

class SyncFusedTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(vector<LoDTensor>) The input tensors of SyncFusedTensor operator.")
        .AsDuplicable();
    AddInput("FusedInput",
             "(LoDTensor) The fuesd input tensor of SyncFusedTensor operator.");
    AddOutput(
        "FusedOutput",
        "(LoDTensor) The fused ouput tensor of SyncFusedTensor operator.");
    AddComment(R"DOC(
SyncFusedTensor Operator.

Synchronize the input tensors to the fused output tensor. When the developer fused
the outputs of some operators to a contiguous address on the GPU device, since these
operators may not have a GPU kernel, the output gerarated cannot be written to the
GPU device as expected, but generated separately on the CPU device, which will cause
the program to use dirty data in GPU device to continue training. When this happens,
this operator will synchronize the output on the CPU device to the GPU device to
ensure the correctness of data.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(sync_fused_tensor, paddle::operators::SyncFusedTensorOp,
                  paddle::operators::SyncFusedTensorOpMaker);
namespace ops = paddle::operators;
REGISTER_OP_CPU_KERNEL(
    sync_fused_tensor,
    ops::SyncFusedTensorOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SyncFusedTensorOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SyncFusedTensorOpKernel<paddle::platform::CPUDeviceContext, double>);
