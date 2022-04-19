// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/optimizers/pow2_decay_with_linear_warmup_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

class Pow2DecayWithLinearWarmupOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    auto dim = phi::make_ddim({1});
    ctx->SetOutputDim("LearningRateOut", dim);
    ctx->SetOutputDim("StepOut", dim);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "LearningRate");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class Pow2DecayWithLinearWarmupOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("LearningRate", "(Tensor) The input learning rate Tensor.");
    AddInput("Step", "(Tensor) The input global step Tensor.");
    AddOutput("LearningRateOut",
              "(Tensor) The output learning rate Tensor. Same with "
              "Input(LearningRate).");
    AddOutput(
        "StepOut",
        "(Tensor) The output learning rate Tensor. Same with Input(Step).");
    AddAttr<int64_t>("warmup_steps", "(int64_t) The warmup steps.");
    AddAttr<int64_t>(
        "total_steps",
        "(int64_t) The total steps for changing the learning rate.");
    AddAttr<float>("base_lr",
                   "(float) The final learning rate value after warmup.");
    AddAttr<float>("end_lr",
                   "(float) The final learning rate value after total_steps.");
    AddComment(R"DOC(
The Pow2DecayWithLinearWarmup learning rate scheduler.

When step_num < warmup_steps, lr = base_lr * step_num / warmup_steps 

When warmup_steps <= step_num <= total_steps, 
   factor = 1 - (step_num - warmup_steps) / (total_steps - warmup_steps) 
   lr = (base_lr - end_lr) * factor * factor + end_lr 

When step_num > total_steps, lr = end_lr

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(pow2_decay_with_linear_warmup,
                             ops::Pow2DecayWithLinearWarmupOp,
                             ops::Pow2DecayWithLinearWarmupOpMaker);
REGISTER_OP_CPU_KERNEL(
    pow2_decay_with_linear_warmup,
    ops::Pow2DecayWithLinearWarmupOpKernel<plat::CPUDeviceContext, double>,
    ops::Pow2DecayWithLinearWarmupOpKernel<plat::CPUDeviceContext, float>);
