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

#include "paddle/fluid/operators/optimizers/distributed_fused_lamb_init_op.h"

namespace paddle {
namespace operators {

class DistributedFusedLambInitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {}

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto dtype = framework::proto::VarType::FP32;  // dtype is not important
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

class DistributedFusedLambInitOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "The initial parameter list.").AsDuplicable();
    AddInput("Grad", "The initial gradient list.").AsDuplicable();

    AddOutput("FP32FusedParam",
              "The fp32 fused param and fp16 fused master weight tensor. Its "
              "shape is [M1+M2], where M1 is the fp32 fused parameter size and "
              "M2 is the fp16 fused master weight parameter size. Note that M1 "
              "and M2 should be exactly divided by N (guaranteed by extra "
              "padding 0), where N is the world size.")
        .AsDispensable();
    AddOutput("FP32FusedGrad", "The fp32 fused grad tensor. Its shape is [M1].")
        .AsDispensable();
    AddOutput("FP16FusedParam",
              "The fp16 fused param tensor. Its shape is [M2].")
        .AsDispensable();
    AddOutput("FP16FusedGrad", "The fp16 fused grad tensor. Its shape is [M2].")
        .AsDispensable();

    AddOutput("Moment1",
              "The sharded fp32 moment1 tensor. Its shape is [(M1+M2)/N].");
    AddOutput("Moment2",
              "The sharded fp32 moment2 tensor. Its shape is [(M1+M2)/N].");
    AddOutput("Beta1Pow",
              "The fp32 beta1 power accumulator tensor. Its shape is [1].");
    AddOutput("Beta2Pow",
              "The fp32 beta2 power accumulator tensor. Its shape is [1].");
    AddOutput(
        "FusedParamOffsets",
        "The numel offset of each parameter inside the FP32FusedParam. Its "
        "shape is [param_num + 1]. It is like [0, n_0, n_0 + n_1, n_0 + n_1 "
        "+ n_2, ...]. It should be in CPUPlace.");
    AddOutput(
        "FP32ShardFusedParamOffsets",
        "The sharded numel offset of each parameter in the local rank. "
        "Its shape is [fp32_local_param_num + 1]. It should be in CPUPlace.");
    AddOutput(
        "FP16ShardFusedParamOffsets",
        "The sharded numel offset of each parameter in the local rank. "
        "Its shape is [fp16_local_param_num + 1]. It should be in CPUPlace.");
    AddOutput("ParamInfo",
              "The param info. It should be in CPUPlace, and its shape is [6]"
              "CPUPlace, and its shape is [8]. It is "
              "[fp32_shard_param_start_idx, fp32_local_param_num, "
              "fp32_global_param_num, fp32_weight_decay_end_idx, "
              "fp16_shard_param_start_idx, "
              "fp16_local_param_num, fp16_global_param_num, "
              "fp16_weight_decay_end_idx].");
    AddOutput("ParamOrder",
              "The reordered parameter order. Inside this op, "
              "the parameter would be reordered by data type and weight decay "
              "value.");
    AddOutput("ParamOut", "The output parameter list.").AsDuplicable();
    AddOutput("MasterParamOut",
              "The output master parameter list. It would share the memory of "
              "each fp32 parameter and fp16 master parameter.")
        .AsDuplicable();
    AddOutput("GradOut", "The output gradient list.").AsDuplicable();
    AddOutput("GlobalScale",
              "The global scale. It is usually the scale factor for AMP.");

    AddAttr<float>("beta1", "The initial value of Beta1Pow.");
    AddAttr<float>("beta2", "The initial value of Beta2Pow.");
    AddAttr<std::vector<int>>("apply_weight_decay",
                              "Whether to apply weight decay.");
    AddAttr<int>("alignment", "The alignment in bytes for the fused tensors.");
    AddAttr<int>("rank", "The global rank of the current process.");
    AddAttr<int>("nranks", "The global world size.");
    AddComment(
        R"DOC(The init operator for the DistributedFusedLamb optimizer.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(distributed_fused_lamb_init,
                             ops::DistributedFusedLambInitOp,
                             ops::DistributedFusedLambInitOpMaker);

REGISTER_OP_CPU_KERNEL(
    distributed_fused_lamb_init,
    ops::DistributedFusedLambInitOpKernel<plat::CPUDeviceContext, float>);
