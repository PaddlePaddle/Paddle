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

#include "paddle/fluid/operators/optimizers/distributed_fused_lamb_op.h"

namespace paddle {
namespace operators {

class DistributedFusedLambOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {}

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto dtype = framework::proto::VarType::FP32;  // dtype is not important
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return expected_kernel_type;
  }
};

class DistributedFusedLambOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "The initial parameter list.").AsDuplicable();
    AddInput("Grad", "The initial gradient list.").AsDuplicable();

    AddInput("FP32FusedParam",
             "The fp32 fused param and fp16 fused master weight tensor. Its "
             "shape is [M1+M2], where M1 is the fp32 fused parameter size and "
             "M2 is the fp16 fused master weight parameter size. Note that M1 "
             "and M2 should be exactly divided by N (guaranteed by extra "
             "padding 0), where N is the world size.")
        .AsDispensable();
    AddInput("FP32FusedGrad", "The fp32 fused grad tensor. Its shape is [M1].")
        .AsDispensable();
    AddInput("FP16FusedParam",
             "The fp16 fused param tensor. Its shape is [M2].")
        .AsDispensable();
    AddInput("FP16FusedGrad", "The fp16 fused grad tensor. Its shape is [M2].")
        .AsDispensable();

    AddInput("Moment1",
             "The sharded fp32 moment1 tensor. Its shape is [(M1+M2)/N].");
    AddInput("Moment2",
             "The sharded fp32 moment2 tensor. Its shape is [(M1+M2)/N].");
    AddInput("Beta1Pow",
             "The fp32 beta1 power accumulator tensor. Its shape is [1].");
    AddInput("Beta2Pow",
             "The fp32 beta2 power accumulator tensor. Its shape is [1].");
    AddInput(
        "FusedParamOffsets",
        "The numel offset of each parameter inside the FP32FusedParam. Its "
        "shape is [param_num + 1]. It is like [0, n_0, n_0 + n_1, n_0 + n_1 "
        "+ n_2, ...]. It should be in CPUPlace.");
    AddInput(
        "FP32ShardFusedParamOffsets",
        "The sharded numel offset of each parameter in the local rank. "
        "Its shape is [fp32_local_param_num + 1]. It should be in CPUPlace.");
    AddInput(
        "FP16ShardFusedParamOffsets",
        "The sharded numel offset of each parameter in the local rank. "
        "Its shape is [fp16_local_param_num + 1]. It should be in CPUPlace.");
    AddInput("ParamInfo",
             "The param info. It should be in CPUPlace, and its shape is [6]"
             "CPUPlace, and its shape is [8]. It is "
             "[fp32_shard_param_start_idx, fp32_local_param_num, "
             "fp32_global_param_num, fp32_weight_decay_end_idx, "
             "fp16_shard_param_start_idx, "
             "fp16_local_param_num, fp16_global_param_num, "
             "fp16_weight_decay_end_idx].");
    AddInput("ParamOrder",
             "The reordered parameter order. Inside this op, "
             "the parameter would be reordered by data type and weight decay "
             "value.");

    AddInput("LearningRate",
             "The fp32 learning rate tensor. Its shape is [1].");
    AddInput("GlobalScale", "The fp32 global scale tensor. Its shape is [1].");

    AddOutput("FP32FusedParamOut", "The updated FP32FusedParam.")
        .AsDispensable();
    AddOutput("FP16FusedParamOut", "The updated FP16FusedParam.")
        .AsDispensable();

    AddOutput("Moment1Out", "The updated Moment1.");
    AddOutput("Moment2Out", "The updated Moment2.");
    AddOutput("Beta1PowOut", "The updated Beta1Pow.");
    AddOutput("Beta2PowOut", "The updated Beta2Pow.");

    AddOutput("ParamOut", "The updated output parameter tensor list.")
        .AsDuplicable();

    AddOutput("FoundInf", "Whether there is NaN/Inf");

    AddAttr<float>("beta1", "The initial Beta1Pow value.");
    AddAttr<float>("beta2", "The initial Beta2Pow value.");
    AddAttr<float>("epsilon",
                   "The epsilon value to maintain numeric stability.");
    AddAttr<float>(
        "max_global_grad_norm",
        "The maximum global gradient l2-norm value for clipping. If "
        "max_global_grad_norm <= 0, no clipping would be performed.");
    AddAttr<float>("weight_decay", "The weight decay value.");
    AddAttr<bool>("clip_after_allreduce",
                  "Whether to clip before allreduce, only valid when the "
                  "world size is larger than 1.");
    AddAttr<bool>(
        "use_master_param_norm",
        "Whether to use master parameter to calculate "
        "the L2-Norm. If it is true, it would be more accurate but be more "
        "NCCL communication data. If it is false, it would be less accurate "
        "and be less NCCL communication data.")
        .SetDefault(true);
    AddAttr<bool>("is_grad_scaled_by_nranks",
                  "Whether the input gradient has been scaled by nranks.")
        .SetDefault(true);
    AddAttr<int>("ring_id", "The ring id of the NCCL communicator.")
        .SetDefault(0);
    AddComment("The DistributedFusedLamb optimizer.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(distributed_fused_lamb,
                             ops::DistributedFusedLambOp,
                             ops::DistributedFusedLambOpMaker);

REGISTER_OP_CPU_KERNEL(
    distributed_fused_lamb,
    ops::DistributedFusedLambOpKernel<plat::CPUDeviceContext, float>);
