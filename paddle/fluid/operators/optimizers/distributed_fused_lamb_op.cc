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
    if (var_name == "LocalParamInfo") {
      return expected_kernel_type;
    } else {
      return framework::OperatorWithKernel::GetKernelTypeForVar(
          var_name, tensor, expected_kernel_type);
    }
  }
};

class DistributedFusedLambOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "The original input parameter tensor list.")
        .AsDuplicable();
    AddInput("Grad", "The original input gradient tensor list.").AsDuplicable();
    AddInput(
        "FP32FusedParam",
        "The fp32 fused param and fp16 master weight param tensor. Its "
        "shape is [M1+M2], where M1 is the fp32 fused parameter size and "
        "M2 is the fp16 fused master weight parameter size. Note that M1 "
        "and M2 should be exactly divided by N, where N is the world size.")
        .AsDispensable();
    AddInput("FP32FusedGrad", "The fp32 fused grad tensor. Its shape is [M1].")
        .AsDispensable();
    AddInput("FP16FusedParam",
             "The fp16 fused param tensor. Its shape is [M2].")
        .AsDispensable();
    AddInput("FP16FusedGrad",
             "The fp16 fused param tensors. Its shape is [M2].")
        .AsDispensable();

    AddInput("LearningRate",
             "The fp32 learning rate tensor. Its shape is [1].");
    AddInput("Moment1", "The fp32 moment1 tensor. Its shape is [(M1+M2)/N].");
    AddInput("Moment2", "The fp32 moment2 tensor. Its shape is [(M1+M2)/N].");
    AddInput("Beta1Pow",
             "The fp32 beta1 power accumulator tensor. Its shape is [1].");
    AddInput("Beta2Pow",
             "The fp32 beta2 power accumulator tensor. Its shape is [1].");
    AddInput("FusedIndices",
             "The param index of each element in FP32FusedParam. Its shape is "
             "[M1+M2].");
    AddInput("WeightDecay",
             "The fp32 weight decay tensor. Its shape is [(M1+M2)/N].");
    AddInput("GlobalScale", "The fp32 global scale tensor. Its shape is [1].");
    AddInput("LocalParamInfo",
             "The local param info inside FP32FusedParam. It should be in "
             "CPUPlace, and its shape is [4].");
    AddInput("FusedParamOffsets",
             "The parameter offset of the fused parameters.");
    AddInput("PartialFusedParamOffsets",
             "The partial parameter offset of the fused parameters.");

    AddOutput("FP32FusedParamOut", "The updated FP32FusedParam.")
        .AsDispensable();
    AddOutput("FP16FusedParamOut", "The updated FP16FusedParam.")
        .AsDispensable();

    AddOutput("Moment1Out", "The updated Moment1.");
    AddOutput("Moment2Out", "The updated Moment2.");
    AddOutput("Beta1PowOut", "The updated Beta1Pow.");
    AddOutput("Beta2PowOut", "The updated Beta2Pow.");

    AddOutput("ParamOut", "The original output parameter tensor list.")
        .AsDuplicable();
    AddOutput("GradOut", "The original output gradient tensor list.")
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
    AddAttr<bool>("clip_after_allreduce",
                  "Whether to clip before allreduce, only valid when the "
                  "world size is larger than 1.");
    AddAttr<bool>(
        "broadcast_master_param",
        "Whether to broadcast master parameter or FP16 parameter. It is only "
        "useful when there is any FP16 parameter. If it is true, the master "
        "weight would be updated, broadcast and cast to be FP16 parameter. "
        "If it is false, the FP16 parameter would be updated, broadcast and "
        "cast to be master weight.");
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
