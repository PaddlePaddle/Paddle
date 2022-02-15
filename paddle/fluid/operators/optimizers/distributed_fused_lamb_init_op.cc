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

    AddOutput(
        "FP32FusedParam",
        "The fp32 fused param and fp16 master weight param tensor. Its "
        "shape is [M1+M2], where M1 is the fp32 fused parameter size and "
        "M2 is the fp16 fused master weight parameter size. Note that M1 "
        "and M2 should be exactly divided by N, where N is the world size.")
        .AsDispensable();
    AddOutput("FP32FusedGrad", "The fp32 fused grad tensor. Its shape is [M1].")
        .AsDispensable();
    AddOutput("FP16FusedParam",
              "The fp16 fused param tensor. Its shape is [M2].")
        .AsDispensable();
    AddOutput("FP16FusedGrad",
              "The fp16 fused param tensors. Its shape is [M2].")
        .AsDispensable();

    AddOutput("Moment1", "The fp32 moment1 tensor. Its shape is [(M1+M2)/N].");
    AddOutput("Moment2", "The fp32 moment2 tensor. Its shape is [(M1+M2)/N].");
    AddOutput("Beta1Pow",
              "The fp32 beta1 power accumulator tensor. Its shape is [1].");
    AddOutput("Beta2Pow",
              "The fp32 beta2 power accumulator tensor. Its shape is [1].");
    AddOutput("FusedIndices",
              "The param index of each element in FP32FusedParam. Its shape is "
              "[M1+M2].");
    AddOutput("FusedParamOffsets",
              "The parameter offset of the fused parameters. Its shape is "
              "[param_num + 1].");
    AddOutput(
        "FP32PartialFusedParamOffsets",
        "The partial parameter offset of the fused FP32 parameters. Its shape "
        "is [fp32_local_param_num + 1].");
    AddOutput(
        "FP16PartialFusedParamOffsets",
        "The partial parameter offset of the fused FP16 parameters. Its shape "
        "is [fp16_local_param_num + 1].");
    /*
    AddOutput("FP32PartialFusedParamOffsetsEx", "");
    AddOutput("FP16PartialFusedParamOffsetsEx", "");
    */
    AddOutput("WeightDecay",
              "The fp32 weight decay tensor. Its shape is [(M1+M2)/N].");
    AddOutput("LocalParamInfo",
              "The local param info inside FP32FusedParam. It should be in "
              "CPUPlace, and its shape is [4].");

    AddOutput("ParamOut", "The output parameter list.").AsDuplicable();
    AddOutput("MasterParamOut",
              "The output master parameter list. It would share the memory of "
              "each FP32 parameter and FP16 master parameter.")
        .AsDuplicable();
    AddOutput("GradOut", "The output gradient list.").AsDuplicable();
    AddOutput("GlobalScale",
              "The global scale. It is usually the scale factor for AMP.");

    AddAttr<float>("beta1", "The initial value of Beta1Pow.");
    AddAttr<float>("beta2", "The initial value of Beta2Pow.");
    AddAttr<std::vector<float>>("weight_decay",
                                "The weight decay for each parameter. Its "
                                "shape is equal to the parameter number.");
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
