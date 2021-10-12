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

#include "paddle/fluid/operators/merged_momentum_op.h"

namespace paddle {
namespace operators {

class MergedMomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {}

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto param_dtype =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(param_dtype, ctx.GetPlace());
  }
};

class MergedMomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "").AsDuplicable();
    AddInput("Grad", "").AsDuplicable();
    AddInput("Velocity", "").AsDuplicable();
    AddInput("LearningRate", "");
    AddInput("MasterParam", "").AsDispensable().AsDuplicable();
    AddOutput("ParamOut", "").AsDuplicable();
    AddOutput("VelocityOut", "").AsDuplicable();
    AddOutput("MasterParamOut", "").AsDispensable().AsDuplicable();
    AddAttr<float>("mu", "");
    AddAttr<bool>("multi_precision", "").SetDefault(false);
    AddAttr<float>("rescale_grad", "").SetDefault(1.0f);
    AddComment(R"DOC(Merged Momentum Optimizer.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(merged_momentum, ops::MergedMomentumOp,
                             ops::MergedMomentumOpMaker);

REGISTER_OP_CPU_KERNEL(
    merged_momentum, ops::MergedMomentumOpKernel<plat::CPUDeviceContext, float>,
    ops::MergedMomentumOpKernel<plat::CPUDeviceContext, double>);
