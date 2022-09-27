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

#include "paddle/fluid/operators/seed_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class SeedOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim("Out", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::INT32,
                                   ctx.device_context());
  }
};

class SeedOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "The output of seed op.");
    AddAttr<int>("seed", "Dropout random seed.").SetDefault(0);
    AddComment(R"DOC(
Seed Operator.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    seed,
    ops::SeedOp,
    ops::SeedOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(seed, ops::CPUSeedKernel<phi::CPUContext, int>);

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(seed).AddCheckpoint(
    R"ROC(
             Upgrade seed add a new attribute [force_cpu])ROC",
    paddle::framework::compatible::OpVersionDesc().NewAttr(
        "force_cpu",
        "If true, Force fill output variable to cpu."
        "memory. Otherwise, fill output variable to the running "
        "device",
        false));
