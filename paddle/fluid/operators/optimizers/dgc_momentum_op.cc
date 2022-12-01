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

#include "paddle/fluid/operators/optimizers/dgc_momentum_op.h"

#include <string>

namespace paddle {
namespace operators {

class DGCMomentumOp : public MomentumOp {
 public:
  using MomentumOp::MomentumOp;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("current_step"),
                   "Input",
                   "current_step",
                   "DGCMomentumOp");
    OP_INOUT_CHECK(ctx->HasInput("nranks"), "Input", "nranks", "DGCMomentumOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Grad_out"), "Output", "Grad_out", "DGCMomentumOp");
    return MomentumOp::InferShape(ctx);
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "current_step" || var_name == "nranks") {
      VLOG(10) << "var_name:" << var_name << " need not to transform";
      return expected_kernel_type;
    }

    return framework::OperatorWithKernel::GetKernelTypeForVar(
        var_name, tensor, expected_kernel_type);
  }
};

class DGCMomentumOpMaker : public MomentumOpMaker {
 public:
  void Make() override {
    AddInput("current_step", "(Tensor) Current step.");
    AddInput("nranks", "(Tensor) The number of trainers.");

    AddOutput("Grad_out", "(Tensor) Output grad gradient");

    AddAttr<float>("rampup_begin_step",
                   "(float, -1.0)"
                   "The period when begin DGC.")
        .SetDefault(-1.0);

    return MomentumOpMaker::Make();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(dgc_momentum,
                             ops::DGCMomentumOp,
                             ops::DGCMomentumOpMaker);

REGISTER_OP_CPU_KERNEL(dgc_momentum,
                       ops::DGCMomentumKernel<phi::CPUContext, float>);
