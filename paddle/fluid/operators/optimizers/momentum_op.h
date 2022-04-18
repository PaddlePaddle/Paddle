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

#pragma once
#include <memory>
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/algorithm.h"

namespace paddle {
namespace operators {

class MomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class MomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Param"), true,
                      platform::errors::NotFound(
                          "Input(param) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Grad"), true,
                      platform::errors::NotFound(
                          "Input(grad) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Velocity"), true,
                      platform::errors::NotFound(
                          "Input(velocity) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("LearningRate"), true,
        platform::errors::NotFound(
            "Input(LearningRate) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("Param").front(),
        framework::proto::VarType::LOD_TENSOR,
        platform::errors::InvalidArgument(
            "The input var's type should be LoDTensor, but the received is %s",
            ctx->GetInputsVarType("Param").front()));

    PADDLE_ENFORCE_EQ(ctx->HasOutput("ParamOut"), true,
                      platform::errors::NotFound(
                          "Output(ParamOut) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("VelocityOut"), true,
        platform::errors::NotFound(
            "Output(VelocityOut) of Momentum should not be null."));

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(phi::product(lr_dims), 0,
                      platform::errors::InvalidArgument(
                          "Maybe the Input variable LearningRate has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function."));
    PADDLE_ENFORCE_EQ(phi::product(lr_dims), 1,
                      platform::errors::InvalidArgument(
                          "Learning_rate should be a scalar. But Received "
                          "LearningRate's dim [%s]",
                          phi::product(lr_dims)));

    auto param_dim = ctx->GetInputDim("Param");
    if (ctx->GetInputsVarType("Grad")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          param_dim, ctx->GetInputDim("Grad"),
          platform::errors::InvalidArgument(
              "Param and Grad input of MomentumOp should have the same "
              "dimension. But received Param's dim [%s] and Grad's dim [%s].",
              param_dim, ctx->GetInputDim("Grad")));
      PADDLE_ENFORCE_EQ(
          param_dim, ctx->GetInputDim("Velocity"),
          platform::errors::InvalidArgument(
              "Param and Velocity of MomentumOp should have the same "
              "dimension. But received Param's dim [%s] and Velocity [%s].",
              param_dim, ctx->GetInputDim("Velocity")));
    }

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("VelocityOut", param_dim);
    if (ctx->HasOutput("MasterParamOut")) {
      ctx->SetOutputDim("MasterParamOut", param_dim);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle
