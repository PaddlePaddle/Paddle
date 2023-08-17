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

#include <string>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class DGCMomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("current_step"),
                   "Input",
                   "current_step",
                   "DGCMomentumOp");
    OP_INOUT_CHECK(ctx->HasInput("nranks"), "Input", "nranks", "DGCMomentumOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Grad_out"), "Output", "Grad_out", "DGCMomentumOp");

    PADDLE_ENFORCE_EQ(ctx->HasInput("Param"),
                      true,
                      platform::errors::NotFound(
                          "Input(param) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Grad"),
                      true,
                      platform::errors::NotFound(
                          "Input(grad) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Velocity"),
                      true,
                      platform::errors::NotFound(
                          "Input(velocity) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("LearningRate"),
        true,
        platform::errors::NotFound(
            "Input(LearningRate) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(ctx->GetInputsVarType("Param").front(),
                      framework::proto::VarType::LOD_TENSOR,
                      platform::errors::InvalidArgument(
                          "The input var's type should be phi::DenseTensor, "
                          "but the received is %s",
                          ctx->GetInputsVarType("Param").front()));

    PADDLE_ENFORCE_EQ(ctx->HasOutput("ParamOut"),
                      true,
                      platform::errors::NotFound(
                          "Output(ParamOut) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("VelocityOut"),
        true,
        platform::errors::NotFound(
            "Output(VelocityOut) of Momentum should not be null."));

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(phi::product(lr_dims),
                      0,
                      platform::errors::InvalidArgument(
                          "Maybe the Input variable LearningRate has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function."));
    PADDLE_ENFORCE_EQ(phi::product(lr_dims),
                      1,
                      platform::errors::InvalidArgument(
                          "Learning_rate should be a scalar. But Received "
                          "LearningRate's dim [%s]",
                          phi::product(lr_dims)));

    auto param_dim = ctx->GetInputDim("Param");
    if (ctx->GetInputsVarType("Grad")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          param_dim,
          ctx->GetInputDim("Grad"),
          platform::errors::InvalidArgument(
              "Param and Grad input of MomentumOp should have the same "
              "dimension. But received Param's dim [%s] and Grad's dim [%s].",
              param_dim,
              ctx->GetInputDim("Grad")));
      PADDLE_ENFORCE_EQ(
          param_dim,
          ctx->GetInputDim("Velocity"),
          platform::errors::InvalidArgument(
              "Param and Velocity of MomentumOp should have the same "
              "dimension. But received Param's dim [%s] and Velocity [%s].",
              param_dim,
              ctx->GetInputDim("Velocity")));
    }

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("VelocityOut", param_dim);
    if (ctx->HasOutput("MasterParamOut")) {
      ctx->SetOutputDim("MasterParamOut", param_dim);
    }
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    if (var_name == "current_step" || var_name == "nranks") {
      VLOG(10) << "var_name:" << var_name << " need not to transform";
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }

    return framework::OperatorWithKernel::GetKernelTypeForVar(
        var_name, tensor, expected_kernel_type);
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class DGCMomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input parameter that has to be updated");
    AddInput("Grad",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input gradient of the parameter");
    AddInput("Velocity",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input velocity (corresponding to the parameter) "
             "that has to be updated");
    AddInput("LearningRate",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input learning rate");
    AddInput("MasterParam", "FP32 master weight for AMP.").AsDispensable();
    AddInput("current_step", "(Tensor) Current step.");
    AddInput("nranks", "(Tensor) The number of trainers.");

    AddOutput("ParamOut",
              "(phi::DenseTensor) This output is updated parameter. "
              "It shared memory with Input(Param).");
    AddOutput("VelocityOut",
              "(phi::DenseTensor) This output is updated velocity. "
              "It shared memory with Input(Velocity).");
    AddOutput("MasterParamOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParam).")
        .AsDispensable();
    AddOutput("Grad_out", "(Tensor) Output grad gradient");

    AddAttr<float>("mu", "(float) Momentum coefficient");
    AddAttr<bool>("use_nesterov",
                  "(bool, default false) "
                  "Use Nesterov Momentum")
        .SetDefault(false);
    AddAttr<std::string>("regularization_method",
                         "(string) regularization_method, right now only "
                         "support l2decay or none")
        .SetDefault("");
    AddAttr<float>("regularization_coeff", "(float) regularization_coeff")
        .SetDefault(0.0f);
    AddAttr<bool>("multi_precision",
                  "(bool, default false) "
                  "Whether to use multi-precision during weight updating.")
        .SetDefault(false);
    AddAttr<float>(
        "rescale_grad",
        "(float, default 1.0) Multiply the gradient with `rescale_grad`"
        "before updating. Often choose to be `1.0/batch_size`.")
        .SetDefault(1.0f);
    AddAttr<float>("rampup_begin_step",
                   "(float, -1.0)"
                   "The period when begin DGC.")
        .SetDefault(-1.0);

    AddComment(R"DOC(
DGC Momentum Operator.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(dgc_momentum,
                             ops::DGCMomentumOp,
                             ops::DGCMomentumOpMaker);
