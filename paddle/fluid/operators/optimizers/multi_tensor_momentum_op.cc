/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/optimizers/multi_tensor_momentum_op.h"

namespace paddle {
namespace operators {

class MTMomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Params",
             "(Tensors) The input parameter tensors of multi_tensor_momentum "
             "operator.");
    AddInput(
        "Grads",
        "(Tensors) The input grad tensors of multi_tensor_momentum operator.");
    AddInput("Velocitys",
             "(Tensors, default Tensor<float>) "
             "Input velocity (corresponding to the parameters) "
             "that has to be updated");
    AddInput("LearningRates",
             "(Tensors, default Tensor<float>) "
             "Input learning rate");
    AddInput("MasterParams", "FP32 master weights for AMP.").AsDispensable();
    AddOutput("ParamOuts",
              "(Tensors) This output is updated parameters. "
              "It shared memory with Input(Params).");
    AddOutput("VelocityOuts",
              "(Tensors) This output is updated velocitys. "
              "It shared memory with Input(Velocitys).");
    AddOutput("MasterParamOuts",
              "The updated FP32 master weights for AMP. "
              "It shared memory with Input(MasterParams).")
        .AsDispensable();

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

    AddComment(R"DOC(
    Multi tensor momentum Optimizer.

    This optimizer has a flag for Nestrov Momentum.
    The update equations are as follows:

    $$
    velocity = mu * velocity + gradient \\
    if (use\_nesterov):   \\
    param = param - (gradient + mu * velocity) * learning\_rate \\
    else:   \\
    param = param - learning\_rate * velocity. \\
    $$

    )DOC");
  }
};

class MTMomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Params"), true,
                      platform::errors::NotFound(
                          "Input(params) of MTMomentum should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Grads"), true,
                      platform::errors::NotFound(
                          "Input(grads) of MTMomentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Velocitys"), true,
        platform::errors::NotFound(
            "Input(velocitys) of MTMomentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("LearningRates"), true,
        platform::errors::NotFound(
            "Input(LearningRates) of MTMomentum should not be null."));

    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("Params").front(),
        framework::proto::VarType::LOD_TENSOR,
        platform::errors::InvalidArgument(
            "The input var's type should be LoDTensor, but the received is %s",
            ctx->GetInputsVarType("Params").front()));

    auto lr_dims = ctx->GetInputsDim("LearningRate");
    for (auto idx = 0; idx < lr_dims.size(); idx++) {
      PADDLE_ENFORCE_NE(framework::product(lr_dims[idx]), 0,
                        platform::errors::InvalidArgument(
                            "Maybe the Input variable LearningRate has not "
                            "been initialized. You may need to confirm "
                            "if you put exe.run(startup_program) "
                            "after optimizer.minimize function."));
      PADDLE_ENFORCE_EQ(framework::product(lr_dims[idx]), 1,
                        platform::errors::InvalidArgument(
                            "Learning_rate should be a scalar. But Received "
                            "LearningRate's dim [%s]",
                            framework::product(lr_dims[idx])));
    }

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("ParamOuts"), true,
        platform::errors::NotFound(
            "Output(ParamOuts) of MTMomentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("VelocityOuts"), true,
        platform::errors::NotFound(
            "Output(VelocityOuts) of MTMomentum should not be null."));

    auto param_dims = ctx->GetInputsDim("Params");
    auto grad_dims = ctx->GetInputsDim("Grads");
    auto velocity_dims = ctx->GetInputsDim("Velocitys");
    PADDLE_ENFORCE_EQ(
        param_dims.size(), grad_dims.size(),
        platform::errors::InvalidArgument(
            "The input(Params) and input(Grads) should have same size in "
            "Operator(multi_tensor_momentum), size of input(Params) is %d "
            "and size of input(Grads) is %d.",
            param_dims.size(), grad_dims.size()));

    PADDLE_ENFORCE_EQ(
        param_dims.size(), velocity_dims.size(),
        platform::errors::InvalidArgument(
            "The input(Params) and input(Velocitys) should have same size in "
            "Operator(multi_tensor_momentum), size of input(Params) is %d "
            "and size of input(Velocitys) is %d.",
            param_dims.size(), velocity_dims.size()));

    if (ctx->GetInputsVarType("Grads")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      for (auto idx = 0; idx < param_dims.size(); idx++) {
        PADDLE_ENFORCE_EQ(
            param_dims[idx], grad_dims[idx],
            platform::errors::InvalidArgument(
                "Param and Grad input of MTMomentumOp should have the same "
                "dimension. But received Param's dim [%s] and Grad's dim [%s].",
                param_dims[idx], grad_dims[idx]));
        PADDLE_ENFORCE_EQ(
            param_dims[idx], velocity_dims[idx],
            platform::errors::InvalidArgument(
                "Param and Velocity of MTMomentumOp should have the same "
                "dimension. But received Param's dim [%s] and Velocity [%s].",
                param_dims[idx], velocity_dims[idx]));
      }
    }

    ctx->SetOutputsDim("ParamOuts", param_dims);
    ctx->SetOutputsDim("VelocityOuts", param_dims);
    if (ctx->HasOutput("MasterParamOuts")) {
      ctx->SetOutputsDim("MasterParamOuts", param_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Params");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    mt_momentum, ops::MTMomentumOp, ops::MTMomentumOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    mt_momentum,
    ops::MTMomentumOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MTMomentumOpKernel<paddle::platform::CPUDeviceContext, double>);
