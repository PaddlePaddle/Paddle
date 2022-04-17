/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/optimizers/lars_momentum_op.h"

namespace paddle {
namespace operators {

class LarsMomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("Param"), "Input", "Param", "LarsMomentum");
    OP_INOUT_CHECK(ctx->HasInputs("Grad"), "Input", "Grad", "LarsMomentum");
    OP_INOUT_CHECK(ctx->HasInputs("Velocity"), "Input", "Velocity",
                   "LarsMomentum");
    OP_INOUT_CHECK(ctx->HasInputs("LearningRate"), "Input", "LearningRate",
                   "LarsMomentum");
    OP_INOUT_CHECK(ctx->HasOutputs("ParamOut"), "Output", "ParamOut",
                   "LarsMomentum");
    OP_INOUT_CHECK(ctx->HasOutputs("VelocityOut"), "Output", "VelocityOut",
                   "LarsMomentum");
    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("Param").front(),
        framework::proto::VarType::LOD_TENSOR,
        platform::errors::InvalidArgument(
            "The input var's type should be LoDTensor, but the received is %s",
            ctx->GetInputsVarType("Param").front()));

    auto lr_dims = ctx->GetInputsDim("LearningRate");
    auto grad_dim = ctx->GetInputsDim("Grad");
    auto param_dim = ctx->GetInputsDim("Param");
    auto velocity_dim = ctx->GetInputsDim("Velocity");
    auto lars_weight_decays =
        ctx->Attrs().Get<std::vector<float>>("lars_weight_decay");
    auto multi_precision = ctx->Attrs().Get<bool>("multi_precision");

    PADDLE_ENFORCE_EQ(
        param_dim.size(), grad_dim.size(),
        platform::errors::InvalidArgument(
            "Input(Param) and Input(Grad) of LarsMomentumOp should have "
            "same quantity. But number of Param is [%d] and Grad is [%d].",
            param_dim.size(), grad_dim.size()));
    PADDLE_ENFORCE_EQ(
        param_dim.size(), velocity_dim.size(),
        platform::errors::InvalidArgument(
            "Input(Param) and Input(Velocity) of LarsMomentumOp should "
            "have same quantity. But number of Param is [%d] and Velocity "
            "is [%d].",
            param_dim.size(), velocity_dim.size()));
    PADDLE_ENFORCE_EQ(
        lars_weight_decays.size(), grad_dim.size(),
        platform::errors::InvalidArgument(
            "Attr(Lars_weight_decay) and "
            "Input(Grad) of LarsMomentumOp should have same quantity. "
            "But number of Lars_weight_decay is [%d] and Grad is [%d].",
            lars_weight_decays.size(), grad_dim.size()));

    if (multi_precision) {
      OP_INOUT_CHECK(ctx->HasInputs("MasterParam"), "Input", "MasterParam",
                     "LarsMomentumMultiPrecision");
      OP_INOUT_CHECK(ctx->HasOutputs("MasterParamOut"), "Output",
                     "MasterParamOut", "LarsMomentumMultiPrecision");
    }
    for (size_t i = 0; i < lr_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(phi::product(lr_dims[i]), 1,
                        platform::errors::InvalidArgument(
                            "Learning_rate should be a scalar. But Received "
                            "LearningRate's dim [%s]",
                            phi::product(lr_dims[i])));
    }

    for (size_t i = 0; i < param_dim.size(); ++i) {
      PADDLE_ENFORCE_EQ(ctx->GetInputsVarType("Grad")[i],
                        framework::proto::VarType::LOD_TENSOR,
                        platform::errors::InvalidArgument(
                            "The Var(%s)'s type should be LoDTensor, "
                            "but the received is %s",
                            ctx->Inputs("Grad")[i].front(),
                            ctx->GetInputsVarType("Grad")[i]));
      PADDLE_ENFORCE_EQ(
          param_dim[i], grad_dim[i],
          platform::errors::InvalidArgument(
              "Input(Param) and Input(Grad) input of LarsMomentumOp shall "
              "have same dimension. But Param`s dim is [%s] and Grad's dim "
              "is [%s].",
              param_dim[i], grad_dim[i]));
      PADDLE_ENFORCE_EQ(
          param_dim[i], velocity_dim[i],
          platform::errors::InvalidArgument(
              "Input(Param) and Input(Velocity) of LarsMomentumOp shall have "
              "same dimension. But Param dim [%s] differs with Velocity dim "
              "[%s].",
              param_dim[i], velocity_dim[i]));
    }
    ctx->SetOutputsDim("ParamOut", param_dim);
    ctx->SetOutputsDim("VelocityOut", param_dim);
    if (ctx->HasOutputs("MasterParamOut")) {
      ctx->SetOutputsDim("MasterParamOut", param_dim);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class LarsMomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(LoDTensor, default LoDTensor<float>) "
             "Input parameter that has to be updated")
        .AsDuplicable();
    AddInput("Grad",
             "(LoDTensor, default LoDTensor<float>) "
             "Input gradient of the parameter")
        .AsDuplicable();
    AddInput("Velocity",
             "(LoDTensor, default LoDTensor<float>) "
             "Input velocity (corresponding to the parameter) "
             "that has to be updated")
        .AsDuplicable();
    AddInput("LearningRate",
             "(LoDTensor, default LoDTensor<float>) "
             "Input learning rate")
        .AsDuplicable();
    AddInput("MasterParam", "FP32 master weight for AMP.")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("ParamOut",
              "(LoDTensor) This output is updated parameter. "
              "It shared memory with Input(Param).")
        .AsDuplicable();
    AddOutput("VelocityOut",
              "(LoDTensor) This output is updated velocity. "
              "It shared memory with Input(Velocity).")
        .AsDuplicable();
    AddOutput("MasterParamOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParam).")
        .AsDuplicable()
        .AsDispensable();
    AddAttr<float>("mu", "(float) Momentum coefficient");
    AddAttr<float>("lars_coeff", "(float, default 0.001) LARS coefficient.")
        .SetDefault(0.001);
    AddAttr<std::vector<float>>(
        "lars_weight_decay",
        "(std::vector<float>, default 0.0005) LARS weight decay params")
        .SetDefault({0.0005});
    AddAttr<float>("epsilon",
                   "(float, default 0.0) epsilon to avoid Division by Zero.")
        .SetDefault(0.0);
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
Lars Momentum Optimizer.

This optimizer use LARS (https://arxiv.org/abs/1708.03888) to optimize each
weight using a local learning rate:

$$
local\_lr = \eta  *
    \frac{\left \| param \right \|}{\left \| grad \right \| + \beta *\left \| param \right \|} \\
velocity = mu * velocity +
    local\_lr * (grad + \beta * param) \\
param = param - velocity. \\
$$

Note that we use lars_weight_decay here to decay weights, you may need not to
use L2 regularizers in case of using LARS.

)DOC");
  }
};

class LarsMomentumOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {}
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    lars_momentum, ops::LarsMomentumOp, ops::LarsMomentumOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::LarsMomentumOpVarTypeInference);
REGISTER_OP_CPU_KERNEL(lars_momentum, ops::LarsMomentumOpKernel<float>,
                       ops::LarsMomentumOpKernel<double>);
