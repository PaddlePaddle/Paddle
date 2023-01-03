/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/instance_norm_op.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/ternary.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

framework::OpKernelType InstanceNormOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  // By default, the type of the scale, bias, mean,
  // and var tensors should both be float. (For float or float16 input tensor)
  // or double (For double input tensor).
  auto in_param_type = framework::proto::VarType::FP32;
  if (input_data_type == framework::proto::VarType::FP64) {
    in_param_type = framework::proto::VarType::FP64;
  }
  if (ctx.HasInput("Scale")) {
    PADDLE_ENFORCE_EQ(in_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Scale")->dtype()),
                      platform::errors::InvalidArgument(
                          "Scale input should be of float type"));
  }
  if (ctx.HasInput("Bias")) {
    PADDLE_ENFORCE_EQ(in_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Bias")->dtype()),
                      platform::errors::InvalidArgument(
                          "Bias input should be of float type"));
  }

  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

void InstanceNormOpMaker::Make() {
  AddAttr<float>("epsilon", "")
      .SetDefault(1e-5)
      .AddCustomChecker([](const float &epsilon) {
        PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f,
                          true,
                          platform::errors::InvalidArgument(
                              "'epsilon' should be between 0.0 and 0.001."));
      });
  AddInput("X", "The input tensor");
  AddInput("Scale",
           "Scale is a 1-dimensional tensor of size C "
           "that is applied to the output")
      .AsDispensable();
  AddInput("Bias",
           "Bias is a 1-dimensional tensor of size C "
           "that is applied to the output")
      .AsDispensable();
  AddOutput("Y", "result after normalization");
  AddOutput("SavedMean",
            "Mean of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate()
      .AsExtra();
  AddOutput("SavedVariance",
            "Variance of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate()
      .AsExtra();
  AddComment(R"DOC(
Instance Normalization.

Instance Norm has been implemented as disscussed in the paper:
https://arxiv.org/pdf/1607.08022.pdf
Can be used as a normalizer function for conv2d and fully_connected operations.
The required data format for this layer is as following:
NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
}

framework::OpKernelType InstanceNormGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar(framework::GradVarName("Y"));
  if (var == nullptr) {
    PADDLE_THROW(
        platform::errors::NotFound("cannot find gradient variable of Y"));
  }
  const phi::DenseTensor *t = nullptr;
  if (var->IsType<phi::DenseTensor>()) {
    t = &var->Get<phi::DenseTensor>();
  } else if (var->IsType<phi::DenseTensor>()) {
    t = &var->Get<phi::DenseTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("gradient variable of Y is empty"));
  }
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
}

framework::OpKernelType InstanceNormDoubleGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar("DY");
  if (var == nullptr) {
    PADDLE_THROW(
        platform::errors::NotFound("cannot find gradient variable of Y"));
  }
  const phi::DenseTensor *t = nullptr;
  if (var->IsType<phi::DenseTensor>()) {
    t = &var->Get<phi::DenseTensor>();
  } else if (var->IsType<phi::DenseTensor>()) {
    t = &var->Get<phi::DenseTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("gradient variable of Y is empty"));
  }
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
}

DECLARE_INPLACE_OP_INFERER(InstanceNormDoubleGradOpInplaceInferer,
                           {"DY", "DDY"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(instance_norm,
                            InstanceNormInferShapeFunctor,
                            PD_INFER_META(phi::InstanceNormInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(instance_norm_grad,
                            InstanceNormGradInferShapeFunctor,
                            PD_INFER_META(phi::InstanceNormGradInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(
    instance_norm_grad_grad,
    InstanceNormDoubleGradInferShapeFunctor,
    PD_INFER_META(phi::InstanceNormDoubleGradInferMeta));
REGISTER_OPERATOR(instance_norm,
                  ops::InstanceNormOp,
                  ops::InstanceNormOpMaker,
                  ops::InstanceNormOpInferVarType,
                  ops::InstanceNormGradMaker<paddle::framework::OpDesc>,
                  ops::InstanceNormGradMaker<paddle::imperative::OpBase>,
                  InstanceNormInferShapeFunctor);
REGISTER_OPERATOR(instance_norm_grad,
                  ops::InstanceNormGradOp,
                  ops::InstanceNormDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::InstanceNormDoubleGradMaker<paddle::imperative::OpBase>,
                  InstanceNormGradInferShapeFunctor);
REGISTER_OPERATOR(instance_norm_grad_grad,
                  ops::InstanceNormDoubleGradOp,
                  ops::InstanceNormDoubleGradOpInplaceInferer,
                  InstanceNormDoubleGradInferShapeFunctor);

REGISTER_OP_VERSION(instance_norm)
    .AddCheckpoint(
        R"ROC(
      Change dispensable of attribute from False to True in instance_norm.
    )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .ModifyAttr(
                "Bias",
                "The arg 'dispensable' of Input 'Bias' is changed: from "
                "'False' to 'True'.",
                true)
            .ModifyAttr(
                "Scale",
                "The arg 'dispensable' of Input 'Scale' is changed: from "
                "'False' to 'True'.",
                true));
