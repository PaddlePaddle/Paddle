/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cstring>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class UpdateLossScalingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = framework::proto::VarType::FP32;
    if (ctx.MultiInputVar("X").size() >= 1) {
      dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    }

    return framework::OpKernelType(dtype, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
#ifndef PADDLE_WITH_XPU
    if (var_name == "FoundInfinite" || var_name == "StopUpdate") {
      return expected_kernel_type;
    }
#endif
    return framework::OperatorWithKernel::GetKernelTypeForVar(
        var_name, tensor, expected_kernel_type);
  }
};

class UpdateLossScalingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensors) The input tensors of update_loss_scaling operator.")
        .AsDuplicable();
    AddInput("FoundInfinite",
             "(Tensor) 1-dim tensor, contains a bool scalar, which indicates "
             "whether there is any infinite gradient.");
    AddInput("PrevLossScaling",
             "(Tensor) 1-dim tensor, previous loss scaling.");
    AddInput("InGoodSteps",
             "(Tensor) 1-dim tensor, accumulates good steps in which all "
             "gradients are finite.");
    AddInput("InBadSteps",
             "(Tensor) 1-dim tensor, accumulates bad steps in which some "
             "gradients are infinite.");
    AddOutput("Out",
              "(Tensors) The output tensor of update_loss_scaling operator.")
        .AsDuplicable();
    AddOutput("LossScaling", "(Tensor) 1-dim tensor, updated loss scaling.");
    AddOutput("OutGoodSteps", "(Tensor) 1-dim tensor, pdated good steps.");
    AddOutput("OutBadSteps", "(Tensor) 1-dim tensor, updated bad steps.");
    AddInput("StopUpdate",
             "(Tensor) 1-dim tensor. Stop updating loss scaling, and just "
             "zero inputs. It has higher priority than Attr(stop_update).")
        .AsDispensable();
    AddAttr<int>("incr_every_n_steps",
                 "A value represents increasing loss scaling every n "
                 "consecutive steps with finite gradients.");
    AddAttr<int>("decr_every_n_nan_or_inf",
                 "A value represents decreasing loss scaling every n "
                 "accumulated steps with nan or inf gradients.");
    AddAttr<float>("incr_ratio",
                   "The multiplier to use when increasing the loss scaling.")
        .AddCustomChecker([](float incr_ratio) {
          PADDLE_ENFORCE_EQ(incr_ratio > 1.0f,
                            true,
                            platform::errors::InvalidArgument(
                                "'incr_ratio' should be greater than 1, but "
                                "the received is %f",
                                incr_ratio));
        });
    AddAttr<float>(
        "decr_ratio",
        "The less-than-one-multiplier to use when decreasing loss scaling.")
        .AddCustomChecker([](float decr_ratio) {
          PADDLE_ENFORCE_EQ(decr_ratio > 0.0f && decr_ratio < 1.0f,
                            true,
                            platform::errors::InvalidArgument(
                                "'decr_ratio' should be between 0 and 1, but "
                                "the received is %f",
                                decr_ratio));
        });
    AddAttr<bool>("stop_update",
                  "Stop updating loss scaling, and just zero inputs.")
        .SetDefault(false);
    AddComment(R"DOC(
Update loss scaling according to overall gradients. If all gradients is
finite after incr_every_n_steps, loss scaling will increase by incr_ratio.
Otherwise, loss scaling will decrease by decr_ratio after
decr_every_n_nan_or_inf steps and each step some gradients are infinite.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = phi::CPUContext;

DECLARE_INFER_SHAPE_FUNCTOR(update_loss_scaling,
                            UpdateLossScalingInferShapeFunctor,
                            PD_INFER_META(phi::UpdateLossScalingInferMeta));
REGISTER_OPERATOR(
    update_loss_scaling,
    ops::UpdateLossScalingOp,
    ops::UpdateLossScalingOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    UpdateLossScalingInferShapeFunctor);
