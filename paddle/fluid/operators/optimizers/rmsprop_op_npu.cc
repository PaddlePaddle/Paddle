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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class RMSPROPNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *grad_var = ctx.InputVar("Grad");
    auto *param_out = ctx.Output<phi::DenseTensor>("ParamOut");
    auto *moment_out = ctx.Output<phi::DenseTensor>("MomentOut");
    auto *mean_square_out = ctx.Output<phi::DenseTensor>("MeanSquareOut");

    param_out->mutable_data<T>(ctx.GetPlace());
    moment_out->mutable_data<T>(ctx.GetPlace());
    mean_square_out->mutable_data<T>(ctx.GetPlace());

    auto epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    auto rho = static_cast<T>(ctx.Attr<float>("decay"));
    auto momentum = static_cast<T>(ctx.Attr<float>("momentum"));
    auto *p_tensor = ctx.Input<phi::DenseTensor>("Param");
    auto *ms_tensor = ctx.Input<phi::DenseTensor>("MeanSquare");
    auto *lr_tensor = ctx.Input<phi::DenseTensor>("LearningRate");
    auto *mom_tensor = ctx.Input<phi::DenseTensor>("Moment");
    bool centered = ctx.Attr<bool>("centered");

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    if (grad_var->IsType<phi::DenseTensor>()) {
      auto *grad_tensor = ctx.Input<phi::DenseTensor>("Grad");
      if (centered) {
        framework::NPUAttributeMap attr_input = {{"use_locking", false}};
        const phi::DenseTensor *rho_tensor = nullptr;
        const phi::DenseTensor *momentum_tensor = nullptr;
        const phi::DenseTensor *epsilon_tensor = nullptr;
        phi::DenseTensor rho_tmp(experimental::DataType::FLOAT32);
        rho_tmp.mutable_data<T>({1}, ctx.GetPlace());
        FillNpuTensorWithConstant<T>(&rho_tmp, rho);
        rho_tensor = &rho_tmp;
        phi::DenseTensor momentum_tmp(experimental::DataType::FLOAT32);
        momentum_tmp.mutable_data<T>({1}, ctx.GetPlace());
        FillNpuTensorWithConstant<T>(&momentum_tmp, momentum);
        momentum_tensor = &momentum_tmp;
        phi::DenseTensor epsilon_tmp(experimental::DataType::FLOAT32);
        epsilon_tmp.mutable_data<T>({1}, ctx.GetPlace());
        FillNpuTensorWithConstant<T>(&epsilon_tmp, epsilon);
        epsilon_tensor = &epsilon_tmp;
        auto *mg_tensor = ctx.Input<phi::DenseTensor>("MeanGrad");
        auto *mean_grad_out = ctx.Output<phi::DenseTensor>("MeanGradOut");
        mean_grad_out->mutable_data<T>(ctx.GetPlace());
        const auto &runner_applycenterrmsprop = NpuOpRunner(
            std::string("ApplyCenteredRMSPropD"),
            {*p_tensor,
             *mg_tensor,
             *ms_tensor,
             *mom_tensor,
             *lr_tensor,
             *rho_tensor,
             *momentum_tensor,
             *epsilon_tensor,
             *grad_tensor},
            {*param_out, *mean_grad_out, *mean_square_out, *moment_out},
            {attr_input});
        runner_applycenterrmsprop.Run(stream);
      } else {
        framework::NPUAttributeMap attr_input = {
            {"rho", rho}, {"momentum", momentum}, {"epsilon", epsilon}};
        const auto &runner_applyrmsprop = NpuOpRunner(
            std::string("ApplyRMSPropD"),
            {*p_tensor, *ms_tensor, *mom_tensor, *lr_tensor, *grad_tensor},
            {*param_out, *mean_square_out, *moment_out},
            {attr_input});
        runner_applyrmsprop.Run(stream);
      }
    } else {
      PADDLE_ENFORCE_EQ(false,
                        true,
                        platform::errors::PermissionDenied(
                            "Unsupported Variable Type of Grad "
                            "in RmspropOp. Excepted LodTensor, "
                            "But received [%s]",
                            paddle::framework::ToTypeName(grad_var->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    rmsprop, ops::RMSPROPNPUKernel<paddle::platform::NPUDeviceContext, float>)
