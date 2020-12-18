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

#include "paddle/fluid/operators/optimizers/adam_op.h"
#include <gflags/gflags.h>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#ifdef PADDLE_WITH_XPU
template <typename DeviceContext, typename T>
class AdamOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong type，Expected Var(%s)'s "
                          "type is LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));
    using paddle::framework::LoDTensor;

    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));

    auto& param = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Param"), "Input",
                                  "Param", "Adam");
    // auto& grad = Ref(ctx.Input<LoDTensor>("Grad"), "Must set Grad");
    auto* grad_var = ctx.InputVar("Grad");
    auto& mom1 = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Moment1"), "Input",
                                 "Moment1", "Adam");
    auto& mom2 = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Moment2"), "Input",
                                 "Moment2", "Adam");
    auto& lr = GET_DATA_SAFELY(ctx.Input<LoDTensor>("LearningRate"), "Input",
                               "LearningRate", "Adam");
    auto& beta1_pow = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Beta1Pow"), "Input",
                                      "Beta1Pow", "Adam");
    auto& beta2_pow = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Beta2Pow"), "Input",
                                      "Beta2Pow", "Adam");

    auto& param_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("ParamOut"),
                                      "Output", "ParamOut", "Adam");
    auto& mom1_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Moment1Out"),
                                     "Output", "Moment1Out", "Adam");
    auto& mom2_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Moment2Out"),
                                     "Output", "Moment2Out", "Adam");

    auto* beta1_pow_out = ctx.Output<LoDTensor>("Beta1PowOut");
    auto* beta2_pow_out = ctx.Output<LoDTensor>("Beta2PowOut");
    PADDLE_ENFORCE_EQ(beta1_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong size, Expected beta1 pow "
                          "output size is 1, but received "
                          "value is:%d.",
                          beta1_pow_out->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong size, Expected beta2 pow "
                          "output size is 1, but received "
                          "value is:%d.",
                          beta2_pow_out->numel()));
                          
    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    if (ctx.HasInput("Beta1Tensor")) {
      auto* beta1_tensor = ctx.Input<framework::Tensor>("Beta1Tensor");
      beta1 = static_cast<T>(GetAttrFromTensor(beta1_tensor));
    }
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    if (ctx.HasInput("Beta2Tensor")) {
      auto* beta2_tensor = ctx.Input<framework::Tensor>("Beta2Tensor");
      beta2 = static_cast<T>(GetAttrFromTensor(beta2_tensor));
    }
    if (grad_var->IsType<framework::LoDTensor>()) {
      auto& grad = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Grad"), "Input",
                                   "Grad", "Adam");
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      const T* beta1_pow_ptr = beta1_pow.template data<T>();
      const T* beta2_pow_ptr = beta2_pow.template data<T>();
      Tensor xpu_beta1_pow;
      Tensor xpu_beta2_pow;
      if (beta1_pow.place() == platform::CPUPlace() &&
          beta2_pow.place() == platform::CPUPlace()) {
        TensorCopy(beta1_pow, ctx.GetPlace(), dev_ctx, &xpu_beta1_pow);
        TensorCopy(beta2_pow, ctx.GetPlace(), dev_ctx, &xpu_beta2_pow);
        dev_ctx.Wait();
        beta1_pow_ptr = xpu_beta1_pow.template data<T>();
        beta2_pow_ptr = xpu_beta2_pow.template data<T>();
      }
      int r = xpu::adam(
          dev_ctx.x_context(), grad.template data<T>(), mom1.template data<T>(),
          mom2.template data<T>(), param.template data<T>(), beta1_pow_ptr,
          beta2_pow_ptr, beta1, beta2, epsilon, lr.template data<T>(),
          mom1_out.template mutable_data<T>(ctx.GetPlace()),
          mom2_out.template mutable_data<T>(ctx.GetPlace()),
          param_out.template mutable_data<T>(ctx.GetPlace()), param.numel());

      //update in cpu and then copy to xpu
      if (beta1_pow.place() == platform::CPUPlace() &&
          beta2_pow.place() == platform::CPUPlace()) {
        const T* beta1_pow_p = beta1_pow.template data<T>();
        beta1_pow_out->mutable_data<T>(platform::CPUPlace())[0] =
            beta1 * beta1_pow_p[0];
        const T* beta2_pow_p = beta2_pow.template data<T>();
        beta2_pow_out->mutable_data<T>(platform::CPUPlace())[0] =
            beta2 * beta2_pow_p[0];
      } else {
        T cpu_beta1_pow_out_data;
        T cpu_beta2_pow_out_data;
        xpu_memcpy(&cpu_beta1_pow_out_data, beta1_pow_ptr, sizeof(T),
                   XPU_DEVICE_TO_HOST);
        cpu_beta1_pow_out_data = cpu_beta1_pow_out_data * beta1;
        xpu_memcpy(&cpu_beta2_pow_out_data, beta2_pow_ptr, sizeof(T),
                   XPU_DEVICE_TO_HOST);
        cpu_beta2_pow_out_data = cpu_beta2_pow_out_data * beta2;

        T* beta1_pow_out_p = beta1_pow_out->mutable_data<T>(ctx.GetPlace());
        T* beta2_pow_out_p = beta2_pow_out->mutable_data<T>(ctx.GetPlace());
        xpu_memcpy(beta1_pow_out_p, &cpu_beta1_pow_out_data, sizeof(T),
                   XPU_HOST_TO_DEVICE);
        xpu_memcpy(beta2_pow_out_p, &cpu_beta2_pow_out_data, sizeof(T),
                   XPU_HOST_TO_DEVICE);
      }

      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::External(
                            "XPU API return wrong value[%d], please check "
                            "where Baidu Kunlun Card is properly installed.",
                            r));
    } else {
      PADDLE_ENFORCE_EQ(1, 2, platform::errors::InvalidArgument(
                                  "Variable type not supported by adam_op"));
    }
  }
};
#endif

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
#ifdef PADDLE_WITH_XPU
REGISTER_OP_XPU_KERNEL(
    adam, ops::AdamOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
