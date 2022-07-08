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

#ifdef PADDLE_WITH_XPU

#include <gflags/gflags.h>

#include <iostream>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

static inline float GetAttrFromTensor(const framework::Tensor* tensor) {
  const float* tensor_data = tensor->data<float>();
  framework::Tensor cpu_tensor;
  if (platform::is_gpu_place(tensor->place()) ||
      platform::is_xpu_place(tensor->place())) {
    paddle::framework::TensorCopySync(
        *tensor, platform::CPUPlace(), &cpu_tensor);
    tensor_data = cpu_tensor.data<float>();
  }
  return tensor_data[0];
}

using framework::OpKernelType;
using framework::Tensor;

template <typename DeviceContext, typename T>
class RmspropOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using paddle::framework::LoDTensor;

    // check Param & Grad tensor type
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<LoDTensor>(),
                      true,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong type，Expected Var(%s)'s "
                          "type is LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));

    const auto* grad_var = ctx.InputVar("Grad");
    PADDLE_ENFORCE_EQ(grad_var->IsType<LoDTensor>(),
                      true,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong type，Expected Var(%s)'s "
                          "type is LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Grad").front(),
                          framework::ToTypeName(grad_var->Type())));

    // inputs
    auto& param = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>("Param"), "Input", "Param", "Rmsprop");
    auto& meanSquare = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>("MeanSquare"), "Input", "MeanSquare", "Rmsprop");
    auto& grad = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>("Grad"), "Input", "Grad", "Rmsprop");
    auto& mom = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>("Moment"), "Input", "Moment", "Rmsprop");

    auto* learning_rate = ctx.Input<Tensor>("LearningRate");
    PADDLE_ENFORCE_EQ(learning_rate->dims().size(),
                      1,
                      platform::errors::InvalidArgument(
                          "learining rate should have dimension = 1."
                          " But received learning rate dim [%s] ",
                          learning_rate->dims().size()));
    T lr = static_cast<T>(GetAttrFromTensor(learning_rate));

    // constants
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    T decay = static_cast<T>(ctx.Attr<float>("decay"));
    T momentum = static_cast<T>(ctx.Attr<float>("momentum"));

    bool centered = ctx.Attr<bool>("centered");
    if (centered) {
      VLOG(0) << "'centered' is not supported in RMSProp XPU version. use "
                 "XPU_BLACK_LIST to disable this op.";
      // TODO(houj04): when XDNN api supports 'center', add input of
      // mean_grad_input and output of mean_grad_output. auto *mean_grad_input =
      // ctx.Input<Tensor>("MeanGrad"); auto *mean_grad_output =
      // ctx.Output<Tensor>("MeanGradOut");
    }

    // outputs
    auto& param_out = GET_DATA_SAFELY(
        ctx.Output<LoDTensor>("ParamOut"), "Output", "ParamOut", "Rmsprop");
    auto& mom_out = GET_DATA_SAFELY(
        ctx.Output<LoDTensor>("MomentOut"), "Output", "MomentOut", "Rmsprop");
    auto& mom_sqrt_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("MeanSquareOut"),
                                         "Output",
                                         "MeanSquareOut",
                                         "Rmsprop");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    // int rmsprop(Context* ctx, const T* g, const T* p, const float* ms, const
    // float* mom, T* p_out, float* ms_out, float* mom_out, float epsilon, float
    // rho, float momentum, float lr, int n);
    int r = xpu::rmsprop(dev_ctx.x_context(),
                         grad.template data<T>(),
                         param.template data<T>(),
                         meanSquare.template data<T>(),
                         mom.template data<T>(),
                         param_out.template mutable_data<T>(ctx.GetPlace()),
                         mom_sqrt_out.template mutable_data<T>(ctx.GetPlace()),
                         mom_out.template mutable_data<T>(ctx.GetPlace()),
                         epsilon,
                         decay,
                         momentum,
                         lr,
                         param.numel());

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "rmsprop");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    rmsprop,
    ops::RmspropOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
