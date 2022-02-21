/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/phi/api/all.h"

#ifdef PADDLE_WITH_XPU
namespace paddle {
namespace operators {

template <typename T>
class FusedConvBiasActXPUKernel : public framework::OpKernel<T> {
 public:
  using XPUT = typename XPUTypeTrait<T>::Type;

  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* bias = context.Input<Tensor>("Bias");
    // The filter will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());
    int groups = context.Attr<int>("groups");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    std::string act_type = context.Attr<std::string>("activation");

    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    auto xpu_act_type = act_type == "relu" ? xpu::Activation_t::RELU
                                           : xpu::Activation_t::LINEAR;

    phi::DDim in_data_dims =
        phi::slice_ddim(input->dims(), 2, input->dims().size());
    framework::DDim filter_data_dims =
        phi::slice_ddim(filter.dims(), 2, filter.dims().size());
    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    const int batch_size = static_cast<int>(input->dims()[0]);
    const int img_c = static_cast<int>(input->dims()[1]);
    const int img_h = static_cast<int>(input->dims()[2]);
    const int img_w = static_cast<int>(input->dims()[3]);
    const int f = static_cast<int>(filter.dims()[0]);

    auto input_data = reinterpret_cast<const XPUT*>(input->data<T>());
    auto filter_data = reinterpret_cast<const XPUT*>(filter.data<T>());
    auto bias_data = bias->data<float>();
    auto output_data = reinterpret_cast<XPUT*>(output->data<T>());
    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    int r = xpu::conv2d_fusion<XPUT, XPUT, XPUT, int16_t>(
        dev_ctx.x_context(), input_data, filter_data, output_data, batch_size,
        img_c, img_h, img_w, f, ksize, strides, paddings, dilations, groups,
        nullptr, nullptr, nullptr, true, bias_data, nullptr, xpu_act_type,
        nullptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_fusion");
  }
};

template <typename T>
class FusedConvBiasActGradXPUKernel : public framework::OpKernel<T> {
 public:
  using XPUT = typename XPUTypeTrait<T>::Type;

  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output = context.Input<Tensor>("Output");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));
    // The filter and filter_grad will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    if (!input_grad && !filter_grad) return;
    int groups = context.Attr<int>("groups");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    std::string act_type = context.Attr<std::string>("activation");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    framework::DDim in_data_dims =
        phi::slice_ddim(input->dims(), 2, input->dims().size());
    framework::DDim filter_data_dims =
        phi::slice_ddim(filter.dims(), 2, filter.dims().size());
    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int r = XPU_SUCCESS;

    // relu_grad
    XPUT* output_grad_or_relu_grad_data = nullptr;
    if (act_type == "relu") {
      output_grad_or_relu_grad_data =
          RAII_GUARD.alloc_l3_or_gm<XPUT>(output_grad->numel());

      auto output_data = reinterpret_cast<const XPUT*>(output->data<T>());
      auto output_grad_data =
          reinterpret_cast<const XPUT*>(output_grad->data<T>());
      r = xpu::relu_grad(dev_ctx.x_context(), output_data, output_data,
                         output_grad_data, output_grad_or_relu_grad_data,
                         output_grad->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu_grad");
    } else {
      output_grad_or_relu_grad_data =
          reinterpret_cast<XPUT*>(const_cast<T*>(output_grad->data<T>()));
    }

    const int batch_size = static_cast<int>(input->dims()[0]);
    const int img_c = static_cast<int>(input->dims()[1]);
    const int img_h = static_cast<int>(input->dims()[2]);
    const int img_w = static_cast<int>(input->dims()[3]);
    const int f = static_cast<int>(filter.dims()[0]);
    XPUT* input_grad_data = nullptr;
    XPUT* filter_grad_data = nullptr;
    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      input_grad_data = reinterpret_cast<XPUT*>(input_grad->data<T>());
    }
    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
      filter_grad_data = reinterpret_cast<XPUT*>(filter_grad->data<T>());
    }

    auto input_data = reinterpret_cast<const XPUT*>(input->data<T>());
    auto filter_data = reinterpret_cast<const XPUT*>(filter.data<T>());
    r = xpu::conv2d_grad<XPUT, XPUT, XPUT, int16_t>(
        dev_ctx.x_context(), input_data, filter_data,
        output_grad_or_relu_grad_data, input_grad_data, filter_grad_data,
        batch_size, img_c, img_h, img_w, f, ksize, strides, paddings, dilations,
        groups, nullptr, nullptr, nullptr, nullptr, nullptr, true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_grad");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(
    fused_conv2d_bias_act,
    ops::FusedConvBiasActXPUKernel<paddle::platform::float16>,
    ops::FusedConvBiasActXPUKernel<float>);
REGISTER_OP_XPU_KERNEL(
    fused_conv2d_bias_act_grad,
    ops::FusedConvBiasActGradXPUKernel<paddle::platform::float16>,
    ops::FusedConvBiasActGradXPUKernel<float>);

#endif
