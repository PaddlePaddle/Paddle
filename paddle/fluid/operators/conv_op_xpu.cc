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
#include "paddle/fluid/operators/conv_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#ifdef PADDLE_WITH_XPU
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class GemmConvXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
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
    const int batch_size = static_cast<int>(input->dims()[0]);
    const int img_c = static_cast<int>(input->dims()[1]);
    const int img_h = static_cast<int>(input->dims()[2]);
    const int img_w = static_cast<int>(input->dims()[3]);
    const int f = static_cast<int>(filter.dims()[0]);
    const int win_h = static_cast<int>(filter.dims()[2]);
    const int win_w = static_cast<int>(filter.dims()[3]);
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> k_size;
    k_size.push_back(win_h);
    k_size.push_back(win_w);
    int r = xpu::conv2d<float, float, float, int16_t>(
        dev_ctx.x_context(), input->data<float>(), filter.data<float>(),
        output->data<float>(), batch_size, img_c, img_h, img_w, f, k_size,
        strides, paddings, dilations, groups, nullptr, nullptr, nullptr, true);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU conv kernel return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};
template <typename DeviceContext, typename T>
class GemmConvGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
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
    const int batch_size = static_cast<int>(input->dims()[0]);
    const int img_c = static_cast<int>(input->dims()[1]);
    const int img_h = static_cast<int>(input->dims()[2]);
    const int img_w = static_cast<int>(input->dims()[3]);
    const int f = static_cast<int>(filter.dims()[0]);
    const int win_h = static_cast<int>(filter.dims()[2]);
    const int win_w = static_cast<int>(filter.dims()[3]);
    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
    }
    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
    }
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> k_size;
    k_size.push_back(win_h);
    k_size.push_back(win_w);
    int r = xpu::conv2d_grad<float, float, float, int16_t>(
        dev_ctx.x_context(), input->data<T>(), filter.data<T>(),
        output_grad->data<T>(), input_grad ? input_grad->data<T>() : nullptr,
        filter_grad ? filter_grad->data<T>() : nullptr, batch_size, img_c,
        img_h, img_w, f, k_size, strides, paddings, dilations, groups, nullptr,
        nullptr, nullptr, nullptr, nullptr, true);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU conv kernel return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    depthwise_conv2d,
    ops::GemmConvXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    conv2d, ops::GemmConvXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    conv2d_grad,
    ops::GemmConvGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    depthwise_conv2d_grad,
    ops::GemmConvGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
