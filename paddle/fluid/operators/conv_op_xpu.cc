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
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#ifdef PADDLE_WITH_XPU
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class GemmConvXPUKernel : public framework::OpKernel<T> {
  using XPUT = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *input = context.Input<Tensor>("Input");
    // The filter will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor *output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());
    int groups = context.Attr<int>("groups");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    const std::string data_format = context.Attr<std::string>("data_format");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    PADDLE_ENFORCE_EQ(
        data_format == "NDHWC",
        false,
        platform::errors::InvalidArgument(
            ("XPU does not support data_format is NDHWC in conv op.")));

    framework::DDim in_data_dims =
        phi::slice_ddim(input->dims(), 2, input->dims().size());
    framework::DDim filter_data_dims =
        phi::slice_ddim(filter.dims(), 2, filter.dims().size());
    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

    int batch_size = static_cast<int>(input->dims()[0]);
    int img_c = static_cast<int>(input->dims()[1]);
    int img_h = static_cast<int>(input->dims()[2]);
    int img_w = static_cast<int>(input->dims()[3]);
    int f = static_cast<int>(filter.dims()[0]);
    bool is_nchw = true;
    if (data_format == "NHWC") {
      img_c = static_cast<int>(input->dims()[3]);
      img_h = static_cast<int>(input->dims()[1]);
      img_w = static_cast<int>(input->dims()[2]);
      is_nchw = false;
    }

    const XPUT *input_data = reinterpret_cast<const XPUT *>(input->data<T>());
    const XPUT *filter_data = reinterpret_cast<const XPUT *>(filter.data<T>());
    XPUT *output_data = reinterpret_cast<XPUT *>(output->data<T>());

    auto &dev_ctx = context.template device_context<DeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

    XPUT *filter_data_tmp;
    const XPUT *filter_data_ptr = filter_data;
    if (data_format == "NHWC") {
      filter_data_tmp = RAII_GUARD.alloc<XPUT>(filter.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(filter_data_tmp);
      std::vector<int> filter_shape = phi::vectorize<int>(filter.dims());
      int r = xpu::transpose<XPUT>(dev_ctx.x_context(),
                                   filter_data,
                                   filter_data_tmp,
                                   filter_shape,
                                   {0, 2, 3, 1});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
      filter_data_ptr = reinterpret_cast<const XPUT *>(filter_data_tmp);
    }

    int r = xpu::conv2d<XPUT, XPUT, XPUT, int16_t>(dev_ctx.x_context(),
                                                   input_data,
                                                   filter_data_ptr,
                                                   output_data,
                                                   batch_size,
                                                   img_c,
                                                   img_h,
                                                   img_w,
                                                   f,
                                                   ksize,
                                                   strides,
                                                   paddings,
                                                   dilations,
                                                   groups,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   is_nchw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d");
  }
};

template <typename DeviceContext, typename T>
class GemmConvGradXPUKernel : public framework::OpKernel<T> {
  using XPUT = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *input = context.Input<Tensor>("Input");
    const Tensor *output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor *input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor *filter_grad =
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
    const std::string data_format = context.Attr<std::string>("data_format");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    PADDLE_ENFORCE_EQ(
        data_format == "NDHWC",
        false,
        platform::errors::InvalidArgument(
            ("XPU doesn't support data_format is NDHWC in conv grad op.")));

    framework::DDim in_data_dims =
        phi::slice_ddim(input->dims(), 2, input->dims().size());
    framework::DDim filter_data_dims =
        phi::slice_ddim(filter.dims(), 2, filter.dims().size());
    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    std::vector<int> filter_shape = phi::vectorize<int>(filter.dims());
    UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

    int batch_size = static_cast<int>(input->dims()[0]);
    int img_c = static_cast<int>(input->dims()[1]);
    int img_h = static_cast<int>(input->dims()[2]);
    int img_w = static_cast<int>(input->dims()[3]);
    int f = static_cast<int>(filter.dims()[0]);
    bool is_nchw = true;
    if (data_format == "NHWC") {
      img_c = static_cast<int>(input->dims()[3]);
      img_h = static_cast<int>(input->dims()[1]);
      img_w = static_cast<int>(input->dims()[2]);
      is_nchw = false;
    }

    const XPUT *input_data = reinterpret_cast<const XPUT *>(input->data<T>());
    const XPUT *filter_data = reinterpret_cast<const XPUT *>(filter.data<T>());
    const XPUT *output_grad_data =
        reinterpret_cast<const XPUT *>(output_grad->data<T>());
    XPUT *input_grad_data = nullptr;
    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      input_grad_data = reinterpret_cast<XPUT *>(input_grad->data<T>());
    }
    XPUT *filter_grad_data = nullptr;
    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
      filter_grad_data = reinterpret_cast<XPUT *>(filter_grad->data<T>());
    }
    auto &dev_ctx = context.template device_context<DeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

    XPUT *filter_data_tmp;
    XPUT *filter_grad_data_tmp;
    const XPUT *filter_data_ptr = filter_data;
    XPUT *filter_grad_data_ptr = filter_grad_data;
    if (data_format == "NHWC") {
      filter_data_tmp = RAII_GUARD.alloc<XPUT>(filter.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(filter_data_tmp);
      int r = xpu::transpose<XPUT>(dev_ctx.x_context(),
                                   filter_data,
                                   filter_data_tmp,
                                   filter_shape,
                                   {0, 2, 3, 1});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
      filter_data_ptr = reinterpret_cast<const XPUT *>(filter_data_tmp);

      if (filter_grad_data != nullptr) {
        filter_grad_data_tmp = RAII_GUARD.alloc<XPUT>(filter.numel());
        PADDLE_ENFORCE_XDNN_NOT_NULL(filter_grad_data_tmp);
        filter_grad_data_ptr = filter_grad_data_tmp;
      }
    }
    int r = xpu::conv2d_grad<XPUT, XPUT, XPUT, int16_t>(dev_ctx.x_context(),
                                                        input_data,
                                                        filter_data_ptr,
                                                        output_grad_data,
                                                        input_grad_data,
                                                        filter_grad_data_ptr,
                                                        batch_size,
                                                        img_c,
                                                        img_h,
                                                        img_w,
                                                        f,
                                                        ksize,
                                                        strides,
                                                        paddings,
                                                        dilations,
                                                        groups,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        is_nchw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_grad");

    if ((filter_grad_data_ptr != nullptr) && (data_format == "NHWC")) {
      std::vector<int> filter_shape_fhwc = {
          filter_shape[0], filter_shape[2], filter_shape[3], filter_shape[1]};
      int r = xpu::transpose<XPUT>(dev_ctx.x_context(),
                                   filter_grad_data_ptr,
                                   filter_grad_data,
                                   filter_shape_fhwc,
                                   {0, 3, 1, 2});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    }
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    conv2d,
    ops::GemmConvXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::GemmConvXPUKernel<paddle::platform::XPUDeviceContext,
                           paddle::platform::float16>);
REGISTER_OP_XPU_KERNEL(
    conv2d_grad,
    ops::GemmConvGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::GemmConvGradXPUKernel<paddle::platform::XPUDeviceContext,
                               paddle::platform::float16>);
REGISTER_OP_XPU_KERNEL(
    depthwise_conv2d,
    ops::GemmConvXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    depthwise_conv2d_grad,
    ops::GemmConvGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
