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
#include "paddle/fluid/operators/conv_transpose_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/device_wrapper.h"
#ifdef PADDLE_WITH_XPU
namespace paddle {
namespace operators {

// target_len == 2 || target_len == 4
inline std::vector<int> vector_extend(const std::vector<int>& src,
                                      int target_len) {
  if (target_len == 2 && src.size() == 1) {
    return {src[0], src[0]};
  }
  if (target_len == 4 && src.size() == 1) {
    return {src[0], src[0], src[0], src[0]};
  }
  if (target_len == 4 && src.size() == 2) {
    return {src[0], src[0], src[1], src[1]};
  }
  return src;
}

template <typename DeviceContext, typename T>
class Conv2DTransposeXPUKernel : public framework::OpKernel<T> {
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
    const std::string data_format = context.Attr<std::string>("data_format");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    PADDLE_ENFORCE_EQ(
        data_format == "NHWC" || data_format == "NDHWC", false,
        platform::errors::InvalidArgument(
            ("XPU do support data_format is NCHW in conv_transpose op.")));

    framework::DDim in_data_dims =
        framework::slice_ddim(input->dims(), 2, input->dims().size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter.dims(), 2, filter.dims().size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    const int batch_size = static_cast<int>(input->dims()[0]);
    const int img_yc = static_cast<int>(input->dims()[1]);
    const int img_yh = static_cast<int>(input->dims()[2]);
    const int img_yw = static_cast<int>(input->dims()[3]);
    const int img_xc = static_cast<int>(output->dims()[1]);
    const int img_xh = static_cast<int>(output->dims()[2]);
    const int img_xw = static_cast<int>(output->dims()[3]);

    {
      std::vector<int> ksize_check = vector_extend(ksize, 2);
      std::vector<int> stride_check = vector_extend(strides, 2);
      std::vector<int> pad_check = vector_extend(paddings, 4);
      std::vector<int> dilation_check = vector_extend(dilations, 2);

      int xh_check = (img_yh - 1) * stride_check[0] - pad_check[0] -
                     pad_check[1] +
                     (dilation_check[0] * (ksize_check[0] - 1) + 1);
      int xw_check = (img_yw - 1) * stride_check[1] - pad_check[2] -
                     pad_check[3] +
                     (dilation_check[1] * (ksize_check[1] - 1) + 1);

      PADDLE_ENFORCE_EQ(
          xh_check == img_xh && xw_check == img_xw, true,
          platform::errors::InvalidArgument(
              ("XPU output size check error in conv_transpose op.")));
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::conv2d_transpose<float, float, float, int16_t>(
        dev_ctx.x_context(), input->data<float>(), filter.data<float>(),
        output->data<float>(), batch_size, img_yc, img_yh, img_yw, img_xc,
        ksize, strides, paddings, dilations, groups, nullptr, nullptr, nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose");
  }
};

template <typename DeviceContext, typename T>
class Conv2DTransposeGradXPUKernel : public framework::OpKernel<T> {
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
    const std::string data_format = context.Attr<std::string>("data_format");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    PADDLE_ENFORCE_EQ(
        data_format == "NHWC" || data_format == "NDHWC", false,
        platform::errors::InvalidArgument(
            ("XPU do support data_format is NCHW in conv grad op.")));

    framework::DDim in_data_dims =
        framework::slice_ddim(input->dims(), 2, input->dims().size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter.dims(), 2, filter.dims().size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    const int batch_size = static_cast<int>(input->dims()[0]);
    const int img_yc = static_cast<int>(input->dims()[1]);
    const int img_yh = static_cast<int>(input->dims()[2]);
    const int img_yw = static_cast<int>(input->dims()[3]);
    const int img_xc = static_cast<int>(output_grad->dims()[1]);
    const int img_xh = static_cast<int>(output_grad->dims()[2]);
    const int img_xw = static_cast<int>(output_grad->dims()[3]);
    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
    }
    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::conv2d_transpose_grad<float, float, float, int16_t>(
        dev_ctx.x_context(), input->data<T>(), filter.data<T>(),
        output_grad->data<T>(), input_grad ? input_grad->data<T>() : nullptr,
        filter_grad ? filter_grad->data<T>() : nullptr, batch_size, img_yc,
        img_yh, img_yw, img_xc, img_xh, img_xw, ksize, strides, paddings,
        dilations, groups, nullptr, nullptr, nullptr, nullptr, nullptr, true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_grad");
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    conv2d_transpose,
    ops::Conv2DTransposeXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(conv2d_transpose_grad,
                       ops::Conv2DTransposeGradXPUKernel<
                           paddle::platform::XPUDeviceContext, float>);
#endif
