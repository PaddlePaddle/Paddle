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
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class DeformableConvXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* offset = ctx.Input<Tensor>("Offset");
    auto* mask = ctx.Input<Tensor>("Mask");
    Tensor filter = *ctx.Input<Tensor>("Filter");
    Tensor* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    const int groups = ctx.Attr<int>("groups");
    const int deformable_groups = ctx.Attr<int>("deformable_groups");
    const int im2col_step = ctx.Attr<int>("im2col_step");
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    const std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    const std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    PADDLE_ENFORCE_EQ(
        deformable_groups == 1, true,
        platform::errors::InvalidArgument((
            "XPU only support deformable_groups == 1 in deformable_conv op.")));
    PADDLE_ENFORCE_EQ(
        groups == 1, true,
        platform::errors::InvalidArgument(
            ("XPU only support groups == 1 in deformable_conv op.")));
    PADDLE_ENFORCE_EQ(filter.dims()[2] <= 8 && filter.dims()[3] <= 8, true,
                      platform::errors::InvalidArgument(
                          "Filter high and weight should less than 8 on xpu "
                          "in deformable_conv op."));

    const int batch_size = static_cast<int>(input->dims()[0]);
    std::vector<int64_t> output_shape_vec(phi::vectorize(output->dims()));

    const T* input_ptr = input->data<T>();
    const T* filter_ptr = filter.data<T>();
    const float* offset_ptr = offset->data<T>();
    const float* mask_ptr = mask->data<T>();
    T* output_prt = output->data<T>();

    // set zeros for d_table_data
    const int zero = 0;
    int r = xpu::constant<T>(dev_ctx.x_context(), output_prt, output->numel(),
                             zero);
    PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                      platform::errors::External(
                          "XPU API return wrong value[%d], please check where "
                          "Baidu Kunlun Card is properly installed.",
                          r));
    int input_dim = input->numel() / input->dims()[0];
    int input_offset_dim = offset->numel() / offset->dims()[0];
    int input_mask_dim = mask->numel() / mask->dims()[0];
    int output_dim =
        output_shape_vec[1] * output_shape_vec[2] * output_shape_vec[3];
    std::vector<int> ksize{static_cast<int>(filter.dims()[2]),
                           static_cast<int>(filter.dims()[3])};
    int n = im2col_step;
    int c = input->dims()[1];
    int h = input->dims()[2];
    int w = input->dims()[3];
    int f = filter.dims()[0];

    for (int i = 0; i < batch_size / im2col_step; ++i) {
      int r = xpu::deformable_conv<float, float, float, int>(
          dev_ctx.x_context(), input_ptr + i * im2col_step * input_dim,
          filter_ptr, offset_ptr + i * im2col_step * input_offset_dim,
          mask_ptr + i * im2col_step * input_mask_dim,
          output_prt + i * im2col_step * output_dim, n, c, h, w, f, ksize,
          strides, paddings, dilations, groups, deformable_groups, nullptr,
          nullptr, nullptr, true);
      PADDLE_ENFORCE_EQ(
          r, XPU_SUCCESS,
          platform::errors::External(
              "XPU deformable_conv kernel return wrong value[%d].", r));
    }
  }
};

template <typename DeviceContext, typename T>
class DeformableConvGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));
    Tensor* offset_grad = ctx.Output<Tensor>(framework::GradVarName("Offset"));
    Tensor* mask_grad = ctx.Output<Tensor>(framework::GradVarName("Mask"));
    T* dx_data = nullptr;
    T* dw_data = nullptr;
    T* dmask_data = nullptr;
    T* doffset_data = nullptr;

    if (input_grad != nullptr) {
      input_grad->mutable_data<T>(ctx.GetPlace());
      dx_data = input_grad->data<T>();
    }
    if (filter_grad != nullptr) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
      dw_data = filter_grad->data<T>();
    }
    if (offset_grad != nullptr) {
      offset_grad->mutable_data<T>(ctx.GetPlace());
      doffset_data = offset_grad->data<T>();
    }
    if (mask_grad != nullptr) {
      mask_grad->mutable_data<T>(ctx.GetPlace());
      dmask_data = mask_grad->data<T>();
    }

    const Tensor* input = ctx.Input<Tensor>("Input");
    Tensor offset = *ctx.Input<Tensor>("Offset");
    Tensor mask = *ctx.Input<Tensor>("Mask");
    Tensor filter = *ctx.Input<Tensor>("Filter");

    int groups = ctx.Attr<int>("groups");
    int deformable_groups = ctx.Attr<int>("deformable_groups");
    int im2col_step = ctx.Attr<int>("im2col_step");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    PADDLE_ENFORCE_EQ(
        deformable_groups == 1, true,
        platform::errors::InvalidArgument((
            "XPU only support deformable_groups == 1 in deformable_conv op.")));
    PADDLE_ENFORCE_EQ(
        groups == 1, true,
        platform::errors::InvalidArgument(
            ("XPU only support groups == 1 in deformable_conv op.")));
    PADDLE_ENFORCE_EQ(filter.dims()[2] <= 8 && filter.dims()[3] <= 8, true,
                      platform::errors::InvalidArgument(
                          "Filter high and weight should less than 8 on xpu "
                          "in deformable_conv op."));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    const int batch_size = static_cast<int>(input->dims()[0]);
    std::vector<int64_t> output_shape_vec(phi::vectorize(output_grad->dims()));
    const T* output_grad_ptr = output_grad->data<T>();
    const T* input_ptr = input->data<T>();
    const T* filter_ptr = filter.data<T>();
    const float* offset_ptr = offset.data<float>();
    const float* mask_ptr = mask.data<float>();
    if (dx_data == nullptr) {
      PADDLE_ENFORCE_EQ(xpu_malloc(reinterpret_cast<void**>(&dx_data),
                                   input->numel() * sizeof(T)),
                        XPU_SUCCESS, platform::errors::ResourceExhausted(
                                         "XPU has no enough memory"));
    }
    if (dw_data == nullptr) {
      PADDLE_ENFORCE_EQ(xpu_malloc(reinterpret_cast<void**>(&dw_data),
                                   filter.numel() * sizeof(T)),
                        XPU_SUCCESS, platform::errors::ResourceExhausted(
                                         "XPU has no enough memory"));
    }
    if (doffset_data == nullptr) {
      PADDLE_ENFORCE_EQ(xpu_malloc(reinterpret_cast<void**>(&doffset_data),
                                   offset.numel() * sizeof(T)),
                        XPU_SUCCESS, platform::errors::ResourceExhausted(
                                         "XPU has no enough memory"));
    }
    if (dmask_data == nullptr) {
      PADDLE_ENFORCE_EQ(xpu_malloc(reinterpret_cast<void**>(&dmask_data),
                                   mask.numel() * sizeof(T)),
                        XPU_SUCCESS, platform::errors::ResourceExhausted(
                                         "XPU has no enough memory"));
    }

    int input_dim = input->numel() / input->dims()[0];
    int input_offset_dim = offset.numel() / offset.dims()[0];
    int input_mask_dim = mask.numel() / mask.dims()[0];
    int output_dim =
        output_shape_vec[1] * output_shape_vec[2] * output_shape_vec[3];
    std::vector<int> ksize{static_cast<int>(filter.dims()[2]),
                           static_cast<int>(filter.dims()[3])};
    int n = im2col_step;
    int c = input->dims()[1];
    int h = input->dims()[2];
    int w = input->dims()[3];
    int f = filter.dims()[0];

    T* filter_grad_tmp = nullptr;
    PADDLE_ENFORCE_EQ(xpu_malloc(reinterpret_cast<void**>(&filter_grad_tmp),
                                 filter_grad->numel() * sizeof(T)),
                      XPU_SUCCESS, platform::errors::ResourceExhausted(
                                       "XPU has no enough memory"));

    // set zeros for d_table_data
    const int zero = 0;
    int r_dx =
        xpu::constant<T>(dev_ctx.x_context(), dx_data, input->numel(), zero);
    int r_dw =
        xpu::constant<T>(dev_ctx.x_context(), dw_data, filter.numel(), zero);
    int r_doffset = xpu::constant<T>(dev_ctx.x_context(), doffset_data,
                                     offset.numel(), zero);
    int r_dmask =
        xpu::constant<T>(dev_ctx.x_context(), dmask_data, mask.numel(), zero);
    int r_filter = xpu::constant<T>(dev_ctx.x_context(), filter_grad_tmp,
                                    filter.numel(), zero);
    auto ret = (r_dx == xpu::Error_t::SUCCESS) && (r_dx == r_dw) &&
               (r_dx == r_doffset) && (r_dx == r_dmask) && (r_dx == r_filter);
    PADDLE_ENFORCE_EQ(ret, true,
                      platform::errors::External(
                          "XPU API return wrong value, please check where "
                          "Baidu Kunlun Card is properly installed."));

    for (int i = 0; i < batch_size / im2col_step; ++i) {
      int r = xpu::deformable_conv_grad<float, float, float, int>(
          dev_ctx.x_context(), input_ptr + i * im2col_step * input_dim,
          filter_ptr, offset_ptr + i * im2col_step * input_offset_dim,
          mask_ptr + i * im2col_step * input_mask_dim,
          output_grad_ptr + i * im2col_step * output_dim,
          dx_data + i * im2col_step * input_dim, filter_grad_tmp,
          doffset_data + i * im2col_step * input_offset_dim,
          dmask_data + i * im2col_step * input_mask_dim, n, c, h, w, f, ksize,
          strides, paddings, dilations, groups, deformable_groups, nullptr,
          nullptr, nullptr, nullptr, nullptr, true);
      PADDLE_ENFORCE_EQ(
          r, XPU_SUCCESS,
          platform::errors::External(
              "XPU deformable_conv_grad kernel return wrong value[%d].", r));
      r = baidu::xpu::api::add<T>(dev_ctx.x_context(), filter_grad_tmp, dw_data,
                                  dw_data, filter.numel());
      PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                        platform::errors::External(
                            "XPU add kernel return wrong value[%d].", r));
    }

    dev_ctx.Wait();
    xpu_free(filter_grad_tmp);
    if (input_grad == nullptr) {
      xpu_free(dx_data);
    }
    if (filter_grad == nullptr) {
      xpu_free(dw_data);
    }
    if (offset_grad == nullptr) {
      xpu_free(doffset_data);
    }
    if (mask_grad == nullptr) {
      xpu_free(dmask_data);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using XPUDeviceContext = paddle::platform::XPUDeviceContext;

REGISTER_OP_XPU_KERNEL(deformable_conv,
                       ops::DeformableConvXPUKernel<XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    deformable_conv_grad,
    ops::DeformableConvGradXPUKernel<XPUDeviceContext, float>);

#endif
