// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/conv_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

template <typename T, typename Context>
void ConvGradKernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& filter,
                    const DenseTensor& out_grad,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings_t,
                    const std::string& padding_algorithm,
                    const std::vector<int>& dilations_t,
                    int groups,
                    const std::string& data_format,
                    DenseTensor* input_grad,
                    DenseTensor* filter_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;
  // The filter and filter_grad will be reshaped in the calculations,
  // so here use an assignment operation,
  // that avoids modifying the variable in the Scope.
  if (!input_grad && !filter_grad) return;
  PADDLE_ENFORCE_EQ(
      data_format == "NDHWC",
      false,
      phi::errors::InvalidArgument(
          ("XPU doesn't support data_format is NDHWC in conv grad op.")));

  phi::DDim in_data_dims =
      common::slice_ddim(input.dims(), 2, input.dims().size());
  phi::DDim filter_data_dims =
      common::slice_ddim(filter.dims(), 2, filter.dims().size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  std::vector<int> filter_shape = common::vectorize<int>(filter.dims());
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  int batch_size = static_cast<int>(input.dims()[0]);
  int img_c = static_cast<int>(input.dims()[1]);
  int img_h = static_cast<int>(input.dims()[2]);
  int img_w = static_cast<int>(input.dims()[3]);
  int f = static_cast<int>(filter.dims()[0]);
  bool is_nchw = true;
  if (data_format == "NHWC") {
    img_c = static_cast<int>(input.dims()[3]);
    img_h = static_cast<int>(input.dims()[1]);
    img_w = static_cast<int>(input.dims()[2]);
    is_nchw = false;
  }

  const XPUType* input_data = reinterpret_cast<const XPUType*>(input.data<T>());
  const XPUType* filter_data =
      reinterpret_cast<const XPUType*>(filter.data<T>());
  const XPUType* output_grad_data =
      reinterpret_cast<const XPUType*>(out_grad.data<T>());
  XPUType* input_grad_data = nullptr;
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    input_grad_data = reinterpret_cast<XPUType*>(input_grad->data<T>());
  }
  XPUType* filter_grad_data = nullptr;
  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    filter_grad_data = reinterpret_cast<XPUType*>(filter_grad->data<T>());
  }
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  XPUType* filter_data_tmp;
  XPUType* filter_grad_data_tmp;
  const XPUType* filter_data_ptr = filter_data;
  XPUType* filter_grad_data_ptr = filter_grad_data;
  if (data_format == "NHWC") {
    filter_data_tmp = RAII_GUARD.alloc<XPUType>(filter.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(filter_data_tmp);
    int r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                    filter_data,
                                    filter_data_tmp,
                                    filter_shape,
                                    {0, 2, 3, 1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    filter_data_ptr = reinterpret_cast<const XPUType*>(filter_data_tmp);

    if (filter_grad_data != nullptr) {
      filter_grad_data_tmp = RAII_GUARD.alloc<XPUType>(filter.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(filter_grad_data_tmp);
      filter_grad_data_ptr = filter_grad_data_tmp;
    }
  }
  int fc_calc_type = FCCalcType<XPUType>();
  if (fc_calc_type == XPUFCCalcType::FC_INT32) {
    int r =
        xpu::conv2d_grad<XPUType, XPUType, XPUType, int>(dev_ctx.x_context(),
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

  } else if (fc_calc_type == XPUFCCalcType::FC_FLOAT) {
    int r =
        xpu::conv2d_grad<XPUType, XPUType, XPUType, float>(dev_ctx.x_context(),
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

  } else if (fc_calc_type == XPUFCCalcType::FC_INT32_WITH_LL) {
    int r = xpu::conv2d_grad<XPUType, XPUType, XPUType, int_with_ll_t>(
        dev_ctx.x_context(),
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
  } else {
    int r = xpu::conv2d_grad<XPUType, XPUType, XPUType, int16_t>(
        dev_ctx.x_context(),
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
  }

  if ((filter_grad_data_ptr != nullptr) && (data_format == "NHWC")) {
    std::vector<int> filter_shape_fhwc = {
        filter_shape[0], filter_shape[2], filter_shape[3], filter_shape[1]};
    int r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                    filter_grad_data_ptr,
                                    filter_grad_data,
                                    filter_shape_fhwc,
                                    {0, 3, 1, 2});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  }
}

template <typename T, typename Context>
void DepthwiseConvGradKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const DenseTensor& filter,
                             const DenseTensor& out_grad,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations,
                             const std::string& data_format,
                             DenseTensor* input_grad,
                             DenseTensor* filter_grad) {
  ConvGradKernel<T, Context>(dev_ctx,
                             input,
                             filter,
                             out_grad,
                             strides,
                             paddings,
                             padding_algorithm,
                             dilations,
                             groups,
                             data_format,
                             input_grad,
                             filter_grad);
}

template <typename T, typename Context>
void Conv3DGradKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& filter,
                      const DenseTensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings_t,
                      const std::string& padding_algorithm,
                      int groups,
                      const std::vector<int>& dilations_t,
                      const std::string& data_format,
                      DenseTensor* input_grad,
                      DenseTensor* filter_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;
  // The filter and filter_grad will be reshaped in the calculations,
  // so here use an assignment operation,
  // that avoids modifying the variable in the Scope.
  if (!input_grad && !filter_grad) return;

  phi::DDim in_data_dims =
      common::slice_ddim(input.dims(), 2, input.dims().size());
  phi::DDim filter_data_dims =
      common::slice_ddim(filter.dims(), 2, filter.dims().size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  std::vector<int> filter_shape = common::vectorize<int>(filter.dims());
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  int batch_size = static_cast<int>(input.dims()[0]);
  int img_c = static_cast<int>(input.dims()[1]);
  int img_d = static_cast<int>(input.dims()[2]);
  int img_h = static_cast<int>(input.dims()[3]);
  int img_w = static_cast<int>(input.dims()[4]);
  int f = static_cast<int>(filter.dims()[0]);
  bool is_ncdhw = true;
  if (data_format == "NDHWC") {
    img_c = static_cast<int>(input.dims()[4]);
    img_d = static_cast<int>(input.dims()[1]);
    img_h = static_cast<int>(input.dims()[2]);
    img_w = static_cast<int>(input.dims()[3]);
    is_ncdhw = false;
  }

  const XPUType* input_data = reinterpret_cast<const XPUType*>(input.data<T>());
  const XPUType* filter_data =
      reinterpret_cast<const XPUType*>(filter.data<T>());
  const XPUType* output_grad_data =
      reinterpret_cast<const XPUType*>(out_grad.data<T>());
  XPUType* input_grad_data = nullptr;
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    input_grad_data = reinterpret_cast<XPUType*>(input_grad->data<T>());
  }
  XPUType* filter_grad_data = nullptr;
  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    filter_grad_data = reinterpret_cast<XPUType*>(filter_grad->data<T>());
  }
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  XPUType* filter_data_tmp;
  XPUType* filter_grad_data_tmp;
  const XPUType* filter_data_ptr = filter_data;
  XPUType* filter_grad_data_ptr = filter_grad_data;
  if (data_format == "NDHWC") {
    filter_data_tmp = RAII_GUARD.alloc<XPUType>(filter.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(filter_data_tmp);
    int r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                    filter_data,
                                    filter_data_tmp,
                                    filter_shape,
                                    {0, 2, 3, 4, 1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    filter_data_ptr = reinterpret_cast<const XPUType*>(filter_data_tmp);

    if (filter_grad_data != nullptr) {
      filter_grad_data_tmp = RAII_GUARD.alloc<XPUType>(filter.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(filter_grad_data_tmp);
      filter_grad_data_ptr = filter_grad_data_tmp;
    }
  }
  int fc_calc_type = FCCalcType<XPUType>();
  if (fc_calc_type == XPUFCCalcType::FC_INT32) {
    int r =
        xpu::conv3d_grad<XPUType, XPUType, XPUType, int>(dev_ctx.x_context(),
                                                         input_data,
                                                         filter_data_ptr,
                                                         output_grad_data,
                                                         input_grad_data,
                                                         filter_grad_data_ptr,
                                                         batch_size,
                                                         img_c,
                                                         img_d,
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
                                                         is_ncdhw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv3d_grad");
  } else if (fc_calc_type == XPUFCCalcType::FC_FLOAT) {
    int r =
        xpu::conv3d_grad<XPUType, XPUType, XPUType, float>(dev_ctx.x_context(),
                                                           input_data,
                                                           filter_data_ptr,
                                                           output_grad_data,
                                                           input_grad_data,
                                                           filter_grad_data_ptr,
                                                           batch_size,
                                                           img_c,
                                                           img_d,
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
                                                           is_ncdhw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv3d_grad");
  } else if (fc_calc_type == XPUFCCalcType::FC_INT32_WITH_LL) {
    int r = xpu::conv3d_grad<XPUType, XPUType, XPUType, int_with_ll_t>(
        dev_ctx.x_context(),
        input_data,
        filter_data_ptr,
        output_grad_data,
        input_grad_data,
        filter_grad_data_ptr,
        batch_size,
        img_c,
        img_d,
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
        is_ncdhw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv3d_grad");
  } else {
    int r = xpu::conv3d_grad<XPUType, XPUType, XPUType, int16_t>(
        dev_ctx.x_context(),
        input_data,
        filter_data_ptr,
        output_grad_data,
        input_grad_data,
        filter_grad_data_ptr,
        batch_size,
        img_c,
        img_d,
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
        is_ncdhw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv3d_grad");
  }

  if ((filter_grad_data_ptr != nullptr) && (data_format == "NDHWC")) {
    std::vector<int> filter_shape_fhwc = {filter_shape[0],
                                          filter_shape[2],
                                          filter_shape[3],
                                          filter_shape[4],
                                          filter_shape[1]};
    int r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                    filter_grad_data_ptr,
                                    filter_grad_data,
                                    filter_shape_fhwc,
                                    {0, 4, 1, 2, 3});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ConvGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(depthwise_conv2d_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvGradKernel,
                   float) {}
PD_REGISTER_KERNEL(conv3d_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::Conv3DGradKernel,
                   float,
                   phi::dtype::float16) {}
