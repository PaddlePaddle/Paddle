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

#include "paddle/phi/kernels/conv_transpose_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
template <typename T, typename Context>
void Conv2dTransposeGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& filter,
                               const DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const IntArray& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               DenseTensor* dx,
                               DenseTensor* dfilter) {
  // The filter and dfilter will be reshaped in the calculations,
  // so here use an assignment operation,
  // that avoids modifying the variable in the Scope.
  DenseTensor filter_ = filter;
  if (!dx && !dfilter) return;
  // 0-size
  if (x.numel() == 0) {
    if (dx) dev_ctx.template Alloc<T>(dx);
    if (dfilter) {
      phi::Full<T, Context>(dev_ctx,
                            phi::IntArray(common::vectorize(dfilter->dims())),
                            0,
                            dfilter);
    }
    return;
  }
  if (filter.numel() == 0) {
    if (dfilter) dev_ctx.template Alloc<T>(dfilter);
    if (dx) {
      phi::Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(dx->dims())), 0, dx);
    }
    return;
  }
  std::vector<int64_t> strides_ =
      std::vector<int64_t>(strides.begin(), strides.end());
  std::vector<int64_t> paddings_ =
      std::vector<int64_t>(paddings.begin(), paddings.end());
  std::vector<int64_t> dilations_ =
      std::vector<int64_t>(dilations.begin(), dilations.end());

  PADDLE_ENFORCE_EQ(
      data_format == "NHWC" || data_format == "NDHWC",
      false,
      errors::InvalidArgument(
          ("XPU do support data_format is NCHW in conv grad op.")));

  DDim in_data_dims = slice_ddim(x.dims(), 2, x.dims().size());
  DDim filter_data_dims = slice_ddim(filter_.dims(), 2, filter_.dims().size());
  std::vector<int64_t> ksize = common::vectorize<int64_t>(filter_data_dims);
  UpdatePaddingAndDilation(&paddings_,
                           &dilations_,
                           padding_algorithm,
                           in_data_dims,
                           strides_,
                           ksize);

  const int64_t batch_size = x.dims()[0];
  const int64_t img_yc = x.dims()[1];
  const int64_t img_yh = x.dims()[2];
  const int64_t img_yw = x.dims()[3];
  const int64_t img_xc = dout.dims()[1];
  const int64_t img_xh = dout.dims()[2];
  const int64_t img_xw = dout.dims()[3];
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
  }
  if (dfilter) {
    dev_ctx.template Alloc<T>(dfilter);
  }
  int fc_calc_type = FCCalcType<T>();
  if (fc_calc_type == XPUFCCalcType::FC_INT32 ||
      fc_calc_type == XPUFCCalcType::FC_INT32_WITH_LL) {
    // xpu api do not support int31 quantization now.
    int r = xpu::conv2d_transpose_grad<float, float, float, int_with_ll_t>(
        dev_ctx.x_context(),
        x.data<T>(),
        filter_.data<T>(),
        dout.data<T>(),
        dx ? dx->data<T>() : nullptr,
        dfilter ? dfilter->data<T>() : nullptr,
        batch_size,
        img_yc,
        img_yh,
        img_yw,
        img_xc,
        img_xh,
        img_xw,
        ksize,
        strides_,
        paddings_,
        dilations_,
        groups,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_grad");
  } else {
    int r = xpu::conv2d_transpose_grad<float, float, float, int16_t>(
        dev_ctx.x_context(),
        x.data<T>(),
        filter_.data<T>(),
        dout.data<T>(),
        dx ? dx->data<T>() : nullptr,
        dfilter ? dfilter->data<T>() : nullptr,
        batch_size,
        img_yc,
        img_yh,
        img_yw,
        img_xc,
        img_xh,
        img_xw,
        ksize,
        strides_,
        paddings_,
        dilations_,
        groups,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_grad");
  }
}

template <typename T, typename Context>
void DepthwiseConv2dTransposeGradKernel(const Context& dev_ctx,
                                        const DenseTensor& x,
                                        const DenseTensor& filter,
                                        const DenseTensor& dout,
                                        const std::vector<int>& strides,
                                        const std::vector<int>& paddings,
                                        const std::vector<int>& output_padding,
                                        const IntArray& output_size,
                                        const std::string& padding_algorithm,
                                        int groups,
                                        const std::vector<int>& dilations,
                                        const std::string& data_format,
                                        DenseTensor* dx,
                                        DenseTensor* dfilter) {
  Conv2dTransposeGradKernel<T, Context>(dev_ctx,
                                        x,
                                        filter,
                                        dout,
                                        strides,
                                        paddings,
                                        output_padding,
                                        output_size,
                                        padding_algorithm,
                                        groups,
                                        dilations,
                                        data_format,
                                        dx,
                                        dfilter);
}
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_transpose_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeGradKernel,
                   float) {}
PD_REGISTER_KERNEL(depthwise_conv2d_transpose_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv2dTransposeGradKernel,
                   float) {}
