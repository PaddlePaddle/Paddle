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
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
template <typename T, typename Context>
void Conv2dTransposeGradKernel(const Context& ctx,
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
  using XPUT = typename XPUTypeTrait<T>::Type;
  DenseTensor filter_ = filter;
  if (!dx && !dfilter) return;

  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;

  PADDLE_ENFORCE_EQ(
      data_format == "NHWC" || data_format == "NDHWC",
      false,
      errors::InvalidArgument(
          ("XPU do support data_format is NCHW in conv grad op.")));

  DDim in_data_dims = slice_ddim(x.dims(), 2, x.dims().size());
  DDim filter_data_dims = slice_ddim(filter_.dims(), 2, filter_.dims().size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, in_data_dims, strides, ksize);

  const int batch_size = static_cast<int>(x.dims()[0]);
  const int img_yc = static_cast<int>(x.dims()[1]);
  const int img_yh = static_cast<int>(x.dims()[2]);
  const int img_yw = static_cast<int>(x.dims()[3]);
  const int img_xc = static_cast<int>(dout.dims()[1]);
  const int img_xh = static_cast<int>(dout.dims()[2]);
  const int img_xw = static_cast<int>(dout.dims()[3]);
  if (dx) {
    ctx.template Alloc<T>(dx);
  }
  if (dfilter) {
    ctx.template Alloc<T>(dfilter);
  }

  auto x_data = reinterpret_cast<const XPUT*>(x.data<T>());
  auto filter_data = reinterpret_cast<XPUT*>(filter_.data<T>());
  auto dout_data = reinterpret_cast<const XPUT*>(dout.data<T>());
  auto dx_data = (dx ? reinterpret_cast<XPUT*>(dx->data<T>()) : nullptr);
  auto dfilter_data =
      (dfilter ? reinterpret_cast<XPUT*>(dfilter->data<T>()) : nullptr);
  int fccal_type = FCCalcType<XPUT>(ctx.x_context());
  PD_VISIT_XPU_QUANT_TYPES(XPUT, fccal_type, "conv2d_transpose_grad", [&] {
    // conv2d_tranpose_grad do not support float quantization currently
    // use int16_t quant as default.
    using RealTGEMM = typename std::
        conditional<std::is_same<TGEMM, float>::value, int16_t, TGEMM>::type;
    int ret =
        xpu::conv2d_transpose_grad<XPUT, XPUT, XPUT, RealTGEMM>(ctx.x_context(),
                                                                x_data,
                                                                filter_data,
                                                                dout_data,
                                                                dx_data,
                                                                dfilter_data,
                                                                batch_size,
                                                                img_yc,
                                                                img_yh,
                                                                img_yw,
                                                                img_xc,
                                                                img_xh,
                                                                img_xw,
                                                                ksize,
                                                                strides,
                                                                paddings_,
                                                                dilations_,
                                                                groups,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                true);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "conv2d_transpose_grad");
  });
}

template <typename T, typename Context>
void DepthwiseConv2dTransposeGradKernel(const Context& ctx,
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
  Conv2dTransposeGradKernel<T, Context>(ctx,
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
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(depthwise_conv2d_transpose_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv2dTransposeGradKernel,
                   float,
                   phi::dtype::float16) {}
