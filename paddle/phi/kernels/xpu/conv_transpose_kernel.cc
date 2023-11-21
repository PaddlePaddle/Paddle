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

#include "paddle/phi/kernels/conv_transpose_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

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

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const IntArray& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  using XPUT = typename XPUTypeTrait<T>::Type;

  ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_EQ(
      data_format == "NHWC" || data_format == "NDHWC",
      false,
      errors::InvalidArgument(
          ("XPU do support data_format is NCHW in conv_transpose op.")));

  DDim in_data_dims = slice_ddim(x.dims(), 2, x.dims().size());
  DDim filter_data_dims = slice_ddim(filter.dims(), 2, filter.dims().size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);

  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, in_data_dims, strides, ksize);

  const int batch_size = static_cast<int>(x.dims()[0]);
  const int img_yc = static_cast<int>(x.dims()[1]);
  const int img_xc = static_cast<int>(out->dims()[1]);
  const int img_xh = static_cast<int>(out->dims()[2]);
  const int img_xw = static_cast<int>(out->dims()[3]);

  auto x_data = reinterpret_cast<const XPUT*>(x.data<T>());
  auto filter_data = reinterpret_cast<const XPUT*>(filter.data<T>());
  auto out_data = reinterpret_cast<XPUT*>(out->data<T>());

  int fccal_type = FCCalcType<XPUT>(ctx.x_context());
  PD_VISIT_XPU_QUANT_TYPES(XPUT, fccal_type, "conv2d_transpose", [&] {
    // conv2d_transpose_v2 do not support int_with_ll quantization, fallback to
    // int.
    using RealTGEMM =
        typename std::conditional<std::is_same<TGEMM, int_with_ll_t>::value,
                                  int,
                                  TGEMM>::type;
    int ret =
        xpu::conv2d_transpose_v2<XPUT, XPUT, XPUT, RealTGEMM>(ctx.x_context(),
                                                              x_data,
                                                              filter_data,
                                                              out_data,
                                                              batch_size,
                                                              img_yc,
                                                              img_xh,
                                                              img_xw,
                                                              img_xc,
                                                              ksize,
                                                              strides,
                                                              paddings_,
                                                              dilations_,
                                                              groups,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              true);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "conv2d_transpose_v2");
  });
}
template <typename T, typename Context>
void DepthwiseConv2dTransposeKernel(const Context& ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& filter,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& paddings,
                                    const std::vector<int>& output_padding,
                                    const IntArray& output_size,
                                    const std::string& padding_algorithm,
                                    int groups,
                                    const std::vector<int>& dilations,
                                    const std::string& data_format,
                                    DenseTensor* out) {
  Conv2dTransposeKernel<T, Context>(ctx,
                                    x,
                                    filter,
                                    strides,
                                    paddings,
                                    output_padding,
                                    output_size,
                                    padding_algorithm,
                                    groups,
                                    dilations,
                                    data_format,
                                    out);
}

}  // namespace phi
PD_REGISTER_KERNEL(depthwise_conv2d_transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv2dTransposeKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv2d_transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeKernel,
                   float,
                   phi::dtype::float16) {}
