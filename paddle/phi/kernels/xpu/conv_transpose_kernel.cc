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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

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
  // The filter will be reshaped in the calculations,
  // so here use an assignment operation,
  // that avoids modifying the variable in the Scope.
  DenseTensor filter_ = filter;

  ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_EQ(
      data_format == "NHWC" || data_format == "NDHWC",
      false,
      errors::InvalidArgument(
          ("XPU do support data_format is NCHW in conv_transpose op.")));

  DDim in_data_dims = slice_ddim(x.dims(), 2, x.dims().size());
  DDim filter_data_dims = slice_ddim(filter_.dims(), 2, filter_.dims().size());
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

  int r = xpu::conv2d_transpose_v2<float, float, float, int16_t>(
      ctx.x_context(),
      x.data<float>(),
      filter_.data<float>(),
      out->data<float>(),
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
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_v2");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    conv2d_transpose, XPU, ALL_LAYOUT, phi::Conv2dTransposeKernel, float) {}
