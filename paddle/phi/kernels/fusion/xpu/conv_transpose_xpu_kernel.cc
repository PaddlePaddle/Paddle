// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
namespace fusion {
template <typename T, typename Context>
void Conv2dTransposeXPUKernel(const Context& ctx,
                              const DenseTensor& x,
                              const paddle::optional<DenseTensor>& x_max,
                              const DenseTensor& filter,
                              const DenseTensor& filter_max,
                              const paddle::optional<DenseTensor>& bias,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::vector<int>& output_padding,
                              const IntArray& output_size,
                              const std::string& padding_algorithm,
                              int groups,
                              const std::vector<int>& dilations,
                              const std::string& data_format,
                              bool has_bias,
                              bool with_act,
                              const std::string& act_type,
                              DenseTensor* out,
                              DenseTensor* out_max) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  ctx.template Alloc<T>(out);
  ctx.template Alloc<float>(out_max);
  bool is_nchw;
  is_nchw = (data_format == "NHWC") ? false : true;

  DDim in_data_dims = slice_ddim(x.dims(), 2, x.dims().size());  // hw
  DDim filter_data_dims = slice_ddim(filter.dims(), 2, filter.dims().size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, in_data_dims, strides, ksize);

  const int batch_size = static_cast<int>(x.dims()[0]);
  const int img_yc = static_cast<int>(x.dims()[1]);
  const int img_xc = static_cast<int>(out->dims()[1]);
  const int img_xh = static_cast<int>(out->dims()[2]);
  const int img_xw = static_cast<int>(out->dims()[3]);
  auto act = xpu::Activation_t::LINEAR;
  if (with_act) {
    if (act_type == "relu") {
      act = xpu::Activation_t::RELU;
    }
  }
  auto bias_data =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  auto x_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  auto filter_max_data = filter_max.data<float>();

  int r = xpu::conv2d_transpose_fusion_v2<XPUType, int16_t, XPUType, int16_t>(
      ctx.x_context(),
      reinterpret_cast<const XPUType*>(x.data<T>()),
      filter.data<int16_t>(),
      reinterpret_cast<XPUType*>(out->data<T>()),
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
      x_max_data,
      filter_max_data,
      out_max->data<float>(),
      bias_data,
      act,
      is_nchw);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_fusion_v2");
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_transpose_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::Conv2dTransposeXPUKernel,
                   float,
                   phi::dtype::float16) {}
