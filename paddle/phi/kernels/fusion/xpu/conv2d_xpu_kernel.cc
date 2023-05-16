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

namespace phi {
namespace fusion {

template <typename T, typename Context>
void Conv2dXPUKernel(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& x_max,
                     const DenseTensor& filter,
                     const DenseTensor& filter_max,
                     const paddle::optional<DenseTensor>& bias,
                     const paddle::optional<DenseTensor>& branch,
                     const paddle::optional<DenseTensor>& branch_max,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const std::string& padding_algorithm,
                     int groups,
                     bool has_bias,
                     bool has_branch,
                     int act_type,
                     float act_param,
                     DenseTensor* out,
                     DenseTensor* out_max) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto input_dims = x.dims();
  auto filter_dims = filter.dims();
  // update paddings and dilations accoring to padding_algorithm
  std::vector<int> paddings_vec = paddings;
  std::vector<int> dilations_vec = dilations;
  DDim in_data_dims = phi::slice_ddim(input_dims, 2, input_dims.size());
  DDim filter_data_dims = phi::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(&paddings_vec,
                                &dilations_vec,
                                padding_algorithm,
                                in_data_dims,
                                strides,
                                ksize);

  int batch = static_cast<int>(input_dims[0]);
  int in_c = static_cast<int>(input_dims[1]);
  int in_h = static_cast<int>(input_dims[2]);
  int in_w = static_cast<int>(input_dims[3]);
  int out_c = static_cast<int>(filter_dims[0]);
  int win_h = static_cast<int>(filter_dims[2]);
  int win_w = static_cast<int>(filter_dims[3]);

  auto* input_data = reinterpret_cast<const XPUType*>(x.data<T>());
  const float* input_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  auto* branch_data =
      branch.get_ptr() == nullptr
          ? nullptr
          : reinterpret_cast<const XPUType*>(branch.get_ptr()->data<T>());
  const float* branch_max_data = branch_max.get_ptr() == nullptr
                                     ? nullptr
                                     : branch_max.get_ptr()->data<float>();
  const float* bias_data =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  xpu::Activation_t act(static_cast<xpu::Activation_t::act_enum>(act_type));
  if (act_type == xpu::Activation_t::LEAKY_RELU) {
    act.leaky_alpha = act_param;
  } else if (act_type == xpu::Activation_t::HARD_SIGMOID) {
    act.hard_sigmoid_slope = act_param;
  }
  int r =
      xpu::conv2d_fusion<XPUType, int16_t, XPUType, int16_t>(  // TX/TW/TY/TGEMM
          /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
          /* const TX* input */ input_data,
          /* const TW* filter */ filter.data<int16_t>(),
          /* TY* output */ out_data,
          /* int64_t n */ batch,
          /* int64_t ic */ in_c,
          /* int64_t h */ in_h,
          /* int64_t w */ in_w,
          /* int64_t oc */ out_c,
          /* const std::vector<int>& ksize */ std::vector<int>{win_h, win_w},
          /* const std::vector<int>& strides */ strides,
          /* const std::vector<int>& paddings */ paddings_vec,
          /* const std::vector<int>& dilations */ dilations_vec,
          /* int64_t groups */ groups,
          /* const float* in_maxptr */ input_max_data,
          /* const float* filter_maxptr */ filter_max.data<float>(),
          /* float* out_maxptr */ ctx.template Alloc<float>(out_max),
          /* bool is_nchw */ true,
          /* const float* bias */ bias_data,
          /* const TY* branch */ branch_data,
          /* const baidu::xpu::api::Activation_t& act */ act,
          /* const float* branch_maxptr */ branch_max_data,
          /* const float* scale */ nullptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_xpu");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::Conv2dXPUKernel,
                   float,
                   phi::dtype::float16) {}
