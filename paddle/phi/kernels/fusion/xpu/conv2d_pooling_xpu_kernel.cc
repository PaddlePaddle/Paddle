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

template <typename T_X,
          typename T_W,
          typename T_OUT,
          typename T_GEMM,
          typename Context>
void Conv2dPoolingXPUKernelImpl(const Context& ctx,
                                const DenseTensor& x,
                                const paddle::optional<DenseTensor>& x_max,
                                const DenseTensor& filter,
                                const DenseTensor& filter_max,
                                const paddle::optional<DenseTensor>& bias,
                                const std::vector<int>& paddings,
                                const std::vector<int>& dilations,
                                const std::vector<int>& strides,
                                const std::string& padding_algorithm,
                                int groups,
                                int act_type,
                                float act_param,
                                const std::vector<int>& pool2d_paddings,
                                const std::vector<int>& pool2d_strides,
                                const std::vector<int>& pool2d_ksize,
                                bool is_avg,
                                DenseTensor* out,
                                DenseTensor* out_max) {
  using XPUTypeX = typename XPUTypeTrait<T_X>::Type;
  using XPUTypeW = typename XPUTypeTrait<T_W>::Type;
  using XPUTypeOut = typename XPUTypeTrait<T_OUT>::Type;
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

  auto* input_data = reinterpret_cast<const XPUTypeX*>(x.data<T_X>());
  const float* input_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  auto* filter_data = reinterpret_cast<const XPUTypeW*>(filter.data<T_W>());
  auto* filter_max_data = filter_max.data<float>();

  const float* bias_data =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  auto* out_data =
      reinterpret_cast<XPUTypeOut*>(ctx.template Alloc<T_OUT>(out));
  auto* out_max_data = ctx.template Alloc<float>(out_max);
  xpu::Activation_t act(static_cast<xpu::Activation_t::act_enum>(act_type));
  if (act_type == xpu::Activation_t::LEAKY_RELU) {
    act.leaky_alpha = act_param;
  } else if (act_type == xpu::Activation_t::HARD_SIGMOID) {
    act.hard_sigmoid_slope = act_param;
  }
  // for std::vector<int64_t>
  std::vector<int64_t> conv_stride(std::begin(strides), std::end(strides));
  std::vector<int64_t> conv_pad(std::begin(paddings_vec),
                                std::end(paddings_vec));
  std::vector<int64_t> conv_dilation(std::begin(dilations_vec),
                                     std::end(dilations_vec));
  std::vector<int64_t> pool_ksize(std::begin(pool2d_ksize),
                                  std::end(pool2d_ksize));
  std::vector<int64_t> pool_stride(std::begin(pool2d_strides),
                                   std::end(pool2d_strides));
  std::vector<int64_t> pool_pad(std::begin(pool2d_paddings),
                                std::end(pool2d_paddings));
  int r = xpu::conv2d_with_pooling<XPUTypeX,
                                   XPUTypeW,
                                   XPUTypeOut,
                                   T_GEMM>(  // TX/TW/TY/TGEMM
      /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
      /* const TX* x */ input_data,
      /* const TW* weight */ filter_data,
      /* TY* y */ out_data,
      /* int64_t n */ batch,
      /* int64_t c */ in_c,
      /* int64_t xh */ in_h,
      /* int64_t xw */ in_w,
      /* int64_t f */ out_c,
      /* const std::vector<int64_t>& conv_ksize */
      std::vector<int64_t>{win_h, win_w},
      /* const std::vector<int64_t>& conv_stride */ conv_stride,
      /* const std::vector<int64_t>& conv_pad */ conv_pad,
      /* const std::vector<int64_t>& conv_dilation */ conv_dilation,
      /* const std::vector<int64_t>& pool_ksize */ pool_ksize,
      /* const std::vector<int64_t>& pool_stride */ pool_stride,
      /* const std::vector<int64_t>& pool_pad */ pool_pad,
      /* bool count_include_pad */ false,
      /* bool is_avg */ is_avg,
      /* bool is_nchw */ true,
      /* const float* x_maxptr */ input_max_data,
      /* const float* w_maxptr */ filter_max_data,
      /* float* y_maxptr */ out_max_data,
      /* const float* bias */ bias_data,
      /* const baidu::xpu::api::Activation_t& act */ act);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_pooling_xpu");
}

#define CONV2D_POOLING_XPU_KERNEL_IMPL(                  \
    x_dtype_, w_dtype_, out_dtype_, gemm_dtype_)         \
  Conv2dPoolingXPUKernelImpl<x_dtype_,                   \
                             w_dtype_,                   \
                             out_dtype_,                 \
                             gemm_dtype_,                \
                             Context>(ctx,               \
                                      x,                 \
                                      x_max,             \
                                      filter,            \
                                      filter_max,        \
                                      bias,              \
                                      paddings,          \
                                      dilations,         \
                                      strides,           \
                                      padding_algorithm, \
                                      groups,            \
                                      act_type,          \
                                      act_param,         \
                                      pool2d_paddings,   \
                                      pool2d_strides,    \
                                      pool2d_ksize,      \
                                      is_avg,            \
                                      out,               \
                                      out_max);

template <typename T, typename Context>
void Conv2dPoolingXPUKernel(const Context& ctx,
                            const DenseTensor& x,
                            const paddle::optional<DenseTensor>& x_max,
                            const DenseTensor& filter,
                            const DenseTensor& filter_max,
                            const paddle::optional<DenseTensor>& bias,
                            const std::vector<int>& paddings,
                            const std::vector<int>& dilations,
                            const std::vector<int>& strides,
                            const std::string& padding_algorithm,
                            int groups,
                            int act_type,
                            float act_param,
                            DataType out_dtype,
                            const std::vector<int>& pool2d_paddings,
                            const std::vector<int>& pool2d_strides,
                            const std::vector<int>& pool2d_ksize,
                            bool is_avg,
                            DenseTensor* out,
                            DenseTensor* out_max) {
  if (out_dtype == DataType::FLOAT32) {
    CONV2D_POOLING_XPU_KERNEL_IMPL(float, int16_t, float, int16_t);
  } else if (out_dtype == DataType::FLOAT16) {
    CONV2D_POOLING_XPU_KERNEL_IMPL(
        dtype::float16, int16_t, dtype::float16, int16_t);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented("Not support out_dtype is %s.",
                                            DataTypeToString(out_dtype)));
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_pooling_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::Conv2dPoolingXPUKernel,
                   float,
                   phi::dtype::float16) {}
