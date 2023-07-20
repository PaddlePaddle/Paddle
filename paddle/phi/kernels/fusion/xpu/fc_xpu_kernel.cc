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

namespace phi {
namespace fusion {

template <typename T_X,
          typename T_W,
          typename T_OUT,
          typename T_GEMM,
          typename Context>
void FcXPUKernelImpl(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& x_max,
                     const DenseTensor& w,
                     const DenseTensor& w_max,
                     const paddle::optional<DenseTensor>& bias,
                     int in_num_col_dims,
                     bool transpose_x,
                     float alpha,
                     float beta,
                     int act_type,
                     float act_alpha,
                     DenseTensor* out,
                     DenseTensor* out_max) {
  using XPUTypeX = typename XPUTypeTrait<T_X>::Type;
  using XPUTypeW = typename XPUTypeTrait<T_W>::Type;
  using XPUTypeOut = typename XPUTypeTrait<T_OUT>::Type;
  auto in_mat_dims = flatten_to_2d(x.dims(), in_num_col_dims);
  int m = in_mat_dims[0];
  int k = in_mat_dims[1];
  int n = w.dims()[0];
  auto* x_data = reinterpret_cast<const XPUTypeX*>(x.data<T_X>());
  const float* x_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  auto* w_data = reinterpret_cast<const XPUTypeW*>(w.data<T_W>());
  auto* w_max_data = w_max.data<float>();
  const float* bias_data =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  auto* out_data =
      reinterpret_cast<XPUTypeOut*>(ctx.template Alloc<T_OUT>(out));
  auto* out_max_data = ctx.template Alloc<float>(out_max);
  xpu::Activation_t act(static_cast<xpu::Activation_t::act_enum>(act_type));
  if (act_type == xpu::Activation_t::LEAKY_RELU) {
    act.leaky_alpha = act_alpha;
  } else if (act_type == xpu::Activation_t::HARD_SIGMOID) {
    act.hard_sigmoid_slope = act_alpha;
  }
  int r =
      xpu::fc_fusion<XPUTypeX, XPUTypeW, XPUTypeOut, T_GEMM>(  // TX/TW/TY/TGEMM
          ctx.x_context(),                                     // ctx
          x_data,                                              // x
          w_data,                                              // w
          out_data,                                            // y
          m,                                                   // m
          n,                                                   // n
          k,                                                   // k
          transpose_x,                                         // x_trans
          true,                                                // w_trans
          x_max_data,                                          // x_maxptr
          w_max_data,                                          // w_maxptr
          out_max_data,                                        // y_maxptr
          transpose_x ? m : k,                                 // ldx
          k,                                                   // ldw
          n,                                                   // ldy
          alpha,                                               // alpha
          beta,                                                // beta
          bias_data,                                           // bias
          act);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_xpu");
}

#define FC_XPU_KERNEL_IMPL(x_dtype_, w_dtype_, out_dtype_, gemm_dtype_) \
  FcXPUKernelImpl<x_dtype_, w_dtype_, out_dtype_, gemm_dtype_>(         \
      ctx,                                                              \
      x,                                                                \
      x_max,                                                            \
      w,                                                                \
      w_max,                                                            \
      bias,                                                             \
      in_num_col_dims,                                                  \
      transpose_x,                                                      \
      alpha,                                                            \
      beta,                                                             \
      act_type,                                                         \
      act_alpha,                                                        \
      out,                                                              \
      out_max);

template <typename T, typename Context>
void FcXPUKernel(const Context& ctx,
                 const DenseTensor& x,
                 const paddle::optional<DenseTensor>& x_max,
                 const DenseTensor& w,
                 const DenseTensor& w_max,
                 const paddle::optional<DenseTensor>& bias,
                 int in_num_col_dims,
                 bool transpose_x,
                 float alpha,
                 float beta,
                 int act_type,
                 float act_alpha,
                 DataType out_dtype,
                 DenseTensor* out,
                 DenseTensor* out_max) {
  if (out_dtype == DataType::FLOAT32) {
    FC_XPU_KERNEL_IMPL(T, int16_t, float, int16_t);
  } else if (out_dtype == DataType::FLOAT16) {
    FC_XPU_KERNEL_IMPL(T, int16_t, dtype::float16, int16_t);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented("Not support out_dtype is %s.",
                                            DataTypeToString(out_dtype)));
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fc_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FcXPUKernel,
                   float,
                   phi::dtype::float16) {}
