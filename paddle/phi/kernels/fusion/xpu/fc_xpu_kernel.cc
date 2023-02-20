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
                 DenseTensor* out,
                 DenseTensor* out_max) {
  auto in_mat_dims = flatten_to_2d(x.dims(), in_num_col_dims);
  int m = in_mat_dims[0];
  int k = in_mat_dims[1];
  int n = w.dims()[0];
  const float* x_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  const float* bias_data =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  xpu::Activation_t act(static_cast<xpu::Activation_t::act_enum>(act_type));
  if (act_type == 5) {
    act.leaky_alpha = act_alpha;
  } else if (act_type == 15) {
    act.hard_sigmoid_slope = act_alpha;
  }
  int r = xpu::fc_fusion<T, int16_t, T, int16_t>(  // TX, TW. TY, TGEMM
      ctx.x_context(),                             // ctx
      x.data<T>(),                                 // x
      w.data<int16_t>(),                           // w
      ctx.template Alloc<T>(out),                  // y
      m,                                           // m
      n,                                           // n
      k,                                           // k
      transpose_x,                                 // x_trans
      true,                                        // w_trans
      x_max_data,                                  // x_maxptr
      w_max.data<float>(),                         // w_maxptr
      ctx.template Alloc<float>(out_max),          // y_maxptr
      transpose_x ? m : k,                         // ldx
      k,                                           // ldw
      n,                                           // ldy
      alpha,                                       // alpha
      beta,                                        // beta
      bias_data,                                   // bias
      act);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_xpu");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fc_xpu, XPU, ALL_LAYOUT, phi::fusion::FcXPUKernel, float) {}
