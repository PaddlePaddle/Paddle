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
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedGemmEpilogueKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& bias,
                             const bool trans_x,
                             const bool trans_y,
                             const std::string& activation,
                             DenseTensor* out,
                             DenseTensor* reserve_space) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::Context* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);

  int r = xpu::SUCCESS;
  xpu::Activation_t act = xpu::Activation_t::LINEAR;
  if (activation == "relu") {
    act = xpu::Activation_t::RELU;
  } else if (activation == "gelu") {
    act = xpu::Activation_t::GELU;
  }

  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y.data<T>());
  XPUType* out_ptr = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out));

  decltype(&xpu_fc_wrapper<XPUType, int16_t>) fc_api_list[5] = {
      &xpu_fc_wrapper<XPUType, int16_t>,
      &xpu_fc_wrapper<XPUType, int32_t>,
      &xpu_fc_wrapper<XPUType, float>,
      &xpu_fc_wrapper<XPUType, int_with_ll_t>,
      &xpu_fc_wrapper<XPUType, tfloat32>,
  };

  auto fccal_type = FCCalcType<XPUType>();

  auto fc_api = fc_api_list[fccal_type];

  // fc + bias + act
  phi::XpuFcInfo fc_info;
  auto mat_x_dims =
      common::flatten_to_2d(x.dims(), trans_x ? 1 : x.dims().size() - 1);
  auto mat_y_dims = y.dims();
  phi::GetFCInfo(mat_x_dims, mat_y_dims, trans_x, trans_y, &fc_info);
  int batch_size = fc_info.bs;
  int m = fc_info.m;
  int n = fc_info.n;
  int k = fc_info.k;
  int ldx = fc_info.stride_x;
  int ldy = fc_info.stride_y;
  int ldout = fc_info.stride_out;
  float* max_x = fc_info.max_x;
  float* max_y = fc_info.max_y;
  float* max_out = fc_info.max_out;

  PADDLE_ENFORCE_LE(
      batch_size,
      1,
      errors::InvalidArgument(
          "FusedGemm do not support batched fc now, but got batch size %d.",
          batch_size));
  const float* bias_ptr = reinterpret_cast<const float*>(bias.data<T>());
  if (!std::is_same<T, float>::value) {
    // TODO(lijin23): Now xblas and xdnn support fp32 bias only, may be removed
    // in the future.
    float* bias_tmp = RAII_GUARD.alloc_l3_or_gm<float>(bias.numel());
    r = xpu::cast<XPUType, float>(
        xpu_ctx,
        reinterpret_cast<const XPUType*>(bias.data<T>()),
        bias_tmp,
        bias.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    bias_ptr = bias_tmp;
  }
  fc_api(xpu_ctx,
         x_ptr,
         y_ptr,
         out_ptr,
         m,
         n,
         k,
         trans_x,
         trans_y,
         max_x,
         max_y,
         max_out,
         ldx,
         ldy,
         ldout,
         1.0f,
         0,
         bias_ptr,
         act);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_gemm_epilogue,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGemmEpilogueKernel,
                   float,
                   phi::dtype::float16) {}
