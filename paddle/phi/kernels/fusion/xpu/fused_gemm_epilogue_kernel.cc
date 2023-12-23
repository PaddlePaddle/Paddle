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
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/scope_guard.h"
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

  auto x_mat_dims =
      common::flatten_to_2d(x.dims(), trans_x ? 1 : x.dims().size() - 1);

  // (M * K) * (K * N) for new api use
  // int64_t M = trans_x ? x_mat_dims[1] : x_mat_dims[0];
  // int64_t K = trans_y ? y->dims()[1] : y->dims()[0];
  // int64_t N = trans_y ? y->dims()[0] : y->dims()[1];

  // 调用新接口，这里先分开调用，等待qingpen的新接口
  int r = 0;
  xpu::Activation_t act = xpu::Activation_t::LINEAR;
  if (activation == "relu") {
    act = xpu::Activation_t::RELU;
  } else if (activation == "gelu") {
    act = xpu::Activation_t::GELU;
  }
  // fc + bias + act
  // 1. fc
  phi::XpuFcInfo fc_info;

  phi::GetFCInfo(x_mat_dims, y.dims(), trans_x, trans_y, &fc_info);
  xpu::Context* xpu_ctx = dev_ctx.x_context();

  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y.data<T>());
  auto* out_tmp_ptr = dev_ctx.template Alloc<T>(out);
  XPUType* out_ptr = reinterpret_cast<XPUType*>(out_tmp_ptr);
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  XPUType* fc_out_ptr = RAII_GUARD.alloc_l3_or_gm<XPUType>(out->numel());
  phi::MatMulXPUFunction<XPUType>(
      xpu_ctx, x_ptr, y_ptr, fc_out_ptr, fc_info, 1.0f);
  XPUType* bias_out_ptr = out_ptr;
  if (activation != "none" && reserve_space) {
    auto* bias_out_temp_ptr = dev_ctx.template Alloc<T>(reserve_space);
    bias_out_ptr = reinterpret_cast<XPUType*>(bias_out_temp_ptr);
  }
  // 2 bias
  const XPUType* bias_ptr = reinterpret_cast<const XPUType*>(bias.data<T>());
  r = xpu::broadcast_add(xpu_ctx,
                         fc_out_ptr,
                         bias_ptr,
                         bias_out_ptr,
                         {fc_info.m, fc_info.n},
                         {fc_info.n});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
  // 3 act
  if (activation == "relu") {
    r = xpu::relu(xpu_ctx, bias_out_ptr, out_ptr, out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu");
  } else if (activation == "gelu") {
    r = xpu::gelu(xpu_ctx, bias_out_ptr, out_ptr, out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "gelu");
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_gemm_epilogue,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedGemmEpilogueKernel,
                   float,
                   phi::dtype::float16) {}
