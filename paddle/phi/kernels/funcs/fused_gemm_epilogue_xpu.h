// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/scope_guard.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

#if defined(PADDLE_WITH_XPU)
namespace phi {
namespace funcs {

template <typename T>
void ComputeFusedGemmEpilogueBackwardXPU(const phi::XPUContext& dev_ctx,
                                         const phi::DenseTensor* dout,
                                         const phi::DenseTensor* x,
                                         const phi::DenseTensor* y,
                                         const phi::DenseTensor* reserve_space,
                                         int64_t M,
                                         int64_t N,
                                         int64_t K,
                                         bool trans_x,
                                         bool trans_y,
                                         const std::string& activation_grad,
                                         phi::DenseTensor* dx,
                                         phi::DenseTensor* dy,
                                         phi::DenseTensor* dbias,
                                         bool use_addto_dx = false,
                                         bool use_addto_dy = false) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  const XPUType* dout_ptr = reinterpret_cast<const XPUType*>(dout->data<T>());
  const XPUType* dout_fc_ptr = dout_ptr;
  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
  const XPUType* y_ptr =
      (y == nullptr) ? nullptr : reinterpret_cast<const XPUType*>(y->data<T>());

  const XPUType* reserve_space_ptr =
      (reserve_space == NULL)
          ? (reinterpret_cast<const XPUType*>(NULL))
          : (reinterpret_cast<const XPUType*>(reserve_space->data<T>()));
  XPUType* d_act_input_ptr = NULL;
  if (activation_grad != "none") {
    d_act_input_ptr = RAII_GUARD.alloc_l3_or_gm<XPUType>(dout->numel());
    dout_fc_ptr = d_act_input_ptr;
  }

  // 1. act_grad  2. fc_grad 3. dbias
  int r = 0;
  if (activation_grad == "relu") {
    r = xpu::relu_grad(xpu_ctx,
                       reserve_space_ptr,
                       reserve_space_ptr,
                       dout_ptr,
                       d_act_input_ptr,
                       dout->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu_grad");
  } else if (activation_grad == "gelu") {
    r = xpu::gelu_grad(xpu_ctx,
                       reserve_space_ptr,
                       reserve_space_ptr,
                       dout_ptr,
                       d_act_input_ptr,
                       dout->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "gelu_grad");
  } else if (activation_grad == "none") {
    // pass
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "activation_grad of fused_gemm_epilogue_grad should be one of {none, "
        "relu, gelu}, but received %s",
        activation_grad));
  }
  auto x_mat_dims =
      common::flatten_to_2d(x->dims(), trans_x ? 1 : x->dims().size() - 1);
  phi::XpuFcInfo info_forward;
  phi::GetFCInfo(
      x_mat_dims, common::make_ddim({K, N}), trans_x, trans_y, &info_forward);

  // 2. fc_grad
  const XPUType* a_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* a_2 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_2 = reinterpret_cast<const XPUType*>(NULL);
  XPUType* c_1;
  if (dx == NULL) {
    c_1 = reinterpret_cast<XPUType*>(NULL);
  } else {
    auto* x_grad_tmp = dev_ctx.template Alloc<T>(dx);
    c_1 = reinterpret_cast<XPUType*>(x_grad_tmp);
  }
  XPUType* c_2;
  if (dy == NULL) {
    c_2 = reinterpret_cast<XPUType*>(NULL);
  } else {
    auto* y_grad_tmp = dev_ctx.template Alloc<T>(dy);
    c_2 = reinterpret_cast<XPUType*>(y_grad_tmp);
  }
  phi::XpuFcInfo info_dx;
  phi::XpuFcInfo info_dy;
  std::tuple<phi::XpuFcInfo,
             phi::XpuFcInfo,
             const XPUType*,
             const XPUType*,
             const XPUType*,
             const XPUType*>
      fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                      &RAII_GUARD,
                                      info_forward,
                                      trans_x,
                                      trans_y,
                                      x_ptr,
                                      y_ptr,
                                      dout_fc_ptr);
  std::tie(info_dx, info_dy, a_1, b_1, a_2, b_2) = fc_info;
  if (dx) {
    float beta = use_addto_dx ? 1.0f : 0.0f;
    phi::MatMulXPUFunction<XPUType>(
        xpu_ctx, a_1, b_1, c_1, info_dx, 1.0f, beta);
  }
  if (dy) {
    float beta = use_addto_dy ? 1.0f : 0.0f;
    phi::MatMulXPUFunction<XPUType>(
        xpu_ctx, a_2, b_2, c_2, info_dy, 1.0f, beta);
  }
  // 3. dbias
  if (dbias) {
    XPUType* dbias_ptr;
    auto* dbias_tmp_ptr = dev_ctx.template Alloc<T>(dbias);
    dbias_ptr = reinterpret_cast<XPUType*>(dbias_tmp_ptr);
    r = xpu::reduce_sum(
        xpu_ctx, dout_fc_ptr, dbias_ptr, {info_forward.m, info_forward.n}, {0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
  }
}

}  // namespace funcs
}  // namespace phi
#endif
