/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#ifdef PADDLE_WITH_XPU
#include <vector>

namespace paddle {
namespace operators {

template <typename XPUType, typename FCT>
int xpu_fc_wrapper(xpu::Context* ctx,
                   const XPUType* x,
                   const XPUType* w,
                   XPUType* y,
                   int m,
                   int n,
                   int k,
                   bool x_trans,
                   bool w_trans,
                   const float* x_maxptr,
                   const float* w_maxptr,
                   float* y_maxptr,
                   int ldx,
                   int ldw,
                   int ldy,
                   float alpha,
                   float beta,
                   const float* bias,
                   const xpu::Activation_t& act) {
  int r = 0;
  if (x_trans && std::getenv("XPU_PADDLE_FC_TRANS_A") != nullptr &&
      std::is_same<float, XPUType>::value) {
    XPUType* l3_addr = nullptr;
    xpu::ctx_guard RAII_GUARD(ctx);
    l3_addr = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * k);
    if (l3_addr == nullptr) return XPUERR_NOMEM;

    std::vector<int> shape = {k, m};
    std::vector<int> axis = {1, 0};
    r = xpu::transpose<XPUType>(ctx, x, l3_addr, shape, axis);
    if (r != XPU_SUCCESS) return r;

    r = xpu::fc_fusion<XPUType, XPUType, XPUType, FCT>(ctx,
                                                       l3_addr,
                                                       w,
                                                       y,
                                                       m,
                                                       n,
                                                       k,
                                                       false,
                                                       w_trans,
                                                       x_maxptr,
                                                       w_maxptr,
                                                       y_maxptr,
                                                       k,
                                                       ldw,
                                                       ldy,
                                                       alpha,
                                                       beta,
                                                       bias,
                                                       act);
    if (r != XPU_SUCCESS) return r;
  } else {
    r = xpu::fc_fusion<XPUType, XPUType, XPUType, FCT>(ctx,
                                                       x,
                                                       w,
                                                       y,
                                                       m,
                                                       n,
                                                       k,
                                                       x_trans,
                                                       w_trans,
                                                       x_maxptr,
                                                       w_maxptr,
                                                       y_maxptr,
                                                       ldx,
                                                       ldw,
                                                       ldy,
                                                       alpha,
                                                       beta,
                                                       bias,
                                                       act);
  }
  return r;
}

}  // namespace operators
}  // namespace paddle
#endif
