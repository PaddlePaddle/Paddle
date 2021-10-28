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
namespace platform {

template <typename XPUType, typename FCT>
int xpu_fc_wrapper(xpu::Context* ctx, const XPUType* x, const XPUType* w,
                   XPUType* y, int m, int n, int k, bool x_trans, bool w_trans,
                   const float* x_maxptr, const float* w_maxptr,
                   float* y_maxptr, int ldx, int ldw, int ldy, float alpha,
                   float beta, const float* bias,
                   const xpu::Activation_t& act) {
  int r = 0;
  if (x_trans && std::getenv("XPU_PADDLE_FC_TRANS_A") != nullptr &&
      std::is_same<float, XPUType>::value) {
    XPUType* l3_addr = nullptr;
    uint32_t l3_size = 63 * 1024 * 1024;  // default 63M L3
    if (std::getenv("XPU_PADDLE_L3_SIZE") != nullptr) {
      l3_size = atoi(std::getenv("XPU_PADDLE_L3_SIZE"));
    }
    if (m * k * sizeof(XPUType) <= l3_size) {
      r = xpu_malloc(reinterpret_cast<void**>(&l3_addr),
                     m * k * sizeof(XPUType), XPU_MEM_L3);
    }
    if (r != XPU_SUCCESS || m * k * sizeof(XPUType) > l3_size) {
      r = xpu_malloc(reinterpret_cast<void**>(&l3_addr),
                     m * k * sizeof(XPUType));
    }
    if (r != XPU_SUCCESS) {
      return r;
    }

    std::vector<int> shape = {k, m};
    std::vector<int> axis = {1, 0};
    r = xpu::transpose<XPUType>(ctx, x, l3_addr, shape, axis);
    if (r != XPU_SUCCESS) {
      xpu_free(l3_addr);
      return r;
    }
    r = xpu::fc_fusion<XPUType, XPUType, XPUType, FCT>(
        ctx, l3_addr, w, y, m, n, k, false, w_trans, x_maxptr, w_maxptr,
        y_maxptr, k, ldw, ldy, alpha, beta, bias, act);
    if (r != XPU_SUCCESS) {
      xpu_free(l3_addr);
      return r;
    }
    r = xpu_wait(ctx->xpu_stream);
    if (r != XPU_SUCCESS) {
      xpu_free(l3_addr);
      return r;
    }
    xpu_free(l3_addr);
  } else {
    r = xpu::fc_fusion<XPUType, XPUType, XPUType, FCT>(
        ctx, x, w, y, m, n, k, x_trans, w_trans, x_maxptr, w_maxptr, y_maxptr,
        ldx, ldw, ldy, alpha, beta, bias, act);
  }
  return r;
}

}  // namespace platform
}  // namespace paddle
#endif
