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

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void AffineChannelGradXPUKernel(const Context& dev_ctx,
                                const DenseTensor& x_in,
                                const DenseTensor& scale_in,
                                const DenseTensor& bias_in,
                                const DenseTensor& out_grad,
                                const std::string& data_layout,
                                DenseTensor* x_grad,
                                DenseTensor* scale_grad,
                                DenseTensor* bias_grad) {
  auto* x = &x_in;
  auto* scale = &scale_in;
  auto* dy = &out_grad;

  auto* dx = x_grad;
  auto* dscale = scale_grad;
  auto* dbias = bias_grad;

  const phi::DataLayout layout = common::StringToDataLayout(data_layout);

  auto dims = x->dims();
  int N = dims[0];
  int C = layout == phi::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int HxW = x->numel() / N / C;

  auto* dy_d = dy->data<T>();
  auto* scale_d = scale->data<T>();

  T* dx_d = dx ? dev_ctx.template Alloc<T>(dx) : nullptr;
  T* dscale_d = dscale ? dev_ctx.template Alloc<T>(dscale) : nullptr;
  T* dbias_d = dbias ? dev_ctx.template Alloc<T>(dbias) : nullptr;

  std::vector<int> x_shape;
  std::vector<int> b_shape;
  std::vector<int> rdims;
  if (layout == phi::DataLayout::kNCHW) {
    x_shape.push_back(N);
    x_shape.push_back(C);
    x_shape.push_back(HxW);
    b_shape.push_back(1);
    b_shape.push_back(C);
    b_shape.push_back(1);
    rdims.push_back(0);
    rdims.push_back(2);
  } else {
    x_shape.push_back(N * HxW);
    x_shape.push_back(C);
    b_shape.push_back(1);
    b_shape.push_back(C);
    rdims.push_back(0);
  }

  int r = 0;
  if (dscale_d && dbias_d) {
    r = xpu::reduce_sum<T>(dev_ctx.x_context(), dy_d, dbias_d, x_shape, rdims);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      common::errors::External(
                          "The reduce_sum XPU OP return wrong value[%d %s]",
                          r,
                          XPUAPIErrorMsg[r]));
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    T* tmp = RAII_GUARD.alloc_l3_or_gm<T>(dy->numel());
    PADDLE_ENFORCE_NOT_NULL(
        tmp, common::errors::External("XPU has no enough memory"));

    r = xpu::mul<T>(dev_ctx.x_context(), dy_d, x->data<T>(), tmp, dy->numel());
    PADDLE_ENFORCE_EQ(
        r,
        xpu::Error_t::SUCCESS,
        common::errors::External(
            "The mul XPU OP return wrong value[%d %s]", r, XPUAPIErrorMsg[r]));
    r = xpu::reduce_sum<T>(dev_ctx.x_context(), tmp, dscale_d, x_shape, rdims);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      common::errors::External(
                          "The reduce_sum XPU OP return wrong value[%d %s]",
                          r,
                          XPUAPIErrorMsg[r]));
  }
  if (dx_d) {
    r = xpu::broadcast_mul(
        dev_ctx.x_context(), dy_d, scale_d, dx_d, x_shape, b_shape);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      common::errors::External(
                          "The broadcast_mul XPU OP return wrong value[%d %s]",
                          r,
                          XPUAPIErrorMsg[r]));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(affine_channel_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::AffineChannelGradXPUKernel,
                   float) {}
