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
void AffineChannelXPUKernel(const Context& dev_ctx,
                            const DenseTensor& x_in,
                            const DenseTensor& scale_in,
                            const DenseTensor& bias_in,
                            const std::string& data_layout,
                            DenseTensor* out) {
  auto* x = &x_in;
  auto* scale = &scale_in;
  auto* bias = &bias_in;

  auto* y = out;
  dev_ctx.template Alloc<T>(y);

  const phi::DataLayout layout = common::StringToDataLayout(data_layout);

  auto dims = x->dims();
  int N = dims[0];
  int C = layout == phi::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int HxW = x->numel() / N / C;

  auto* scale_d = scale->data<T>();
  auto* bias_d = bias->data<T>();

  auto* x_d = x->data<T>();
  auto* y_d = y->data<T>();
  std::vector<int> x_shape;
  std::vector<int> b_shape;
  if (layout == phi::DataLayout::kNCHW) {
    x_shape.push_back(N);
    x_shape.push_back(C);
    x_shape.push_back(HxW);
    b_shape.push_back(1);
    b_shape.push_back(C);
    b_shape.push_back(1);
  } else {
    x_shape.push_back(N * HxW);
    x_shape.push_back(C);
    b_shape.push_back(1);
    b_shape.push_back(C);
  }
  int r = 0;
  r = xpu::broadcast_mul(
      dev_ctx.x_context(), x_d, scale_d, y_d, x_shape, b_shape);
  PADDLE_ENFORCE_EQ(r,
                    xpu::Error_t::SUCCESS,
                    common::errors::External(
                        "The broadcast_mul XPU OP return wrong value[%d %s]",
                        r,
                        XPUAPIErrorMsg[r]));
  r = xpu::broadcast_add(
      dev_ctx.x_context(), y_d, bias_d, y_d, x_shape, b_shape);
  PADDLE_ENFORCE_EQ(r,
                    xpu::Error_t::SUCCESS,
                    common::errors::External(
                        "The broadcast_add XPU OP return wrong value[%d %s]",
                        r,
                        XPUAPIErrorMsg[r]));
}

}  // namespace phi

PD_REGISTER_KERNEL(
    affine_channel, XPU, ALL_LAYOUT, phi::AffineChannelXPUKernel, float) {}
