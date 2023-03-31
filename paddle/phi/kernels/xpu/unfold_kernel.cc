// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/unfold_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/unfold_functor.h"

namespace phi {

template <typename T, typename Context>
void UnfoldKernel(const Context& ctx,
                  const DenseTensor& x,
                  const std::vector<int>& kernel_sizes,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(out);
  const std::string data_format = phi::DataLayoutToString(x.layout());
  bool is_nchw = data_format == "NCHW";
  PADDLE_ENFORCE_EQ(is_nchw,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Unfold op only supports datalayout == NCHW"));
  auto x_dims = x.dims();
  int n = static_cast<int>(x_dims[0]);
  int c = static_cast<int>(x_dims[1]);
  int h = static_cast<int>(x_dims[2]);
  int w = static_cast<int>(x_dims[3]);

  int out_height = phi::funcs::CalcOutputSize(x_dims[2],
                                              kernel_sizes[0],
                                              dilations[0],
                                              paddings[0],
                                              paddings[2],
                                              strides[0]);
  int out_width = phi::funcs::CalcOutputSize(x_dims[3],
                                             kernel_sizes[1],
                                             dilations[1],
                                             paddings[1],
                                             paddings[3],
                                             strides[1]);

  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  XPUType* out_pre_trans = RAII_GUARD.alloc_l3_or_gm<XPUType>(out->numel());
  int r = xpu::im2col(ctx.x_context(),
                      reinterpret_cast<const XPUType*>(x.data<T>()),
                      out_pre_trans,
                      n,
                      c,
                      h,
                      w,
                      kernel_sizes,
                      strides,
                      paddings,
                      dilations,
                      is_nchw);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "im2col");

  r = xpu::transpose(
      ctx.x_context(),
      out_pre_trans,
      reinterpret_cast<XPUType*>(out->data<T>()),
      {n, out_height, out_width, c, kernel_sizes[0], kernel_sizes[1]},
      {0, 3, 4, 5, 1, 2});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
}
}  // namespace phi

PD_REGISTER_KERNEL(
    unfold, XPU, ALL_LAYOUT, phi::UnfoldKernel, float, phi::dtype::float16) {}
