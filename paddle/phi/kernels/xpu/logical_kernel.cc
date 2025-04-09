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

#include "paddle/phi/kernels/logical_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void LogicalNotKernel(const Context& ctx,
                      const DenseTensor& x,
                      DenseTensor* out) {
  ctx.template Alloc<bool>(out);
  int r =
      xpu::logical_not(ctx.x_context(), x.data<T>(), out->data<T>(), x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "logical_not");
}

template <typename T, typename XPUType>
void LogicalBinaryKernel(
    const XPUContext& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    DenseTensor* out,
    std::function<int(
        xpu::Context*, const XPUType*, const XPUType*, bool*, int64_t)> func,
    std::string funcname = "logical") {
  dev_ctx.template Alloc<bool>(out);

  int r = xpu::SUCCESS;
  const auto* x_data = x.data<T>();
  const auto* y_data = y.data<T>();
  auto* out_data = out->data<T>();

  if (x.numel() == out->numel() && y.numel() == out->numel()) {
    r = func(dev_ctx.x_context(),
             reinterpret_cast<const XPUType*>(x_data),
             reinterpret_cast<const XPUType*>(y_data),
             reinterpret_cast<bool*>(out_data),
             out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, funcname);
    return;
  }

  // x or y need to do broadcast
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  int max_dim = std::max(x_dims.size(), y_dims.size());
  int axis = std::abs(x_dims.size() - y_dims.size());

  std::vector<int64_t> x_dims_vec(max_dim, 1);
  std::vector<int64_t> y_dims_vec(max_dim, 1);
  if (x_dims.size() == max_dim) {
    for (int i = 0; i < max_dim; i++) {
      x_dims_vec[i] = x_dims[i];
    }
  } else {
    for (int i = 0; i < x_dims.size(); i++) {
      x_dims_vec[i + axis] = x_dims[i];
    }
  }
  if (y_dims.size() == max_dim) {
    for (int i = 0; i < max_dim; i++) {
      y_dims_vec[i] = y_dims[i];
    }
  } else {
    for (int i = 0; i < y_dims.size(); i++) {
      y_dims_vec[i + axis] = y_dims[i];
    }
  }
  if (x_dims_vec.size() == 0) {
    x_dims_vec = std::vector<int64_t>({1});
  }

  if (y_dims_vec.size() == 0) {
    y_dims_vec = std::vector<int64_t>({1});
  }

  bool is_x_need_broadcast = false;
  bool is_y_need_broadcast = false;
  auto out_vec = common::vectorize(out->dims());
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_vec[i] != out_vec[i]) {
      is_x_need_broadcast = true;
      break;
    }
  }
  for (int i = 0; i < max_dim; i++) {
    if (y_dims_vec[i] != out_vec[i]) {
      is_y_need_broadcast = true;
      break;
    }
  }

  auto xpu_context = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_context);
  if (is_x_need_broadcast) {
    T* x_data_broadcast = RAII_GUARD.alloc_l3_or_gm<T>(out->numel());
    r = xpu::broadcast<XPUType>(xpu_context,
                                reinterpret_cast<const XPUType*>(x_data),
                                reinterpret_cast<XPUType*>(x_data_broadcast),
                                x_dims_vec,
                                out_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
    x_data = x_data_broadcast;
  }
  if (is_y_need_broadcast) {
    T* y_data_broadcast = RAII_GUARD.alloc_l3_or_gm<T>(out->numel());
    r = xpu::broadcast<XPUType>(xpu_context,
                                reinterpret_cast<const XPUType*>(y_data),
                                reinterpret_cast<XPUType*>(y_data_broadcast),
                                y_dims_vec,
                                out_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
    y_data = y_data_broadcast;
  }

  r = func(xpu_context,
           reinterpret_cast<const XPUType*>(x_data),
           reinterpret_cast<const XPUType*>(y_data),
           reinterpret_cast<bool*>(out_data),
           out->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, funcname);
}

template <typename T, typename Context>
void LogicalAndKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  LogicalBinaryKernel<T, XPUType>(
      dev_ctx, x, y, out, xpu::logical_and<XPUType, XPUType>, "logical_and");
}

template <typename T, typename Context>
void LogicalOrKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  LogicalBinaryKernel<T, XPUType>(
      dev_ctx, x, y, out, xpu::logical_or<XPUType, XPUType>, "logical_or");
}

template <typename T, typename Context>
void LogicalXorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  LogicalBinaryKernel<T, XPUType>(
      dev_ctx, x, y, out, xpu::logical_xor<XPUType, XPUType>, "logical_xor");
}
}  // namespace phi

PD_REGISTER_KERNEL(logical_not, XPU, ALL_LAYOUT, phi::LogicalNotKernel, bool) {}
PD_REGISTER_KERNEL(logical_and, XPU, ALL_LAYOUT, phi::LogicalAndKernel, bool) {}
PD_REGISTER_KERNEL(logical_or, XPU, ALL_LAYOUT, phi::LogicalOrKernel, bool) {}
PD_REGISTER_KERNEL(logical_xor, XPU, ALL_LAYOUT, phi::LogicalXorKernel, bool) {}
