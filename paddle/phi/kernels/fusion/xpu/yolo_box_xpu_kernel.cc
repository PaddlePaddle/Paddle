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
void YoloBoxXPUKernel(const Context& ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& x_max,
                      const DenseTensor& grid,
                      const DenseTensor& stride,
                      const DenseTensor& anchor_grid,
                      float offset,
                      DenseTensor* out,
                      DenseTensor* out_max) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  // float* x_max
  float* x_max_data = nullptr;
  const float* grid_data;
  const float* stride_data;
  const float* anchor_grid_data;
  // fix precision of fp16 model
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  if (std::is_same<T, phi::dtype::float16>::value) {
    float* grid_data_temp = RAII_GUARD.alloc_l3_or_gm<float>(grid.numel());
    int r = xpu::cast<XPUType, float>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(grid.data<T>()),
        grid_data_temp,
        grid.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    float* stride_data_temp = RAII_GUARD.alloc_l3_or_gm<float>(stride.numel());
    r = xpu::cast<XPUType, float>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(stride.data<T>()),
        stride_data_temp,
        stride.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    float* anchor_grid_data_temp =
        RAII_GUARD.alloc_l3_or_gm<float>(anchor_grid.numel());
    r = xpu::cast<XPUType, float>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(anchor_grid.data<T>()),
        anchor_grid_data_temp,
        anchor_grid.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    grid_data = grid_data_temp;
    stride_data = stride_data_temp;
    anchor_grid_data = anchor_grid_data_temp;
  } else {
    grid_data = grid.data<float>();
    stride_data = stride.data<float>();
    anchor_grid_data = anchor_grid.data<float>();
  }
  std::vector<int64_t> x_shape = common::vectorize(x.dims());
  std::vector<int64_t> grid_shape = common::vectorize(grid.dims());
  std::vector<int64_t> stride_shape = common::vectorize(stride.dims());
  std::vector<int64_t> anchor_grid_shape =
      common::vectorize(anchor_grid.dims());
  // yolo_box_coord only support fp32&&fp16 precision
  int r = xpu::yolo_box_coord<XPUType>(
      /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
      /* const T* x */ x_data,
      /* T* y */ out_data,
      /* const std::vector<int64_t>& x_shape */ x_shape,
      /* const float* grid */ grid_data,
      /* const float* stride */ stride_data,
      /* const float* anchor_grid */ anchor_grid_data,
      /* const std::vector<int64_t>& grid_shape */ grid_shape,
      /* const std::vector<int64_t>& stride_shape */ stride_shape,
      /* const std::vector<int64_t>& anchor_grid */ anchor_grid_shape,
      /* float offset */ offset,
      /* float* x_max */ x_max_data,
      /* float* y_max */ ctx.template Alloc<float>(out_max));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "yolo_box_xpu");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(yolo_box_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::YoloBoxXPUKernel,
                   float,
                   phi::dtype::float16) {}
