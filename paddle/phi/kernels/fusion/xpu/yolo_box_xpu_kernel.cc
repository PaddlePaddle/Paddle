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
                      const DenseTensor& grid,
                      const DenseTensor& stride,
                      const DenseTensor& anchor_grid,
                      const DenseTensor& offset,
                      const paddle::optional<DenseTensor>& x_max,
                      DenseTensor* out,
                      DenseTensor* out_max) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  const float* x_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  auto* y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  const float* y_max_data =
      y_max.get_ptr() == nullptr ? nullptr : y_max.get_ptr()->data<float>();
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));

  std::vector<int64_t> x_shape = phi::vectorize(grid.dims());
  std::vector<int64_t> y_shape = phi::vectorize(strid.dims());
  int r = xpu::yolo_box_coord<XPUType>(
      /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
      /* const T* x */ x_data,
      /* const T* y */ y_data,
      /* const std::vector<int64_t>& x_shape */ x_shape,
      /* const float* grid */ grid_data,
      /* const float* stride */ stride_data,
      /* const float* anchor_grid */ anchor_grid_data,
      /* const std::vector<int64_t>& grid_shape */ grid_shape,
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
