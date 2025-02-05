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

#include "paddle/phi/kernels/grid_sample_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GridSampleGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& grid,
                          const DenseTensor& out_grid,
                          const std::string& mode,
                          const std::string& padding_mode,
                          bool align_corners,
                          DenseTensor* x_grad,
                          DenseTensor* grid_grad) {
  PADDLE_ENFORCE_EQ(
      x.dims().size(),
      4,
      common::errors::InvalidArgument(
          ("XPU is only support input_dims == 4 in grid_sample_grad op.")));

  const int64_t n = grid.dims()[0];
  const int64_t out_h = grid.dims()[1];
  const int64_t out_w = grid.dims()[2];
  const int64_t c = x.dims()[1];
  const int64_t in_h = x.dims()[2];
  const int64_t in_w = x.dims()[3];

  x_grad->Resize({n, c, in_h, in_w});
  T* x_grad_ptr = dev_ctx.template Alloc<T>(x_grad);

  T* grid_grad_ptr = nullptr;
  if (grid_grad != nullptr) {
    grid_grad->Resize({n, out_h, out_w, 2});
    grid_grad_ptr = dev_ctx.template Alloc<T>(grid_grad);
  }

  bool is_nearest = false;
  if (mode == "nearest") {
    is_nearest = true;
  }
  int64_t padding_mode_type = 0;
  if (padding_mode == "border") {
    padding_mode_type = 1;
  } else if (padding_mode == "reflection") {
    padding_mode_type = 2;
  }

  int r = xpu::grid_sample_grad<T>(dev_ctx.x_context(),
                                   x.data<T>(),
                                   grid.data<T>(),
                                   out_grid.data<T>(),
                                   x_grad_ptr,
                                   grid_grad_ptr,
                                   n,
                                   c,
                                   in_h,
                                   in_w,
                                   out_h,
                                   out_w,
                                   is_nearest,
                                   align_corners,
                                   padding_mode_type,
                                   true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "grid_sample_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    grid_sample_grad, XPU, ALL_LAYOUT, phi::GridSampleGradKernel, float) {}
