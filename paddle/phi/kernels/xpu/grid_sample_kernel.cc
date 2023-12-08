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

#include "paddle/phi/kernels/grid_sample_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GridSampleKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& grid,
                      const std::string& mode,
                      const std::string& padding_mode,
                      bool align_corners,
                      DenseTensor* out) {
  // attrs
  // paddle.nn.functional.grid_sample(x, grid, mode='bilinear',
  // padding_mode='zeros', align_corners=True, name=None)
  const std::string data_format = common::DataLayoutToString(x.layout());

  // attr to real param
  bool is_nearest_bool;
  if (mode == "bilinear") {
    is_nearest_bool = false;
  } else if (mode == "nearest") {
    is_nearest_bool = true;
  } else {
    PADDLE_THROW(errors::InvalidArgument(
        "should not reach here: mode should be either 'bilinear' or "
        "'nearest', bot got %s.",
        mode));
  }

  // attention: 0: zeros, 2: reflection, 1: border according to XDNN api.
  int padding_mode_int;
  if (padding_mode == "zeros") {
    padding_mode_int = 0;
  } else if (padding_mode == "reflection") {
    padding_mode_int = 2;
  } else if (padding_mode == "border") {
    padding_mode_int = 1;
  } else {
    PADDLE_THROW(errors::InvalidArgument(
        "should not reach here: padding_mode should be either 'zeros' or "
        "'reflection' or 'border', bot got %s.",
        padding_mode));
  }

  const T* input_data = x.data<T>();
  const T* grid_data = grid.data<T>();

  int n = x.dims()[0];
  int c = x.dims()[1];

  if (x.dims().size() == 4) {  // 2D grid sample
    int h = x.dims()[2];
    int w = x.dims()[3];
    int out_h = grid.dims()[1];
    int out_w = grid.dims()[2];

    bool is_nchw_bool;
    if (data_format == "NCHW") {
      is_nchw_bool = true;
    } else if (data_format == "NHWC") {
      is_nchw_bool = false;
    } else {
      PADDLE_THROW(errors::InvalidArgument(
          "should not reach here: data_format should be either 'NCHW' or "
          "'NHWC', bot got %s.",
          data_format));
    }

    out->Resize(common::make_ddim({n, c, out_h, out_w}));
    T* output_data = dev_ctx.template Alloc<T>(out);

    int r = xpu::grid_sample(dev_ctx.x_context(),
                             input_data,
                             grid_data,
                             output_data,
                             n,
                             c,
                             h,
                             w,
                             out_h,
                             out_w,
                             is_nearest_bool,
                             align_corners,
                             padding_mode_int,
                             is_nchw_bool);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "grid_sampler");
  } else {  // 3D grid sample
    int d = x.dims()[2];
    int h = x.dims()[3];
    int w = x.dims()[4];
    int out_d = grid.dims()[1];
    int out_h = grid.dims()[2];
    int out_w = grid.dims()[3];

    out->Resize(common::make_ddim({n, c, out_d, out_h, out_w}));
    T* output_data = dev_ctx.template Alloc<T>(out);

    int r = xpu::grid_sample3d(dev_ctx.x_context(),
                               input_data,
                               grid_data,
                               output_data,
                               n,
                               c,
                               d,
                               h,
                               w,
                               out_d,
                               out_h,
                               out_w,
                               is_nearest_bool,
                               align_corners,
                               padding_mode_int,
                               true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "grid_sampler3d");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(grid_sample, XPU, ALL_LAYOUT, phi::GridSampleKernel, float) {
}
