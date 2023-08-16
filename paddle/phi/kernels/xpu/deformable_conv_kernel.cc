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

#include "paddle/phi/kernels/deformable_conv_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DeformableConvKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& offset,
                          const DenseTensor& filter,
                          const paddle::optional<DenseTensor>& mask,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          const std::vector<int>& dilations,
                          int deformable_groups,
                          int groups,
                          int im2col_step,
                          DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (phi::backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId()) ==
      phi::backends::xpu::XPUVersion::XPU1) {
    PADDLE_ENFORCE_EQ(
        deformable_groups == 1,
        true,
        errors::InvalidArgument(("XPU1 only support deformable_groups == 1 in "
                                 "deformable_conv op.")));
    PADDLE_ENFORCE_EQ(
        groups == 1,
        true,
        errors::InvalidArgument(
            ("XPU1 only support groups == 1 in deformable_conv op.")));
  }
  PADDLE_ENFORCE_EQ(filter.dims()[2] <= 8 && filter.dims()[3] <= 8,
                    true,
                    errors::InvalidArgument(
                        "Filter high and weight should less than 8 on xpu "
                        "in deformable_conv op."));

  const int batch_size = static_cast<int>(x.dims()[0]);
  std::vector<int64_t> output_shape_vec(phi::vectorize(out->dims()));

  const T* input_ptr = x.data<T>();
  const T* filter_ptr = filter.data<T>();
  const float* offset_ptr = offset.data<T>();
  const float* mask_ptr = mask->data<T>();
  T* output_prt = out->data<T>();

  // set zeros for d_table_data
  const int zero = 0;
  int r = xpu::constant<T>(dev_ctx.x_context(), output_prt, out->numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  int input_dim = x.numel() / x.dims()[0];
  int input_offset_dim = offset.numel() / offset.dims()[0];
  int input_mask_dim = mask->numel() / mask->dims()[0];
  int output_dim =
      output_shape_vec[1] * output_shape_vec[2] * output_shape_vec[3];
  std::vector<int> ksize{static_cast<int>(filter.dims()[2]),
                         static_cast<int>(filter.dims()[3])};
  int n = im2col_step;
  int c = x.dims()[1];
  int h = x.dims()[2];
  int w = x.dims()[3];
  int f = filter.dims()[0];

  for (int i = 0; i < batch_size / im2col_step; ++i) {
    int r = xpu::deformable_conv<float, float, float, int>(
        dev_ctx.x_context(),
        input_ptr + i * im2col_step * input_dim,
        filter_ptr,
        offset_ptr + i * im2col_step * input_offset_dim,
        mask_ptr + i * im2col_step * input_mask_dim,
        output_prt + i * im2col_step * output_dim,
        n,
        c,
        h,
        w,
        f,
        ksize,
        strides,
        paddings,
        dilations,
        groups,
        deformable_groups,
        nullptr,
        nullptr,
        nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "deformable_conv");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    deformable_conv, XPU, ALL_LAYOUT, phi::DeformableConvKernel, float) {}
