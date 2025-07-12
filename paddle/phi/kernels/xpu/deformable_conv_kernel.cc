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
#include "paddle/phi/kernels/full_kernel.h"

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
  if (x.numel() == 0 || filter.numel() == 0) {
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(out->dims())), 0, out);
    return;
  }
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

  const int64_t batch_size = x.dims()[0];
  std::vector<int64_t> output_shape_vec(common::vectorize(out->dims()));

  const T* input_ptr = x.data<T>();
  const T* filter_ptr = filter.data<T>();
  const float* offset_ptr = offset.data<T>();
  const float* mask_ptr = mask->data<T>();
  T* output_prt = out->data<T>();

  // set zeros for d_table_data
  const int zero = 0;
  int r = xpu::constant<T>(dev_ctx.x_context(), output_prt, out->numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  int64_t input_dim = x.numel() / x.dims()[0];
  int64_t input_offset_dim = offset.numel() / offset.dims()[0];
  int64_t input_mask_dim = mask->numel() / mask->dims()[0];
  int64_t output_dim =
      output_shape_vec[1] * output_shape_vec[2] * output_shape_vec[3];
  std::vector<int64_t> ksize{filter.dims()[2], filter.dims()[3]};
  int64_t n = static_cast<int64_t>(im2col_step);
  int64_t c = x.dims()[1];
  int64_t h = x.dims()[2];
  int64_t w = x.dims()[3];
  int64_t f = filter.dims()[0];

  for (int64_t i = 0; i < batch_size / n; ++i) {
    int r = xpu::deformable_conv<float, float, float, int>(
        dev_ctx.x_context(),
        input_ptr + i * n * input_dim,
        filter_ptr,
        offset_ptr + i * n * input_offset_dim,
        mask_ptr + i * n * input_mask_dim,
        output_prt + i * n * output_dim,
        n,
        c,
        h,
        w,
        f,
        ksize,
        std::vector<int64_t>{strides.begin(), strides.end()},
        std::vector<int64_t>{paddings.begin(), paddings.end()},
        std::vector<int64_t>{dilations.begin(), dilations.end()},
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
