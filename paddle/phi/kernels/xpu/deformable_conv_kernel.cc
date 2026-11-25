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
                          const optional<DenseTensor>& mask,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          const std::vector<int>& dilations,
                          int deformable_groups,
                          int groups,
                          int im2col_step,
                          DenseTensor* out) {
  if (x.numel() == 0 || filter.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }
  dev_ctx.template Alloc<T>(out);

  if (backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId()) ==
      backends::xpu::XPUVersion::XPU1) {
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
  std::vector<int64_t> output_shape_vec(vectorize(out->dims()));

  const T* input_ptr = x.data<T>();
  const T* filter_ptr = filter.data<T>();
  const float* offset_ptr = offset.data<T>();
  T* output_prt = out->data<T>();

  int64_t output_dim =
      output_shape_vec[1] * output_shape_vec[2] * output_shape_vec[3];
  std::vector<int64_t> ksize{filter.dims()[2], filter.dims()[3]};

  DenseTensor effective_mask;
  const T* mask_ptr = nullptr;
  int64_t input_mask_dim = 0;
  if (mask) {
    mask_ptr = mask->data<T>();
    input_mask_dim = mask->numel() / mask->dims()[0];
  } else {
    // Deformable conv v1 has no modulation mask. XDNN expects a valid mask
    // buffer, so use an all-one effective mask to preserve v1 semantics.
    effective_mask.Resize({batch_size,
                           deformable_groups * ksize[0] * ksize[1],
                           output_shape_vec[2],
                           output_shape_vec[3]});
    T* effective_mask_ptr = dev_ctx.template Alloc<T>(&effective_mask);
    int r_mask = xpu::constant<T>(
        dev_ctx.x_context(), effective_mask_ptr, effective_mask.numel(), 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r_mask, "constant");
    mask_ptr = effective_mask_ptr;
    input_mask_dim = effective_mask.numel() / effective_mask.dims()[0];
  }

  // set zeros for d_table_data
  const int zero = 0;
  int r = xpu::constant<T>(dev_ctx.x_context(), output_prt, out->numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  int64_t input_dim = x.numel() / x.dims()[0];
  int64_t input_offset_dim = offset.numel() / offset.dims()[0];
  int64_t n = static_cast<int64_t>(im2col_step);
  int64_t c = x.dims()[1];
  int64_t h = x.dims()[2];
  int64_t w = x.dims()[3];
  int64_t f = filter.dims()[0];
  std::vector<int64_t> conv_paddings{paddings.begin(), paddings.end()};

  DenseTensor padded_x;
  const T* conv_input_ptr = input_ptr;
  int64_t conv_h = h;
  int64_t conv_w = w;
  int64_t conv_input_dim = input_dim;
  // XDNN rejects legal Paddle cases whose symmetric padding is as large as the
  // kernel. Materialize the zero padding and call XDNN with padding 0 instead;
  // this keeps the math identical while avoiding XDNN_INVALID_PARAM.
  if (paddings[0] >= ksize[0] || paddings[1] >= ksize[1]) {
    conv_h = h + 2 * paddings[0];
    conv_w = w + 2 * paddings[1];
    padded_x.Resize({batch_size, c, conv_h, conv_w});
    T* padded_x_ptr = dev_ctx.template Alloc<T>(&padded_x);
    int r_pad =
        xpu::pad<T>(dev_ctx.x_context(),
                    input_ptr,
                    padded_x_ptr,
                    std::vector<int64_t>{batch_size, c, h, w},
                    std::vector<int64_t>{0, 0, paddings[0], paddings[1]},
                    std::vector<int64_t>{0, 0, paddings[0], paddings[1]},
                    static_cast<T>(0));
    PADDLE_ENFORCE_XDNN_SUCCESS(r_pad, "pad");
    conv_input_ptr = padded_x_ptr;
    conv_input_dim = c * conv_h * conv_w;
    conv_paddings = {0, 0};
  }

  for (int64_t i = 0; i < batch_size / n; ++i) {
    int r = xpu::deformable_conv<float, float, float, int>(
        dev_ctx.x_context(),
        conv_input_ptr + i * n * conv_input_dim,
        filter_ptr,
        offset_ptr + i * n * input_offset_dim,
        mask_ptr + i * n * input_mask_dim,
        output_prt + i * n * output_dim,
        n,
        c,
        conv_h,
        conv_w,
        f,
        ksize,
        std::vector<int64_t>{strides.begin(), strides.end()},
        conv_paddings,
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
