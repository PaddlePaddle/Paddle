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

#include "paddle/phi/kernels/deformable_conv_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T, typename Context>
void DeformableConvGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& offset,
                              const DenseTensor& filter,
                              const optional<DenseTensor>& mask,
                              const DenseTensor& out_grad,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::vector<int>& dilations,
                              int deformable_groups,
                              int groups,
                              int im2col_step,
                              DenseTensor* dx,
                              DenseTensor* offset_grad,
                              DenseTensor* filter_grad,
                              DenseTensor* mask_grad) {
  if (x.numel() == 0 || filter.numel() == 0) {
    if (dx) Full<T, Context>(dev_ctx, dx->dims(), 0, dx);
    if (offset_grad)
      Full<T, Context>(dev_ctx, offset_grad->dims(), 0, offset_grad);
    if (filter_grad)
      Full<T, Context>(dev_ctx, filter_grad->dims(), 0, filter_grad);
    if (mask_grad) Full<T, Context>(dev_ctx, mask_grad->dims(), 0, mask_grad);
    return;
  }
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  T* dx_data = nullptr;
  T* dw_data = nullptr;
  T* dmask_data = nullptr;
  T* doffset_data = nullptr;

  if (dx != nullptr) {
    dx_data = dev_ctx.template Alloc<T>(dx);
  }
  if (filter_grad != nullptr) {
    dw_data = dev_ctx.template Alloc<T>(filter_grad);
  }
  if (offset_grad != nullptr) {
    doffset_data = dev_ctx.template Alloc<T>(offset_grad);
  }
  if (mask_grad != nullptr) {
    dmask_data = dev_ctx.template Alloc<T>(mask_grad);
  }

  if (backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId()) ==
      backends::xpu::XPUVersion::XPU1) {
    PADDLE_ENFORCE_EQ(
        deformable_groups == 1,
        true,
        errors::InvalidArgument(("XPU1 only support deformable_groups == 1 in "
                                 "deformable_conv_grad op.")));
  }
  PADDLE_ENFORCE_EQ(filter.dims()[2] <= 8 && filter.dims()[3] <= 8,
                    true,
                    errors::InvalidArgument(
                        "Filter high and weight should less than 8 on xpu "
                        "in deformable_conv_grad op."));

  const int64_t batch_size = x.dims()[0];
  std::vector<int64_t> output_shape_vec(vectorize(out_grad.dims()));
  const T* output_grad_ptr = out_grad.data<T>();
  const T* input_ptr = x.data<T>();
  const T* filter_ptr = filter.data<T>();
  const float* offset_ptr = offset.data<float>();
  if (dx_data == nullptr) {
    dx_data = RAII_GUARD.alloc_l3_or_gm<T>(x.numel());
    PADDLE_ENFORCE_NOT_NULL(
        dx_data, errors::ResourceExhausted("XPU has no enough memory"));
  }
  if (dw_data == nullptr) {
    dw_data = RAII_GUARD.alloc_l3_or_gm<T>(filter.numel());
    PADDLE_ENFORCE_NOT_NULL(
        dw_data, errors::ResourceExhausted("XPU has no enough memory"));
  }
  if (doffset_data == nullptr) {
    doffset_data = RAII_GUARD.alloc_l3_or_gm<T>(offset.numel());
    PADDLE_ENFORCE_NOT_NULL(
        doffset_data, errors::ResourceExhausted("XPU has no enough memory"));
  }
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
  const int64_t mask_numel = input_mask_dim * batch_size;
  if (dmask_data == nullptr) {
    dmask_data = RAII_GUARD.alloc_l3_or_gm<T>(mask_numel);
    PADDLE_ENFORCE_NOT_NULL(
        dmask_data, errors::ResourceExhausted("XPU has no enough memory"));
  }

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
  T* padded_dx_data = nullptr;
  // Match the forward workaround for legal large-padding cases rejected by
  // XDNN: run grad on the materialized padded input, then crop dx back to x's
  // shape.
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
    padded_dx_data = RAII_GUARD.alloc_l3_or_gm<T>(batch_size * conv_input_dim);
    PADDLE_ENFORCE_NOT_NULL(
        padded_dx_data, errors::ResourceExhausted("XPU has no enough memory"));
    conv_paddings = {0, 0};
  }
  T* conv_dx_data = padded_dx_data == nullptr ? dx_data : padded_dx_data;

  T* filter_grad_tmp = RAII_GUARD.alloc_l3_or_gm<T>(filter_grad->numel());
  PADDLE_ENFORCE_NOT_NULL(
      filter_grad_tmp, errors::ResourceExhausted("XPU has no enough memory"));

  // set zeros for d_table_data
  const int zero = 0;
  int r_dx =
      xpu::constant<T>(dev_ctx.x_context(), conv_dx_data, x.numel(), zero);
  if (padded_dx_data != nullptr) {
    r_dx = xpu::constant<T>(
        dev_ctx.x_context(), padded_dx_data, batch_size * conv_input_dim, zero);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r_dx, "constant");
  int r_dw =
      xpu::constant<T>(dev_ctx.x_context(), dw_data, filter.numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r_dw, "constant");
  int r_doffset =
      xpu::constant<T>(dev_ctx.x_context(), doffset_data, offset.numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r_doffset, "constant");
  int r_dmask =
      xpu::constant<T>(dev_ctx.x_context(), dmask_data, mask_numel, zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r_dmask, "constant");
  int r_filter = xpu::constant<T>(
      dev_ctx.x_context(), filter_grad_tmp, filter.numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r_filter, "constant");

  if (groups > 1 && deformable_groups == 1) {
    // XDNN grad rejects grouped deformable conv although forward supports it.
    // Decompose only the grouped-conv case into independent groups and
    // accumulate shared offset/mask gradients, preserving the existing wrapper.
    const int64_t c_per_group = c / groups;
    const int64_t f_per_group = f / groups;
    const int64_t temp_input_dim = c_per_group * conv_h * conv_w;
    const int64_t temp_output_dim =
        f_per_group * output_shape_vec[2] * output_shape_vec[3];
    for (int64_t i = 0; i < batch_size / n; ++i) {
      for (int64_t g = 0; g < groups; ++g) {
        T* temp_x = RAII_GUARD.alloc_l3_or_gm<T>(n * temp_input_dim);
        T* temp_dx = RAII_GUARD.alloc_l3_or_gm<T>(n * temp_input_dim);
        T* temp_filter = RAII_GUARD.alloc_l3_or_gm<T>(
            f_per_group * c_per_group * ksize[0] * ksize[1]);
        T* temp_dw = RAII_GUARD.alloc_l3_or_gm<T>(f_per_group * c_per_group *
                                                  ksize[0] * ksize[1]);
        T* temp_out_grad = RAII_GUARD.alloc_l3_or_gm<T>(n * temp_output_dim);
        T* temp_doffset = RAII_GUARD.alloc_l3_or_gm<T>(n * input_offset_dim);
        T* temp_dmask = RAII_GUARD.alloc_l3_or_gm<T>(n * input_mask_dim);
        PADDLE_ENFORCE_NOT_NULL(
            temp_x, errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_dx, errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_filter, errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_dw, errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_out_grad,
            errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_doffset,
            errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_dmask, errors::ResourceExhausted("XPU has no enough memory"));

        int r = xpu::slice<T>(
            dev_ctx.x_context(),
            conv_input_ptr + i * n * conv_input_dim,
            temp_x,
            std::vector<int64_t>{n, c, conv_h, conv_w},
            std::vector<int64_t>{0, g * c_per_group, 0, 0},
            std::vector<int64_t>{n, (g + 1) * c_per_group, conv_h, conv_w});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");
        r = xpu::slice<T>(
            dev_ctx.x_context(),
            filter_ptr,
            temp_filter,
            std::vector<int64_t>{f, c_per_group, ksize[0], ksize[1]},
            std::vector<int64_t>{g * f_per_group, 0, 0, 0},
            std::vector<int64_t>{
                (g + 1) * f_per_group, c_per_group, ksize[0], ksize[1]});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");
        r = xpu::slice<T>(dev_ctx.x_context(),
                          output_grad_ptr + i * n * output_dim,
                          temp_out_grad,
                          std::vector<int64_t>{
                              n, f, output_shape_vec[2], output_shape_vec[3]},
                          std::vector<int64_t>{0, g * f_per_group, 0, 0},
                          std::vector<int64_t>{n,
                                               (g + 1) * f_per_group,
                                               output_shape_vec[2],
                                               output_shape_vec[3]});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");

        r = xpu::constant<T>(
            dev_ctx.x_context(), temp_dx, n * temp_input_dim, zero);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
        r = xpu::constant<T>(dev_ctx.x_context(),
                             temp_dw,
                             f_per_group * c_per_group * ksize[0] * ksize[1],
                             zero);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
        r = xpu::constant<T>(
            dev_ctx.x_context(), temp_doffset, n * input_offset_dim, zero);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
        r = xpu::constant<T>(
            dev_ctx.x_context(), temp_dmask, n * input_mask_dim, zero);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

        r = xpu::deformable_conv_grad<float, float, float, int>(
            dev_ctx.x_context(),
            temp_x,
            temp_filter,
            offset_ptr + i * n * input_offset_dim,
            mask_ptr + i * n * input_mask_dim,
            temp_out_grad,
            temp_dx,
            temp_dw,
            temp_doffset,
            temp_dmask,
            n,
            c_per_group,
            conv_h,
            conv_w,
            f_per_group,
            ksize,
            std::vector<int64_t>{strides.begin(), strides.end()},
            conv_paddings,
            std::vector<int64_t>{dilations.begin(), dilations.end()},
            1,
            1,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            true);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "deformable_conv_grad");

        T* padded_temp_dx = RAII_GUARD.alloc_l3_or_gm<T>(n * conv_input_dim);
        T* padded_temp_dw = RAII_GUARD.alloc_l3_or_gm<T>(filter.numel());
        PADDLE_ENFORCE_NOT_NULL(
            padded_temp_dx,
            errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            padded_temp_dw,
            errors::ResourceExhausted("XPU has no enough memory"));
        r = xpu::pad<T>(
            dev_ctx.x_context(),
            temp_dx,
            padded_temp_dx,
            std::vector<int64_t>{n, c_per_group, conv_h, conv_w},
            std::vector<int64_t>{0, g * c_per_group, 0, 0},
            std::vector<int64_t>{0, c - (g + 1) * c_per_group, 0, 0},
            static_cast<T>(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
        r = baidu::xpu::api::add<T>(dev_ctx.x_context(),
                                    padded_temp_dx,
                                    conv_dx_data + i * n * conv_input_dim,
                                    conv_dx_data + i * n * conv_input_dim,
                                    n * conv_input_dim);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
        r = xpu::pad<T>(
            dev_ctx.x_context(),
            temp_dw,
            padded_temp_dw,
            std::vector<int64_t>{f_per_group, c_per_group, ksize[0], ksize[1]},
            std::vector<int64_t>{g * f_per_group, 0, 0, 0},
            std::vector<int64_t>{f - (g + 1) * f_per_group, 0, 0, 0},
            static_cast<T>(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
        r = baidu::xpu::api::add<T>(dev_ctx.x_context(),
                                    padded_temp_dw,
                                    dw_data,
                                    dw_data,
                                    filter.numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
        r = baidu::xpu::api::add<T>(dev_ctx.x_context(),
                                    temp_doffset,
                                    doffset_data + i * n * input_offset_dim,
                                    doffset_data + i * n * input_offset_dim,
                                    n * input_offset_dim);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
        r = baidu::xpu::api::add<T>(dev_ctx.x_context(),
                                    temp_dmask,
                                    dmask_data + i * n * input_mask_dim,
                                    dmask_data + i * n * input_mask_dim,
                                    n * input_mask_dim);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
      }
    }
  } else if (groups == 1 && deformable_groups > 1) {
    // Decompose by deformable group to avoid inaccurate dx from XDNN for this
    // shape family. Each call uses one deformable group and full output
    // channels.
    const int64_t c_per_deformable_group = c / deformable_groups;
    const int64_t offset_dim_per_group = input_offset_dim / deformable_groups;
    const int64_t mask_dim_per_group = input_mask_dim / deformable_groups;
    const int64_t temp_input_dim = c_per_deformable_group * conv_h * conv_w;
    for (int64_t i = 0; i < batch_size / n; ++i) {
      for (int64_t g = 0; g < deformable_groups; ++g) {
        T* temp_x = RAII_GUARD.alloc_l3_or_gm<T>(n * temp_input_dim);
        T* temp_dx = RAII_GUARD.alloc_l3_or_gm<T>(n * temp_input_dim);
        T* temp_filter = RAII_GUARD.alloc_l3_or_gm<T>(
            f * c_per_deformable_group * ksize[0] * ksize[1]);
        T* temp_dw = RAII_GUARD.alloc_l3_or_gm<T>(f * c_per_deformable_group *
                                                  ksize[0] * ksize[1]);
        T* temp_offset = RAII_GUARD.alloc_l3_or_gm<T>(n * offset_dim_per_group);
        T* temp_mask = RAII_GUARD.alloc_l3_or_gm<T>(n * mask_dim_per_group);
        T* temp_doffset =
            RAII_GUARD.alloc_l3_or_gm<T>(n * offset_dim_per_group);
        T* temp_dmask = RAII_GUARD.alloc_l3_or_gm<T>(n * mask_dim_per_group);
        PADDLE_ENFORCE_NOT_NULL(
            temp_x, errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_dx, errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_filter, errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_dw, errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_offset, errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_mask, errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_doffset,
            errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            temp_dmask, errors::ResourceExhausted("XPU has no enough memory"));

        int r = xpu::slice<T>(
            dev_ctx.x_context(),
            conv_input_ptr + i * n * conv_input_dim,
            temp_x,
            std::vector<int64_t>{n, c, conv_h, conv_w},
            std::vector<int64_t>{0, g * c_per_deformable_group, 0, 0},
            std::vector<int64_t>{
                n, (g + 1) * c_per_deformable_group, conv_h, conv_w});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");
        r = xpu::slice<T>(
            dev_ctx.x_context(),
            filter_ptr,
            temp_filter,
            std::vector<int64_t>{f, c, ksize[0], ksize[1]},
            std::vector<int64_t>{0, g * c_per_deformable_group, 0, 0},
            std::vector<int64_t>{
                f, (g + 1) * c_per_deformable_group, ksize[0], ksize[1]});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");
        r = xpu::slice<float>(
            dev_ctx.x_context(),
            offset_ptr + i * n * input_offset_dim,
            temp_offset,
            std::vector<int64_t>{n,
                                 deformable_groups * 2 * ksize[0] * ksize[1],
                                 output_shape_vec[2],
                                 output_shape_vec[3]},
            std::vector<int64_t>{0, g * 2 * ksize[0] * ksize[1], 0, 0},
            std::vector<int64_t>{n,
                                 (g + 1) * 2 * ksize[0] * ksize[1],
                                 output_shape_vec[2],
                                 output_shape_vec[3]});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");
        r = xpu::slice<T>(
            dev_ctx.x_context(),
            mask_ptr + i * n * input_mask_dim,
            temp_mask,
            std::vector<int64_t>{n,
                                 deformable_groups * ksize[0] * ksize[1],
                                 output_shape_vec[2],
                                 output_shape_vec[3]},
            std::vector<int64_t>{0, g * ksize[0] * ksize[1], 0, 0},
            std::vector<int64_t>{n,
                                 (g + 1) * ksize[0] * ksize[1],
                                 output_shape_vec[2],
                                 output_shape_vec[3]});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");

        r = xpu::constant<T>(
            dev_ctx.x_context(), temp_dx, n * temp_input_dim, zero);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
        r = xpu::constant<T>(dev_ctx.x_context(),
                             temp_dw,
                             f * c_per_deformable_group * ksize[0] * ksize[1],
                             zero);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
        r = xpu::constant<T>(
            dev_ctx.x_context(), temp_doffset, n * offset_dim_per_group, zero);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
        r = xpu::constant<T>(
            dev_ctx.x_context(), temp_dmask, n * mask_dim_per_group, zero);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

        r = xpu::deformable_conv_grad<float, float, float, int>(
            dev_ctx.x_context(),
            temp_x,
            temp_filter,
            temp_offset,
            temp_mask,
            output_grad_ptr + i * n * output_dim,
            temp_dx,
            temp_dw,
            temp_doffset,
            temp_dmask,
            n,
            c_per_deformable_group,
            conv_h,
            conv_w,
            f,
            ksize,
            std::vector<int64_t>{strides.begin(), strides.end()},
            conv_paddings,
            std::vector<int64_t>{dilations.begin(), dilations.end()},
            1,
            1,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            true);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "deformable_conv_grad");

        T* padded_temp_dx = RAII_GUARD.alloc_l3_or_gm<T>(n * conv_input_dim);
        T* padded_temp_dw = RAII_GUARD.alloc_l3_or_gm<T>(filter.numel());
        T* padded_temp_doffset =
            RAII_GUARD.alloc_l3_or_gm<T>(n * input_offset_dim);
        T* padded_temp_dmask = RAII_GUARD.alloc_l3_or_gm<T>(n * input_mask_dim);
        PADDLE_ENFORCE_NOT_NULL(
            padded_temp_dx,
            errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            padded_temp_dw,
            errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            padded_temp_doffset,
            errors::ResourceExhausted("XPU has no enough memory"));
        PADDLE_ENFORCE_NOT_NULL(
            padded_temp_dmask,
            errors::ResourceExhausted("XPU has no enough memory"));
        r = xpu::pad<T>(
            dev_ctx.x_context(),
            temp_dx,
            padded_temp_dx,
            std::vector<int64_t>{n, c_per_deformable_group, conv_h, conv_w},
            std::vector<int64_t>{0, g * c_per_deformable_group, 0, 0},
            std::vector<int64_t>{0, c - (g + 1) * c_per_deformable_group, 0, 0},
            static_cast<T>(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
        r = baidu::xpu::api::add<T>(dev_ctx.x_context(),
                                    padded_temp_dx,
                                    conv_dx_data + i * n * conv_input_dim,
                                    conv_dx_data + i * n * conv_input_dim,
                                    n * conv_input_dim);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
        r = xpu::pad<T>(
            dev_ctx.x_context(),
            temp_dw,
            padded_temp_dw,
            std::vector<int64_t>{f, c_per_deformable_group, ksize[0], ksize[1]},
            std::vector<int64_t>{0, g * c_per_deformable_group, 0, 0},
            std::vector<int64_t>{0, c - (g + 1) * c_per_deformable_group, 0, 0},
            static_cast<T>(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
        r = baidu::xpu::api::add<T>(dev_ctx.x_context(),
                                    padded_temp_dw,
                                    dw_data,
                                    dw_data,
                                    filter.numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
        r = xpu::pad<T>(
            dev_ctx.x_context(),
            temp_doffset,
            padded_temp_doffset,
            std::vector<int64_t>{n,
                                 2 * ksize[0] * ksize[1],
                                 output_shape_vec[2],
                                 output_shape_vec[3]},
            std::vector<int64_t>{0, g * 2 * ksize[0] * ksize[1], 0, 0},
            std::vector<int64_t>{
                0, (deformable_groups - g - 1) * 2 * ksize[0] * ksize[1], 0, 0},
            static_cast<T>(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
        r = baidu::xpu::api::add<T>(dev_ctx.x_context(),
                                    padded_temp_doffset,
                                    doffset_data + i * n * input_offset_dim,
                                    doffset_data + i * n * input_offset_dim,
                                    n * input_offset_dim);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
        r = xpu::pad<T>(
            dev_ctx.x_context(),
            temp_dmask,
            padded_temp_dmask,
            std::vector<int64_t>{n,
                                 ksize[0] * ksize[1],
                                 output_shape_vec[2],
                                 output_shape_vec[3]},
            std::vector<int64_t>{0, g * ksize[0] * ksize[1], 0, 0},
            std::vector<int64_t>{
                0, (deformable_groups - g - 1) * ksize[0] * ksize[1], 0, 0},
            static_cast<T>(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
        r = baidu::xpu::api::add<T>(dev_ctx.x_context(),
                                    padded_temp_dmask,
                                    dmask_data + i * n * input_mask_dim,
                                    dmask_data + i * n * input_mask_dim,
                                    n * input_mask_dim);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
      }
    }
  } else {
    for (int64_t i = 0; i < batch_size / n; ++i) {
      int r = xpu::deformable_conv_grad<float, float, float, int>(
          dev_ctx.x_context(),
          conv_input_ptr + i * n * conv_input_dim,
          filter_ptr,
          offset_ptr + i * n * input_offset_dim,
          mask_ptr + i * n * input_mask_dim,
          output_grad_ptr + i * n * output_dim,
          conv_dx_data + i * n * conv_input_dim,
          filter_grad_tmp,
          doffset_data + i * n * input_offset_dim,
          dmask_data + i * n * input_mask_dim,
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
          nullptr,
          nullptr,
          true);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "deformable_conv_grad");

      r = baidu::xpu::api::add<T>(dev_ctx.x_context(),
                                  filter_grad_tmp,
                                  dw_data,
                                  dw_data,
                                  filter.numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
    }
  }

  if (padded_dx_data != nullptr) {
    int r_slice = xpu::slice<T>(
        dev_ctx.x_context(),
        padded_dx_data,
        dx_data,
        std::vector<int64_t>{batch_size, c, conv_h, conv_w},
        std::vector<int64_t>{0, 0, paddings[0], paddings[1]},
        std::vector<int64_t>{batch_size, c, paddings[0] + h, paddings[1] + w});
    PADDLE_ENFORCE_XDNN_SUCCESS(r_slice, "slice");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(deformable_conv_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::DeformableConvGradKernel,
                   float) {}
