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

namespace phi {

template <typename T, typename Context>
void DeformableConvGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& offset,
                              const DenseTensor& filter,
                              const paddle::optional<DenseTensor>& mask,
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

  if (phi::backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId()) ==
      phi::backends::xpu::XPUVersion::XPU1) {
    PADDLE_ENFORCE_EQ(
        deformable_groups == 1,
        true,
        errors::InvalidArgument(("XPU1 only support deformable_groups == 1 in "
                                 "deformable_conv_grad op.")));
  }
  PADDLE_ENFORCE_EQ(
      groups == 1,
      true,
      errors::InvalidArgument(
          ("XPU only support groups == 1 in deformable_conv_grad op.")));
  PADDLE_ENFORCE_EQ(filter.dims()[2] <= 8 && filter.dims()[3] <= 8,
                    true,
                    errors::InvalidArgument(
                        "Filter high and weight should less than 8 on xpu "
                        "in deformable_conv_grad op."));

  const int batch_size = static_cast<int>(x.dims()[0]);
  std::vector<int64_t> output_shape_vec(common::vectorize(out_grad.dims()));
  const T* output_grad_ptr = out_grad.data<T>();
  const T* input_ptr = x.data<T>();
  const T* filter_ptr = filter.data<T>();
  const float* offset_ptr = offset.data<float>();
  const float* mask_ptr = mask->data<float>();
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
  if (dmask_data == nullptr) {
    dmask_data = RAII_GUARD.alloc_l3_or_gm<T>(mask->numel());
    PADDLE_ENFORCE_NOT_NULL(
        dmask_data, errors::ResourceExhausted("XPU has no enough memory"));
  }

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

  T* filter_grad_tmp = RAII_GUARD.alloc_l3_or_gm<T>(filter_grad->numel());
  PADDLE_ENFORCE_NOT_NULL(
      filter_grad_tmp, errors::ResourceExhausted("XPU has no enough memory"));

  // set zeros for d_table_data
  const int zero = 0;
  int r_dx = xpu::constant<T>(dev_ctx.x_context(), dx_data, x.numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r_dx, "constant");
  int r_dw =
      xpu::constant<T>(dev_ctx.x_context(), dw_data, filter.numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r_dw, "constant");
  int r_doffset =
      xpu::constant<T>(dev_ctx.x_context(), doffset_data, offset.numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r_doffset, "constant");
  int r_dmask =
      xpu::constant<T>(dev_ctx.x_context(), dmask_data, mask->numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r_dmask, "constant");
  int r_filter = xpu::constant<T>(
      dev_ctx.x_context(), filter_grad_tmp, filter.numel(), zero);
  PADDLE_ENFORCE_XDNN_SUCCESS(r_filter, "constant");

  for (int i = 0; i < batch_size / im2col_step; ++i) {
    int r = xpu::deformable_conv_grad<float, float, float, int>(
        dev_ctx.x_context(),
        input_ptr + i * im2col_step * input_dim,
        filter_ptr,
        offset_ptr + i * im2col_step * input_offset_dim,
        mask_ptr + i * im2col_step * input_mask_dim,
        output_grad_ptr + i * im2col_step * output_dim,
        dx_data + i * im2col_step * input_dim,
        filter_grad_tmp,
        doffset_data + i * im2col_step * input_offset_dim,
        dmask_data + i * im2col_step * input_mask_dim,
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
        nullptr,
        nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "deformable_conv_grad");

    r = baidu::xpu::api::add<T>(
        dev_ctx.x_context(), filter_grad_tmp, dw_data, dw_data, filter.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(deformable_conv_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::DeformableConvGradKernel,
                   float) {}
