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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/deformable_conv_functor.h"

namespace phi {

template <typename T>
HOSTDEVICE T DmcnGetGradientWeight(T argmax_h,
                                   T argmax_w,
                                   const int h,
                                   const int w,
                                   const int height,
                                   const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  weight = (h == argmax_h_low && w == argmax_w_low)
               ? (h + 1 - argmax_h) * (w + 1 - argmax_w)
               : weight;
  weight = (h == argmax_h_low && w == argmax_w_high)
               ? (h + 1 - argmax_h) * (argmax_w + 1 - w)
               : weight;
  weight = (h == argmax_h_high && w == argmax_w_low)
               ? (argmax_h + 1 - h) * (w + 1 - argmax_w)
               : weight;
  weight = (h == argmax_h_high && w == argmax_w_high)
               ? (argmax_h + 1 - h) * (argmax_w + 1 - w)
               : weight;

  return weight;
}

template <typename T>
HOSTDEVICE T DmcnGetCoordinateWeight(T argmax_h,
                                     T argmax_w,
                                     const int height,
                                     const int width,
                                     const T* im_data,
                                     const int data_width,
                                     const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    weight += (argmax_h_low >= 0 && argmax_w_low >= 0)
                  ? -1 * (argmax_w_low + 1 - argmax_w) *
                        im_data[argmax_h_low * data_width + argmax_w_low]
                  : 0;

    weight += (argmax_h_low >= 0 && argmax_w_high <= width - 1)
                  ? -1 * (argmax_w - argmax_w_low) *
                        im_data[argmax_h_low * data_width + argmax_w_high]
                  : 0;

    weight += (argmax_h_high <= height - 1 && argmax_w_low >= 0)
                  ? (argmax_w_low + 1 - argmax_w) *
                        im_data[argmax_h_high * data_width + argmax_w_low]
                  : 0;
    weight += (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
                  ? (argmax_w - argmax_w_low) *
                        im_data[argmax_h_high * data_width + argmax_w_high]
                  : 0;
  } else if (bp_dir == 1) {
    weight += (argmax_h_low >= 0 && argmax_w_low >= 0)
                  ? -1 * (argmax_h_low + 1 - argmax_h) *
                        im_data[argmax_h_low * data_width + argmax_w_low]
                  : 0;
    weight += (argmax_h_low >= 0 && argmax_w_high <= width - 1)
                  ? (argmax_h_low + 1 - argmax_h) *
                        im_data[argmax_h_low * data_width + argmax_w_high]
                  : 0;
    weight += (argmax_h_high <= height - 1 && argmax_w_low >= 0)
                  ? -1 * (argmax_h - argmax_h_low) *
                        im_data[argmax_h_high * data_width + argmax_w_low]
                  : 0;
    weight += (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
                  ? (argmax_h - argmax_h_low) *
                        im_data[argmax_h_high * data_width + argmax_w_high]
                  : 0;
  }

  return weight;
}

template <typename T, typename Context>
void ModulatedDeformableCol2imCoord(const Context& dev_ctx,
                                    const T* data_col,
                                    const T* data_im,
                                    const T* data_offset,
                                    const T* data_mask,
                                    const std::vector<int64_t>& im_shape,
                                    const std::vector<int64_t>& col_shape,
                                    const std::vector<int64_t>& kernel_shape,
                                    const std::vector<int>& paddings,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& dilations,
                                    const int deformable_groups,
                                    T* grad_offset,
                                    T* grad_mask);

template <typename T, typename Context>
void ModulatedDeformableCol2im(const Context& dev_ctx,
                               const T* data_col,
                               const T* data_offset,
                               const T* data_mask,
                               const std::vector<int64_t>& im_shape,
                               const std::vector<int64_t>& col_shape,
                               const std::vector<int64_t>& kernel_shape,
                               const std::vector<int>& pad,
                               const std::vector<int>& stride,
                               const std::vector<int>& dilation,
                               const int deformable_group,
                               T* grad_im);

template <typename T, typename Context>
void FilterGradAddup(const Context& dev_ctx,
                     const int nthreads,
                     const int n,
                     const int height,
                     const int width,
                     const T* dweight_3d,
                     T* filter_grad);

template <typename T, typename Context>
void DeformableConvGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& offset,
                              const DenseTensor& filter,
                              paddle::optional<const DenseTensor&> mask,
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
  const int batch_size = static_cast<int>(x.dims()[0]);

  DDim input_shape = phi::slice_ddim(x.dims(), 1, x.dims().size());
  std::vector<int64_t> input_shape_vec = phi::vectorize(input_shape);
  std::vector<int64_t> filter_shape_vec(phi::vectorize(filter.dims()));
  std::vector<int64_t> output_shape_vec(phi::vectorize(out_grad.dims()));

  std::vector<int64_t> col_buffer_shape_vec(filter_shape_vec.size());
  col_buffer_shape_vec[0] = x.dims()[1] * filter.dims()[2] * filter.dims()[3];
  col_buffer_shape_vec[1] = im2col_step;
  for (size_t j = 0; j < filter_shape_vec.size() - 2; ++j) {
    col_buffer_shape_vec[j + 2] = output_shape_vec[j + 2];
  }
  std::vector<int64_t> output_buffer_shape_vec(1);
  output_buffer_shape_vec[0] = batch_size * output_shape_vec[1] *
                               output_shape_vec[2] * output_shape_vec[3];

  DenseTensor col_buffer = Empty<T>(dev_ctx, col_buffer_shape_vec);
  DenseTensor output_buffer;
  output_buffer.ShareDataWith(out_grad).Resize(
      make_ddim(output_buffer_shape_vec));

  int64_t M =
      input_shape_vec[0] / groups * filter_shape_vec[2] * filter_shape_vec[3];
  int64_t N = im2col_step * output_shape_vec[2] * output_shape_vec[3];
  int64_t K = output_shape_vec[1] / groups;

  DDim weight_3d_shape = {groups, K, M};
  DDim out_grad_4d_shape = {batch_size / im2col_step, groups, K, N};
  DDim col_buffer_3d_shape = {groups, M, N};
  DDim filter_grad_shape = {groups, K, M};

  DenseTensor weight_3d;
  weight_3d.ShareDataWith(filter).Resize(weight_3d_shape);
  DenseTensor out_grad_4d;
  out_grad_4d.ShareDataWith(output_buffer).Resize(out_grad_4d_shape);
  DenseTensor col_buffer_3d;
  col_buffer_3d.ShareDataWith(col_buffer).Resize(col_buffer_3d_shape);

  phi::funcs::SetConstant<Context, T> set_zero;
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  int input_dim = x.numel() / x.dims()[0];
  int input_offset_dim = offset.numel() / offset.dims()[0];
  int input_mask_dim = mask ? mask->numel() / mask->dims()[0] : 0;

  if (filter_grad) {
    Full<T>(dev_ctx,
            {filter_grad_shape.Get(), filter_grad_shape.size()},
            0,
            filter_grad);
  }

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    set_zero(dev_ctx, dx, static_cast<T>(0));
  }

  if (offset_grad) {
    dev_ctx.template Alloc<T>(offset_grad);
    set_zero(dev_ctx, offset_grad, static_cast<T>(0));

    if (mask_grad) {
      dev_ctx.template Alloc<T>(mask_grad);
      set_zero(dev_ctx, mask_grad, static_cast<T>(0));
    }
  }

  for (int i = 0; i < batch_size / im2col_step; ++i) {
    DenseTensor out_grad_3d = out_grad_4d.Slice(i, i + 1).Resize(
        phi::slice_ddim(out_grad_4d.dims(), 1, out_grad_4d.dims().size()));
    for (int g = 0; g < groups; ++g) {
      DenseTensor weight_3d_slice = weight_3d.Slice(g, g + 1).Resize(
          phi::slice_ddim(weight_3d.dims(), 1, weight_3d.dims().size()));
      DenseTensor out_grad_3d_slice = out_grad_3d.Slice(g, g + 1).Resize(
          phi::slice_ddim(out_grad_3d.dims(), 1, out_grad_3d.dims().size()));
      DenseTensor col_buffer_3d_slice =
          col_buffer_3d.Slice(g, g + 1).Resize(phi::slice_ddim(
              col_buffer_3d.dims(), 1, col_buffer_3d.dims().size()));
      blas.MatMul(weight_3d_slice,
                  true,
                  out_grad_3d_slice,
                  false,
                  T(1.0),
                  &col_buffer_3d_slice,
                  T(0.0));
    }
    col_buffer.Resize(make_ddim(col_buffer_shape_vec));

    T* col_buffer_ptr = col_buffer.data<T>();
    const T* input_ptr = x.data<T>();
    const T* offset_ptr = offset.data<T>();
    const T* mask_data_ptr =
        mask ? mask->data<T>() + i * im2col_step * input_mask_dim : nullptr;
    if (offset_grad) {
      T* offset_grad_ptr = offset_grad->data<T>();
      T* mask_grad_data_ptr =
          mask_grad ? mask_grad->data<T>() + i * im2col_step * input_mask_dim
                    : nullptr;
      // get grad of offset and mask
      ModulatedDeformableCol2imCoord(
          dev_ctx,
          col_buffer_ptr,
          input_ptr + i * im2col_step * input_dim,
          offset_ptr + i * im2col_step * input_offset_dim,
          mask_data_ptr,
          input_shape_vec,
          col_buffer_shape_vec,
          filter_shape_vec,
          paddings,
          strides,
          dilations,
          deformable_groups,
          offset_grad_ptr + i * im2col_step * input_offset_dim,
          mask_grad_data_ptr);
    }
    if (dx) {
      T* dx_ptr = dx->data<T>();
      // get grad of input
      ModulatedDeformableCol2im(dev_ctx,
                                col_buffer_ptr,
                                offset_ptr + i * im2col_step * input_offset_dim,
                                mask_data_ptr,
                                input_shape_vec,
                                col_buffer_shape_vec,
                                filter_shape_vec,
                                paddings,
                                strides,
                                dilations,
                                deformable_groups,
                                dx_ptr + i * im2col_step * input_dim);
      dx->Resize(x.dims());
    }

    funcs::ModulatedDeformableIm2col(
        dev_ctx,
        input_ptr + i * im2col_step * input_dim,
        offset_ptr + i * im2col_step * input_offset_dim,
        mask_data_ptr,
        input_shape_vec,
        col_buffer_shape_vec,
        filter_shape_vec,
        paddings,
        strides,
        dilations,
        deformable_groups,
        col_buffer_ptr);

    col_buffer_3d.Resize(col_buffer_3d_shape);

    if (filter_grad) {
      DenseTensor dweight_3d = Empty<T>(
          dev_ctx, {filter_grad_shape.Get(), filter_grad_shape.size()});
      for (int g = 0; g < groups; ++g) {
        DenseTensor out_grad_3d_slice = out_grad_3d.Slice(g, g + 1).Resize(
            phi::slice_ddim(out_grad_3d.dims(), 1, out_grad_3d.dims().size()));
        DenseTensor col_buffer_3d_slice =
            col_buffer_3d.Slice(g, g + 1).Resize(phi::slice_ddim(
                col_buffer_3d.dims(), 1, col_buffer_3d.dims().size()));
        DenseTensor dweight_3d_slice = dweight_3d.Slice(g, g + 1).Resize(
            phi::slice_ddim(dweight_3d.dims(), 1, dweight_3d.dims().size()));

        blas.MatMul(out_grad_3d_slice,
                    false,
                    col_buffer_3d_slice,
                    true,
                    T(1.0),
                    &dweight_3d_slice,
                    T(0.0));
      }

      // update grad of weights
      FilterGradAddup<T>(dev_ctx,
                         dweight_3d.numel(),
                         groups,
                         K,
                         M,
                         dweight_3d.data<T>(),
                         filter_grad->data<T>());
    }
  }
  if (filter_grad) {
    filter_grad->Resize(filter.dims());
  }
}

}  // namespace phi
