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
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {

template <typename T>
HOSTDEVICE T DmcnIm2colBilinear(const T* bottom_data,
                                const int data_width,
                                const int height,
                                const int width,
                                T h,
                                T w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh;
  T hw = 1 - lw;

  T v1 =
      (h_low >= 0 && w_low >= 0) ? bottom_data[h_low * data_width + w_low] : 0;
  T v2 = (h_low >= 0 && w_high <= width - 1)
             ? bottom_data[h_low * data_width + w_high]
             : 0;
  T v3 = (h_high <= height - 1 && w_low >= 0)
             ? bottom_data[h_high * data_width + w_low]
             : 0;
  T v4 = (h_high <= height - 1 && w_high <= width - 1)
             ? bottom_data[h_high * data_width + w_high]
             : 0;

  T w1 = hh * hw;
  T w2 = hh * lw;
  T w3 = lh * hw;
  T w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T, typename Context>
void ModulatedDeformableIm2col(const Context& dev_ctx,
                               const T* data_im,
                               const T* data_offset,
                               const T* data_mask,
                               const std::vector<int64_t>& im_shape,
                               const std::vector<int64_t>& col_shape,
                               const std::vector<int64_t>& filter_shape,
                               const std::vector<int>& paddings,
                               const std::vector<int>& strides,
                               const std::vector<int>& dilations,
                               const int deformable_groups,
                               T* data_col);

template <typename T, typename Context>
void DeformableConvKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& offset,
                          const DenseTensor& filter,
                          const DenseTensor& mask,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          const std::vector<int>& dilations,
                          int deformable_groups,
                          int groups,
                          int im2col_step,
                          DenseTensor* out) {
  const int batch_size = static_cast<int>(x.dims()[0]);

  std::vector<int64_t> filter_shape_vec(phi::vectorize(filter.dims()));
  std::vector<int64_t> output_shape_vec(phi::vectorize(out->dims()));

  // col_shape_vec: {c_i * k_h * k_w, im2col_step, o_h, o_w}
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
  DenseTensor output_buffer = Empty<T>(dev_ctx, output_buffer_shape_vec);

  int64_t M = output_shape_vec[1] / groups;
  int64_t N = im2col_step * output_shape_vec[2] * output_shape_vec[3];
  int64_t K = x.dims()[1] * filter_shape_vec[2] * filter_shape_vec[3] / groups;

  DenseTensor weight_3d;
  weight_3d.ShareDataWith(filter).Resize(phi::make_ddim({groups, M, K}));

  DenseTensor col_buffer_3d;
  col_buffer_3d.ShareDataWith(col_buffer)
      .Resize(phi::make_ddim({groups, K, N}));

  DenseTensor output_4d;
  output_4d.ShareDataWith(output_buffer)
      .Resize(phi::make_ddim({batch_size / im2col_step, groups, M, N}));

  DDim input_shape = phi::slice_ddim(x.dims(), 1, x.dims().size());
  std::vector<int64_t> input_shape_vec = phi::vectorize(input_shape);

  int input_dim = x.numel() / x.dims()[0];
  int input_offset_dim = offset.numel() / offset.dims()[0];
  int input_mask_dim = mask.numel() / mask.dims()[0];

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  const T* input_ptr = x.data<T>();
  const T* offset_ptr = offset.data<T>();
  const T* mask_ptr = mask.data<T>();
  T* col_buffer_ptr = col_buffer.data<T>();

  for (int i = 0; i < batch_size / im2col_step; ++i) {
    ModulatedDeformableIm2col(dev_ctx,
                              input_ptr + i * im2col_step * input_dim,
                              offset_ptr + i * im2col_step * input_offset_dim,
                              mask_ptr + i * im2col_step * input_mask_dim,
                              input_shape_vec,
                              col_buffer_shape_vec,
                              filter_shape_vec,
                              paddings,
                              strides,
                              dilations,
                              deformable_groups,
                              col_buffer_ptr);
    DenseTensor output_3d = output_4d.Slice(i, i + 1).Resize(
        phi::slice_ddim(output_4d.dims(), 1, output_4d.dims().size()));
    // get the product of pixel and weight
    for (int g = 0; g < groups; ++g) {
      DenseTensor weight_3d_slice = weight_3d.Slice(g, g + 1).Resize(
          phi::slice_ddim(weight_3d.dims(), 1, weight_3d.dims().size()));
      DenseTensor col_buffer_3d_slice =
          col_buffer_3d.Slice(g, g + 1).Resize(phi::slice_ddim(
              col_buffer_3d.dims(), 1, col_buffer_3d.dims().size()));
      DenseTensor output_3d_slice = output_3d.Slice(g, g + 1).Resize(
          phi::slice_ddim(output_3d.dims(), 1, output_3d.dims().size()));
      blas.MatMul(weight_3d_slice,
                  false,
                  col_buffer_3d_slice,
                  false,
                  T(1.0),
                  &output_3d_slice,
                  T(0.0));
    }
  }
  out->ShareDataWith(output_buffer).Resize(phi::make_ddim(output_shape_vec));
}

}  // namespace phi
