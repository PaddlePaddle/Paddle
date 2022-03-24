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
#include "paddle/phi/kernels/funcs/deformable_conv_functor.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void DeformableConvKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& offset,
                          const DenseTensor& filter,
                          paddle::optional<const DenseTensor&> mask,
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
  int input_mask_dim = mask ? mask->numel() / mask->dims()[0] : 0;

  const T* input_ptr = x.data<T>();
  const T* offset_ptr = offset.data<T>();
  const T* mask_ptr = mask ? mask->data<T>() : nullptr;
  T* col_buffer_ptr = col_buffer.data<T>();

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  for (int i = 0; i < batch_size / im2col_step; ++i) {
    const T* temp_mask_ptr =
        mask_ptr ? mask_ptr + i * im2col_step * input_mask_dim : nullptr;
    funcs::ModulatedDeformableIm2col(
        dev_ctx,
        input_ptr + i * im2col_step * input_dim,
        offset_ptr + i * im2col_step * input_offset_dim,
        temp_mask_ptr,
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
