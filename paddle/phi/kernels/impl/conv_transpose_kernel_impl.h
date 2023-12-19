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

#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/phi/kernels/conv_transpose_kernel.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/im2col.h"
#include "paddle/phi/kernels/funcs/slice.h"
#include "paddle/phi/kernels/funcs/vol2col.h"

namespace phi {

template <typename T, typename Context>
void ConvTransposeRawKernel(const Context& ctx,
                            const DenseTensor& x,
                            const DenseTensor& filter,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::string& padding_algorithm,
                            int groups,
                            const std::vector<int>& dilations,
                            const std::string& data_format,
                            DenseTensor* out) {
  const DataLayout data_layout = common::StringToDataLayout(data_format);
  // The filter will be reshaped, so it should not be constant
  DenseTensor filter_ = filter;
  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;

  auto x_dims = x.dims();
  auto filter_dims = filter_.dims();
  auto out_dims = out->dims();
  const int batch_size = static_cast<int>(x.dims()[0]);

  DDim in_data_dims;
  if (data_layout != DataLayout::kNHWC) {
    in_data_dims = slice_ddim(x_dims, 2, x_dims.size());
  } else {
    in_data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  }
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, in_data_dims, strides, ksize);

  // x_shape_vec: {n, c, h, w} or {n, c, d, h, w} for channel_first
  // x_shape_vec: {n, h, w, c} or {n, d, h, w, c} for channel_last
  std::vector<int64_t> x_shape_vec = common::vectorize(x.dims());
  // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
  std::vector<int64_t> filter_shape_vec = common::vectorize(filter_.dims());

  // use col_shape in the im2col and col2im (or vol2col and col2vol)
  // calculation
  // col_shape_vec: {o_c/g, k_h, k_w, h, w} or {o_c/g, k_d, k_h, k_w, d, h, w}
  size_t data_dim = filter_shape_vec.size() - 2;
  std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
  if (data_layout != DataLayout::kNHWC) {
    col_shape_vec[0] = out_dims[1] / groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = x_shape_vec[j + 2];
    }
  } else {
    col_shape_vec[0] = out_dims[out_dims.size() - 1] / groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = x_shape_vec[j + 1];
    }
  }
  DDim col_shape(common::make_ddim(col_shape_vec));

  // use col_matrix_shape in the gemm calculation
  // size: (o_c/g * k_h * k_w, h * w) or (o_c/g * k_d * k_h * k_w, d * h * w)
  DDim col_matrix_shape = flatten_to_2d(col_shape, data_dim + 1);

  DenseTensor col;
  col.Resize(col_shape);
  ctx.template Alloc<T>(&col);
  // col_matrix shares the same piece of data with col,
  // but will be reshaped into a two-dimensional matrix shape
  // to call the matrix multiplication interface.
  DenseTensor col_matrix;
  col_matrix.ShareDataWith(col);
  col_matrix.Resize(col_matrix_shape);

  // out size: (o_c, o_h, o_w) or (o_c, o_d, o_h, o_w) for channel_first
  // out size: (o_h, o_w, o_c) or (o_d, o_h, o_w, o_c) for channel_last
  DDim out_shape = slice_ddim(out->dims(), 1, out->dims().size());

  // x matrix size: (i_c, h * w) or (i_c, d * h * w) for channel_first
  // x matrix size: (h * w, i_c) or (d * h * w, i_c) for channel_last
  DDim x_matrix_shape;
  if (data_layout != DataLayout::kNHWC) {
    x_matrix_shape = {x_dims[1], col_matrix_shape[1]};
  } else {
    x_matrix_shape = {col_matrix_shape[1], x_dims[x_dims.size() - 1]};
  }

  // filter size: (i_c, o_c/g * k_h * k_w) or (i_c, o_c/g * k_d * k_h * k_w)
  DDim filter_matrix_shape;
  if (data_layout != DataLayout::kNHWC) {
    filter_matrix_shape = {x_dims[1], col_matrix_shape[0]};
  } else {
    filter_matrix_shape = {x_dims[x_dims.size() - 1], col_matrix_shape[0]};
  }
  filter_.Resize(filter_matrix_shape);

  ctx.template Alloc<T>(out);

  funcs::SetConstant<Context, T> set_zero;

  auto blas = funcs::GetBlas<Context, T>(ctx);
  set_zero(ctx, out, static_cast<T>(0));

  int in_step = (data_layout != DataLayout::kNHWC
                     ? static_cast<int>(x_dims[1]) / groups
                     : static_cast<int>(x_dims[x_dims.size() - 1]) / groups);

  int out_step =
      (data_layout != DataLayout::kNHWC
           ? static_cast<int>(out_dims[1]) / groups
           : static_cast<int>(out_dims[out_dims.size() - 1]) / groups);
  phi::funcs::Col2ImFunctor<phi::funcs::ColFormat::kCFO, Context, T> col2im;
  phi::funcs::Col2VolFunctor<Context, T> col2vol;
  funcs::ConcatFunctor<Context, T> concat_functor;

  // convolution transpose: gemm + col2im or col2vol (similar to conv-backward
  // on x)
  size_t D = x.dims().size();
  for (int i = 0; i < batch_size; i++) {
    // batch with size (i_c, h * w) or (i_c, d * h * w) for channel_first
    // batch with size (h * w, i_c) or (d * h * w, i_c) for channel_last
    DenseTensor x_batch = x.Slice(i, i + 1).Resize(x_matrix_shape);

    // out size: (o_c, o_h, o_w) or (o_c, o_d, o_h, o_w) for channel_first
    // out size: (o_h, o_w, o_c) or (o_d, o_h, o_w, o_c) for channel_last
    DenseTensor out_batch = out->Slice(i, i + 1).Resize(out_shape);

    std::vector<DenseTensor> out_batch_vec;
    for (int g = 0; g < groups; g++) {
      int64_t start = g * in_step;
      int64_t end = (g + 1) * in_step;
      int axes = (data_layout != DataLayout::kNHWC ? 0 : 1);
      DenseTensor filter_slice = filter_.Slice(g * in_step, (g + 1) * in_step);
      DenseTensor in_slice, out_slice;

      // col_matrix = filter_slice * x_slice
      // of shape (o_c/g * k_h * k_w, h * w)
      // or (o_c/g * k_d * k_h * k_w, d * h * w)
      if (data_layout != DataLayout::kNHWC) {
        in_slice = x_batch.Slice(g * in_step, (g + 1) * in_step);
        out_slice = out_batch.Slice(g * out_step, (g + 1) * out_step);
        blas.MatMul(filter_slice,
                    true,
                    in_slice,
                    false,
                    static_cast<T>(1.0),
                    &col_matrix,
                    static_cast<T>(0.0));
      } else {
        funcs::Slice<Context, T, 2>(ctx, &x_batch, &in_slice, start, end, axes);
        start = g * out_step;
        end = (g + 1) * out_step;
        axes = D - 2;
        if (D == 4U) {
          funcs::Slice<Context, T, 3>(
              ctx, &out_batch, &out_slice, start, end, axes);
        } else if (D == 5U) {
          funcs::Slice<Context, T, 4>(
              ctx, &out_batch, &out_slice, start, end, axes);
        }
        blas.MatMul(filter_slice,
                    true,
                    in_slice,
                    true,
                    static_cast<T>(1.0),
                    &col_matrix,
                    static_cast<T>(0.0));
      }

      if (data_dim == 2U) {
        // col2im: col_matrix -> dy from (o_c/g * k_h * k_w, h * w) to (o_c/g,
        // o_h, o_w) or (o_h, o_w, o_c/g)
        col2im(ctx,
               col,
               dilations_,
               strides,
               std::vector<int>{
                   paddings_[0], paddings_[2], paddings_[1], paddings_[3]},
               &out_slice,
               data_layout);
      } else if (data_dim == 3U) {
        // col2vol: col_matrix -> dy from (o_c/g * k_d * k_h * k_w, d * h * w)
        // to (o_c/g, o_d, o_h, o_w) or (o_d, o_h, o_w, o_c/g)
        col2vol(
            ctx, col, dilations_, strides, paddings_, &out_slice, data_layout);
      }
      if (data_layout == DataLayout::kNHWC) {
        out_batch_vec.push_back(out_slice);
      }
    }
    if (data_layout == DataLayout::kNHWC) {
      concat_functor(ctx, out_batch_vec, static_cast<int>(D - 2), &out_batch);
    }
  }
}

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding UNUSED,
                           const IntArray& output_size UNUSED,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  ConvTransposeRawKernel<T, Context>(ctx,
                                     x,
                                     filter,
                                     strides,
                                     paddings,
                                     padding_algorithm,
                                     groups,
                                     dilations,
                                     data_format,
                                     out);
}

template <typename T, typename Context>
void Conv3dTransposeKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding UNUSED,
                           const std::vector<int>& output_size UNUSED,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  ConvTransposeRawKernel<T, Context>(ctx,
                                     x,
                                     filter,
                                     strides,
                                     paddings,
                                     padding_algorithm,
                                     groups,
                                     dilations,
                                     data_format,
                                     out);
}

}  // namespace phi
