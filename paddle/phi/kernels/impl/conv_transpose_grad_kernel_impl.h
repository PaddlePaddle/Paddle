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

#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/operators/math/vol2col.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/conv_transpose_grad_kernel.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/slice.h"

namespace phi {

template <typename T, typename Context>
void ConvTransposeGradRawKernel(const Context& ctx,
                                const DenseTensor& x,
                                const DenseTensor& filter,
                                const DenseTensor& dout,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::string& padding_algorithm,
                                int groups,
                                const std::vector<int>& dilations,
                                const std::string& data_format,
                                DenseTensor* dx,
                                DenseTensor* dfilter) {
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_format);
  // For filter, we do not use const pointer because we will do reshape,
  // but we should avoid modifying its value.
  DenseTensor filter_ = filter;

  if ((!dx) && (!dfilter)) {
    return;
  }

  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;

  auto x_dims = x.dims();
  auto filter_dims = filter_.dims();
  auto dout_dims = dout.dims();
  const int batch_size = static_cast<int>(x.dims()[0]);

  DDim in_data_dims;
  if (data_layout != DataLayout::kNHWC) {
    in_data_dims = slice_ddim(x_dims, 2, x_dims.size());
  } else {
    in_data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  }
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, in_data_dims, strides, ksize);

  // x_shape_vec: {n, c, h, w} or {n, c, d, h, w} for channel_first
  // x_shape_vec: {n, h, w, c} or {n, d, h, w, c} for channel_last
  std::vector<int64_t> x_shape_vec = vectorize(x.dims());
  // filter_shape_vec: {i_c, o_c, k_h, k_w} or {i_c, o_c, k_d, k_h, k_w}
  std::vector<int64_t> filter_shape_vec = vectorize(filter_.dims());

  // use col_shape in the im2col and col2im (or vol2col and col2vol)
  // calculation
  // col_shape_vec: {o_c, k_h, k_w, h, w} or {o_c, k_d, k_h, k_w, d, h, w} for
  size_t data_dim = filter_shape_vec.size() - 2;
  std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
  if (data_layout != DataLayout::kNHWC) {
    col_shape_vec[0] = dout_dims[1];
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = x_shape_vec[j + 2];
    }
  } else {
    col_shape_vec[0] = dout_dims[dout_dims.size() - 1];
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = x_shape_vec[j + 1];
    }
  }
  DDim col_shape(make_ddim(col_shape_vec));

  // use col_matrix_shape in the gemm calculation
  // size: (o_c * k_h * k_w, h * w) or (o_c * k_d * k_h * k_w, d * h * w)
  DDim col_matrix_shape = flatten_to_2d(col_shape, data_dim + 1);

  // output size: (o_c, o_h, o_w) or (o_c, o_d, o_h, o_w) for channel_first
  // output size: (o_h, o_w, o_c) or (o_d, o_h, o_w, o_c) for channel_last
  DDim output_shape = slice_ddim(dout.dims(), 1, dout.dims().size());

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
    filter_matrix_shape = {x_dims[1], col_matrix_shape[0] / groups};
  } else {
    filter_matrix_shape = {x_dims[x_dims.size() - 1],
                           col_matrix_shape[0] / groups};
  }
  filter_.Resize(filter_matrix_shape);

  int in_step = (data_layout != DataLayout::kNHWC
                     ? static_cast<int>(x_dims[1]) / groups
                     : static_cast<int>(x_dims[x_dims.size() - 1]) / groups);
  int col_step = static_cast<int>(col_matrix_shape[0]) / groups;

  // convolution transpose grad on x:
  // im2col + gemm (similar to conv-forward)
  // x need to compute gradient
  auto blas = funcs::GetBlas<Context, T>(ctx);
  if (dx || dfilter) {
    DenseTensor col;
    col.Resize(col_shape);
    ctx.template Alloc<T>(&col);
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    DenseTensor col_matrix;
    col_matrix.ShareDataWith(col);
    col_matrix.Resize(col_matrix_shape);

    DenseTensor dfilter_;
    funcs::SetConstant<Context, T> set_zero;

    paddle::operators::math::
        Im2ColFunctor<paddle::operators::math::ColFormat::kCFO, Context, T>
            im2col;
    paddle::operators::math::Vol2ColFunctor<Context, T> vol2col;
    funcs::ConcatFunctor<Context, T> concat_functor;

    if (dx) {
      ctx.template Alloc<T>(dx);
      set_zero(ctx, dx, static_cast<T>(0));
    }
    if (dfilter) {  // dfilter_ size (i_c, o_c/g, k_h, k_w)
      ctx.template Alloc<T>(dfilter);
      set_zero(ctx, dfilter, static_cast<T>(0));
      dfilter_ = *dfilter;
      dfilter_.Resize(filter_matrix_shape);
    }

    size_t D = x.dims().size();
    for (int i = 0; i < batch_size; i++) {
      // batch with size (o_c, o_h, o_w) or (o_c, o_d, o_h, o_w) for
      // channel_first
      // batch with size (o_h, o_w, o_c) or (o_d, o_h, o_w, o_c) for
      // channel_last
      DenseTensor dout_batch = dout.Slice(i, i + 1).Resize(output_shape);

      if (data_dim == 2U) {
        // im2col: dy -> col matrix
        // from (o_c, o_h, o_w) to (o_c * k_h * k_w, i_h * i_w) for
        // channel_first
        // from (o_h, o_w, o_c) to (o_c * k_h * k_w, i_h * i_w) for
        // channel_last
        im2col(ctx,
               dout_batch,
               dilations_,
               strides,
               std::vector<int>{
                   paddings_[0], paddings_[2], paddings_[1], paddings_[3]},
               &col,
               data_layout);
      } else if (data_dim == 3U) {
        // vol2col: dy -> col_matrix
        // from (o_c, o_d, o_h, o_w) to (o_c * k_d * k_h * k_w, i_d * i_h *
        // i_w) for channel_first
        // from (o_d, o_h, o_w, o_c) to (i_d * i_h * i_w, o_c * k_d * k_h *
        // k_w) for channel_last
        vol2col(
            ctx, dout_batch, dilations_, strides, paddings_, &col, data_layout);
      }
      if (dx) {
        // batch with size (i_c, i_h, i_w) or (i_h, i_w, i_c)
        DenseTensor dx_batch = dx->Slice(i, i + 1).Resize(x_matrix_shape);

        // gemm: dx = filter * dy
        // (i_c, o_c * k_h * k_w) * (o_c * k_h * k_w, i_h * i_w) -> (i_c, i_h
        // * i_w)
        // or
        // (i_c, o_c * k_d * k_h * k_w) * (o_c * k_d * k_h * k_w, i_d * i_h *
        // i_w) -> (i_c,
        // i_d, i_h, i_w)
        // gemm: dx = dy^T * filter^T for channel_last

        std::vector<DenseTensor> dx_batch_vec;
        for (int g = 0; g < groups; g++) {
          // dx_slice: (i_c/g, i_h * i_w) or (i_c/g, i_d * i_h * i_w)
          // for channel_first
          // dx_slice: (i_h * i_w, i_c/g) or (i_d * i_h * i_w, i_c/g)
          // for channel_last
          // filter_slice: (i_c/g, o_c/g * k_h * k_w)
          DenseTensor filter_slice =
              filter_.Slice(g * in_step, (g + 1) * in_step);
          // col_matrix_slice: (o_c/g * k_h * k_w, h * w) or (o_c/g * k_d *
          // k_h * k_w, d * h * w)
          DenseTensor col_matrix_slice =
              col_matrix.Slice(g * col_step, (g + 1) * col_step);
          if (data_layout != DataLayout::kNHWC) {
            DenseTensor dx_slice =
                dx_batch.Slice(g * in_step, (g + 1) * in_step);
            blas.MatMul(filter_slice,
                        false,
                        col_matrix_slice,
                        false,
                        static_cast<T>(1.0),
                        &dx_slice,
                        static_cast<T>(0.0));
          } else {
            DenseTensor dx_slice;
            funcs::Slice<Context, T, 2>(
                ctx, &dx_batch, &dx_slice, g * in_step, (g + 1) * in_step, 1);
            blas.MatMul(col_matrix_slice,
                        true,
                        filter_slice,
                        true,
                        static_cast<T>(1.0),
                        &dx_slice,
                        static_cast<T>(0.0));
            DDim dx_slice_shape;
            if (data_dim == 2U) {
              dx_slice_shape = {x_dims[1], x_dims[2], in_step};
            } else {
              dx_slice_shape = {x_dims[1], x_dims[2], x_dims[3], in_step};
            }
            dx_slice = dx_slice.Resize(dx_slice_shape);
            dx_batch_vec.push_back(dx_slice);
          }
        }
        if (data_layout == DataLayout::kNHWC) {
          concat_functor(ctx, dx_batch_vec, static_cast<int>(D - 2), &dx_batch);
        }
      }
      if (dfilter) {
        // x batch: (i_c, i_h * i_w) or (i_h, i_w * i_c)
        DenseTensor in_batch = x.Slice(i, i + 1).Resize(x_matrix_shape);
        // gemm: d_filter = x * dy^T
        // (i_c, i_h * i_w) * (i_h * i_w, o_c * k_h * k_w) -> (i_c, o_c * k_h
        // * k_w)
        // or
        // (i_c, i_d * i_h * i_w) * (i_d * i_h * i_w, o_c * k_d * k_h * k_w)
        // -> (i_c, o_c * k_d *
        // k_h * k_w)
        // gemm: d_filter = x^T * dy^T for channel_last

        for (int g = 0; g < groups; g++) {
          DenseTensor dfilter_slice =
              dfilter_.Slice(g * in_step, (g + 1) * in_step);
          DenseTensor col_matrix_slice =
              col_matrix.Slice(g * col_step, (g + 1) * col_step);
          if (data_layout != DataLayout::kNHWC) {
            DenseTensor in_batch_slice =
                in_batch.Slice(g * in_step, (g + 1) * in_step);
            blas.MatMul(in_batch_slice,
                        false,
                        col_matrix_slice,
                        true,
                        static_cast<T>(1.0),
                        &dfilter_slice,
                        static_cast<T>(1.0));
          } else {
            DenseTensor in_batch_slice;
            funcs::Slice<Context, T, 2>(ctx,
                                        &in_batch,
                                        &in_batch_slice,
                                        g * in_step,
                                        (g + 1) * in_step,
                                        1);
            blas.MatMul(in_batch_slice,
                        true,
                        col_matrix_slice,
                        true,
                        static_cast<T>(1.0),
                        &dfilter_slice,
                        static_cast<T>(1.0));
          }
        }
      }
    }
  }
}

template <typename T, typename Context>
void Conv2dTransposeGradKernel(const Context& ctx,
                               const DenseTensor& x,
                               const DenseTensor& filter,
                               const DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const IntArray& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               DenseTensor* dx,
                               DenseTensor* dfilter) {
  ConvTransposeGradRawKernel<T, Context>(ctx,
                                         x,
                                         filter,
                                         dout,
                                         strides,
                                         paddings,
                                         padding_algorithm,
                                         groups,
                                         dilations,
                                         data_format,
                                         dx,
                                         dfilter);
}

template <typename T, typename Context>
void Conv3dTransposeGradKernel(const Context& ctx,
                               const DenseTensor& x,
                               const DenseTensor& filter,
                               const DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::vector<int>& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               DenseTensor* dx,
                               DenseTensor* dfilter) {
  ConvTransposeGradRawKernel<T, Context>(ctx,
                                         x,
                                         filter,
                                         dout,
                                         strides,
                                         paddings,
                                         padding_algorithm,
                                         groups,
                                         dilations,
                                         data_format,
                                         dx,
                                         dfilter);
}

}  // namespace phi
