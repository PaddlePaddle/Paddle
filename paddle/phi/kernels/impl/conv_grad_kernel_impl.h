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
#include "paddle/phi/kernels/conv_grad_kernel.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void ConvGradKernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& filter_t,
                    const DenseTensor& output_grad,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings_t,
                    const std::string& padding_algorithm,
                    int groups,
                    const std::vector<int>& dilations_t,
                    const std::string& data_format,
                    bool use_addto,
                    int workspace_size_MB,
                    bool exhaustive_search,
                    DenseTensor* input_grad,
                    DenseTensor* filter_grad) {
  // The filter and filter_grad will be reshaped in the calculations,
  // so here use an assignment operation,
  // that avoids modifying the variable in the Scope.

  if (!input_grad && !filter_grad) return;
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;

  DenseTensor filter = filter_t;
  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  DenseTensor transformed_input(input.type());
  DenseTensor transformed_output_grad(output_grad.type());

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(dev_ctx, &input, &transformed_input);
    TransToChannelFirst<Context, T>(dev_ctx, &input, &transformed_input);

    ResizeToChannelFirst<Context, T>(
        dev_ctx, &output_grad, &transformed_output_grad);
    TransToChannelFirst<Context, T>(
        dev_ctx, &output_grad, &transformed_output_grad);
  } else {
    transformed_input = input;
    transformed_output_grad = output_grad;
  }

  // update padding and dilation
  auto in_dims = transformed_input.dims();
  auto filter_dims = filter.dims();
  DDim in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation<int>(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  const int batch_size = static_cast<int>(transformed_input.dims()[0]);

  // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
  std::vector<int64_t> filter_shape_vec(vectorize(filter.dims()));
  // output_shape_vec: {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
  std::vector<int64_t> output_shape_vec(
      vectorize(transformed_output_grad.dims()));

  // use col_shape in the im2col calculation
  // col_shape_vec: {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h, k_w, o_d,
  // o_h, o_w}
  size_t data_dim = filter_shape_vec.size() - 2;
  std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
  col_shape_vec[0] = transformed_input.dims()[1] / groups;
  for (size_t j = 0; j < data_dim; ++j) {
    col_shape_vec[j + 1] = filter_shape_vec[j + 2];
    col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
  }
  DDim col_shape(make_ddim(col_shape_vec));

  // use col_matrix_shape in the gemm calculation
  // size: (i_c/g * k_h * k_w, o_h * o_w)
  // or
  // (i_c/g * k_d * k_h * k_w, o_d * o_h * o_w)
  DDim col_matrix_shape = flatten_to_2d(col_shape, data_dim + 1);

  DDim input_shape =
      slice_ddim(transformed_input.dims(), 1, transformed_input.dims().size());

  DDim filter_matrix_shape = {filter.dims()[0],
                              filter.numel() / filter.dims()[0]};
  filter.Resize(filter_matrix_shape);

  DDim output_matrix_shape = {
      transformed_output_grad.dims()[1],
      transformed_output_grad.numel() / (transformed_output_grad.dims()[0] *
                                         transformed_output_grad.dims()[1])};

  // convolution backward input operator:  gemm + col2im(or col2vol)
  // convolution backward weight operator: im2col(or vol2col) + gemm
  int in_step = static_cast<int>(transformed_input.dims()[1]) / groups;
  int out_step = static_cast<int>(transformed_output_grad.dims()[1]) / groups;

  bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);

  DenseTensor col;
  // col_matrix shares the same piece of data with col,
  // but will be reshaped into a two-dimensional matrix shape
  // to call the matrix multiplication interface.
  DenseTensor col_matrix;
  if (is_expand) {
    col.Resize(col_shape);
    dev_ctx.template Alloc<T>(&col);
    col_matrix.ShareDataWith(col);
    col_matrix.Resize(col_matrix_shape);
  }

  phi::funcs::SetConstant<Context, T> set_zero;
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    DenseTensor transformed_input_grad(input_grad->type());
    if (channel_last) {
      ResizeToChannelFirst<Context, T>(
          dev_ctx, input_grad, &transformed_input_grad);

    } else {
      transformed_input_grad = *input_grad;
    }
    // if is_expand is false, the operation of set_zero is unnecessary,
    // because math::matmul will reset input_grad.
    if (is_expand) {
      set_zero(dev_ctx, &transformed_input_grad, static_cast<T>(0));
    }
    paddle::operators::math::Col2VolFunctor<Context, T> col2vol;
    paddle::operators::math::
        Col2ImFunctor<paddle::operators::math::ColFormat::kCFO, Context, T>
            col2im;

    for (int i = 0; i < batch_size; i++) {
      DenseTensor out_grad_batch =
          transformed_output_grad.Slice(i, i + 1).Resize(output_matrix_shape);
      DenseTensor in_grad_batch =
          transformed_input_grad.Slice(i, i + 1).Resize(input_shape);
      for (int g = 0; g < groups; g++) {
        // gemm
        DenseTensor out_grad_slice =
            out_grad_batch.Slice(g * out_step, (g + 1) * out_step);
        DenseTensor filter_slice =
            filter.Slice(g * out_step, (g + 1) * out_step);

        DenseTensor in_grad_slice =
            in_grad_batch.Slice(g * in_step, (g + 1) * in_step);

        if (!is_expand) {
          col_matrix.ShareDataWith(in_grad_slice);
          col_matrix.Resize(col_matrix_shape);
        }
        blas.MatMul(filter_slice,
                    true,
                    out_grad_slice,
                    false,
                    T(1.0),
                    &col_matrix,
                    T(0.0));

        if (is_expand && data_dim == 2U) {
          col2im(dev_ctx,
                 col,
                 dilations,
                 strides,
                 std::vector<int>{
                     paddings[0], paddings[2], paddings[1], paddings[3]},
                 &in_grad_slice);
        } else if (is_expand && data_dim == 3U) {
          col2vol(dev_ctx, col, dilations, strides, paddings, &in_grad_slice);
        }
      }
    }
    if (channel_last) {
      TransToChannelLast<Context, T>(
          dev_ctx, &transformed_input_grad, input_grad);
    }
  }

  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    Tensor filter_grad_ = *filter_grad;
    filter_grad_.Resize(filter_matrix_shape);
    set_zero(dev_ctx, filter_grad, static_cast<T>(0));
    paddle::operators::math::
        Im2ColFunctor<paddle::operators::math::ColFormat::kCFO, Context, T>
            im2col;
    paddle::operators::math::Vol2ColFunctor<Context, T> vol2col;
    for (int i = 0; i < batch_size; i++) {
      DenseTensor out_grad_batch =
          transformed_output_grad.Slice(i, i + 1).Resize(output_matrix_shape);
      DenseTensor in_batch =
          transformed_input.Slice(i, i + 1).Resize(input_shape);
      for (int g = 0; g < groups; g++) {
        // im2col
        DenseTensor out_grad_slice =
            out_grad_batch.Slice(g * out_step, (g + 1) * out_step);
        DenseTensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

        if (!is_expand) {
          col.ShareDataWith(in_slice);
          col_matrix.ShareDataWith(col);
          col_matrix.Resize(col_matrix_shape);
        } else if (data_dim == 2U) {
          im2col(dev_ctx,
                 in_slice,
                 dilations,
                 strides,
                 std::vector<int>{
                     paddings[0], paddings[2], paddings[1], paddings[3]},
                 &col);

        } else if (data_dim == 3U) {
          vol2col(dev_ctx, in_slice, dilations, strides, paddings, &col);
        }

        // gemm
        DenseTensor filter_grad_slice =
            filter_grad_.Slice(g * out_step, (g + 1) * out_step);
        blas.MatMul(out_grad_slice,
                    false,
                    col_matrix,
                    true,
                    T(1.0),
                    &filter_grad_slice,
                    T(1.0));
      }
    }
  }
}

}  // namespace phi
