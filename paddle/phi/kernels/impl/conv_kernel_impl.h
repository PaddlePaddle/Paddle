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

#include "paddle/phi/kernels/conv_kernel.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/im2col.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/vol2col.h"

namespace phi {

template <typename T, typename Context>
void ConvKernelImpl(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& filter_t,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings_t,
                    const std::string& padding_algorithm,
                    int groups,
                    const std::vector<int>& dilations_t,
                    const std::string& data_format,
                    DenseTensor* output) {
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;
  DenseTensor filter = filter_t;
  // The filter will be reshaped in the calculations,
  // so here use an assignment operation,
  // that avoids modifying the variable in the Scope.
  dev_ctx.template Alloc<T>(output);

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  DenseTensor transformed_input(input.type());
  DenseTensor transformed_output(output->type());

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(dev_ctx, &input, &transformed_input);
    TransToChannelFirst<Context, T>(dev_ctx, &input, &transformed_input);

    ResizeToChannelFirst<Context, T>(dev_ctx, output, &transformed_output);

  } else {
    transformed_input = input;
    transformed_output = *output;
  }

  // update padding and dilation
  auto trans_in_dims = transformed_input.dims();
  auto filter_dims = filter.dims();

  DDim in_data_dims = slice_ddim(trans_in_dims, 2, trans_in_dims.size());
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());

  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  const int batch_size = static_cast<int>(transformed_input.dims()[0]);

  // filter_shape_vec:
  // {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
  std::vector<int64_t> filter_shape_vec(common::vectorize(filter.dims()));

  // output_shape_vec:
  // {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
  std::vector<int64_t> output_shape_vec(
      common::vectorize(transformed_output.dims()));

  // use col_shape in the im2col calculation
  // col_shape_vec:
  // {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h, k_w,
  // o_d,o_h, o_w}
  size_t data_dim = filter_shape_vec.size() - 2;

  std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
  col_shape_vec[0] = trans_in_dims[1] / groups;
  for (size_t j = 0; j < data_dim; ++j) {
    col_shape_vec[j + 1] = filter_shape_vec[j + 2];
    col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
  }

  DDim col_shape(common::make_ddim(col_shape_vec));

  // use col_matrix_shape in the gemm calculation
  // size:
  // (i_c/g * k_h * k_w, o_h * o_w) or (i_c/g * k_d * k_h * k_w, o_d * o_h *
  // o_w)

  DDim col_matrix_shape = flatten_to_2d(col_shape, data_dim);

  bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);

  DenseTensor col;
  // col_matrix shares the same piece of data with col,
  // but will be reshaped into a two-dimensional matrix shape
  // to call the matrix multiplication interface.
  DenseTensor col_matrix;
  if (is_expand) {
    // col = context.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
    col.Resize(col_shape);
    dev_ctx.template Alloc<T>(&col);
    col_matrix.ShareDataWith(col);
    col_matrix.Resize(col_matrix_shape);
  }

  DDim in_matrix_shape =
      slice_ddim(transformed_input.dims(), 1, transformed_input.dims().size());

  DDim filter_matrix_shape = {filter.dims()[0],
                              filter.numel() / filter.dims()[0]};
  filter.Resize(filter_matrix_shape);

  DDim output_matrix_shape = {
      transformed_output.dims()[1],
      transformed_output.numel() /
          (transformed_output.dims()[0] * transformed_output.dims()[1])};

  // convolution operator: im2col(or vol2col) + gemm
  int in_step = static_cast<int>(transformed_input.dims()[1]) / groups;
  int out_step = static_cast<int>(transformed_output.dims()[1]) / groups;

  phi::funcs::Im2ColFunctor<phi::funcs::ColFormat::kCFO, Context, T> im2col;
  phi::funcs::Vol2ColFunctor<Context, T> vol2col;

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  for (int i = 0; i < batch_size; i++) {
    DenseTensor in_batch =
        transformed_input.Slice(i, i + 1).Resize(in_matrix_shape);
    DenseTensor out_batch =
        transformed_output.Slice(i, i + 1).Resize(output_matrix_shape);

    for (int g = 0; g < groups; g++) {
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
      DenseTensor out_slice = out_batch.Slice(g * out_step, (g + 1) * out_step);
      DenseTensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);
      blas.MatMul(
          filter_slice, false, col_matrix, false, T(1.0), &out_slice, T(0.0));
    }
  }
  if (channel_last) {
    TransToChannelLast<Context, T>(dev_ctx, &transformed_output, output);
  }
}

}  // namespace phi
