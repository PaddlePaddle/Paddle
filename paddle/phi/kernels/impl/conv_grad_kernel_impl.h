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

#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/im2col.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/vol2col.h"

namespace phi {

template <typename T, typename Context>
void ConvGradKernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& filter_t,
                    const DenseTensor& output_grad,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings_t,
                    const std::string& padding_algorithm,
                    const std::vector<int>& dilations_t,
                    int groups,
                    const std::string& data_format,
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
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation<int>(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  const int batch_size = static_cast<int>(transformed_input.dims()[0]);

  // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
  std::vector<int64_t> filter_shape_vec(common::vectorize(filter.dims()));
  // output_shape_vec: {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
  std::vector<int64_t> output_shape_vec(
      common::vectorize(transformed_output_grad.dims()));

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
  DDim col_shape(common::make_ddim(col_shape_vec));

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
    phi::funcs::Col2ImFunctor<phi::funcs::ColFormat::kCFO, Context, T> col2im;
    phi::funcs::Col2VolFunctor<Context, T> col2vol;

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
    phi::funcs::Im2ColFunctor<phi::funcs::ColFormat::kCFO, Context, T> im2col;
    phi::funcs::Vol2ColFunctor<Context, T> vol2col;
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

template <typename T, typename Context>
void ConvGradGradKernel(const Context& dev_ctx,
                        const DenseTensor& input,
                        const DenseTensor& filter,
                        const DenseTensor& out_grad,
                        const paddle::optional<DenseTensor>& input_grad_grad,
                        const paddle::optional<DenseTensor>& filter_grad_grad,
                        const std::vector<int>& strides_t,
                        const std::vector<int>& paddings_t,
                        const std::string& padding_algorithm,
                        const std::vector<int>& dilations_t,
                        int groups,
                        const std::string& data_format,
                        DenseTensor* input_grad,
                        DenseTensor* filter_grad,
                        DenseTensor* out_grad_grad) {
  const DenseTensor* X = &input;
  const DenseTensor* dY = &out_grad;
  const DenseTensor* ddX = input_grad_grad.get_ptr();
  const DenseTensor* ddW_in = filter_grad_grad.get_ptr();

  DenseTensor* ddY = out_grad_grad;
  DenseTensor* dW = filter_grad;
  DenseTensor* dX = input_grad;
  DenseTensor W = filter;

  if (!ddY && !dW && !dX) return;

  const std::vector<int> strides = strides_t;
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  // transform Tensor
  DenseTensor transformed_X(X->type());
  DenseTensor transformed_dY(dY->type());
  DenseTensor transformed_ddX(X->type());

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(dev_ctx, X, &transformed_X);
    TransToChannelFirst<Context, T>(dev_ctx, X, &transformed_X);

    ResizeToChannelFirst<Context, T>(dev_ctx, dY, &transformed_dY);
    TransToChannelFirst<Context, T>(dev_ctx, dY, &transformed_dY);

    if (ddX) {
      ResizeToChannelFirst<Context, T>(dev_ctx, ddX, &transformed_ddX);
      TransToChannelFirst<Context, T>(dev_ctx, ddX, &transformed_ddX);
    }
  } else {
    transformed_X = *X;
    transformed_dY = *dY;
    if (ddX) {
      transformed_ddX = *ddX;
    }
  }

  // update padding and dilation
  auto in_dims = transformed_X.dims();
  auto filter_dims = W.dims();

  DDim in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  const int batch_size = static_cast<int>(transformed_X.dims()[0]);
  std::vector<int64_t> filter_shape_vec(common::vectorize(W.dims()));
  std::vector<int64_t> output_shape_vec(
      common::vectorize(transformed_dY.dims()));

  size_t data_dim = filter_shape_vec.size() - 2;
  std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
  // col_shape [in_channel/group, kh, kw, oh, ow]
  col_shape_vec[0] = transformed_X.dims()[1] / groups;
  for (size_t j = 0; j < data_dim; ++j) {
    col_shape_vec[j + 1] = filter_shape_vec[j + 2];
    col_shape_vec[j + data_dim + 1] = output_shape_vec[j + 2];
  }
  DDim col_shape(common::make_ddim(col_shape_vec));
  // col_matrix_shape [in_channel/group * kh * kw, oh * ow]
  DDim col_matrix_shape = flatten_to_2d(col_shape, data_dim + 1);
  // input_shape [Cin, H, W]
  DDim input_shape =
      slice_ddim(transformed_X.dims(), 1, transformed_X.dims().size());
  // filter_matrix_shape [Cout, Cin * kh * kw]
  DDim filter_matrix_shape = {W.dims()[0], W.numel() / W.dims()[0]};

  W.Resize(filter_matrix_shape);
  DDim output_matrix_shape = {
      transformed_dY.dims()[1],
      transformed_dY.numel() /
          (transformed_dY.dims()[0] * transformed_dY.dims()[1])};
  int in_step = static_cast<int>(transformed_X.dims()[1]) / groups;
  int out_step = static_cast<int>(transformed_dY.dims()[1]) / groups;

  bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
  DenseTensor col;
  DenseTensor col_matrix;
  if (is_expand) {
    col.Resize(col_shape);
    dev_ctx.template Alloc<T>(&col);
    col_matrix.ShareDataWith(col);
    col_matrix.Resize(col_matrix_shape);
  }

  phi::funcs::SetConstant<Context, T> set_zero;
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  // dx convolution double grad:  gemm + col2im(col2vol)
  // dx = ddw * dy  ==> dx(N, Cin, H, W), ddw(Cout, Cin, kh, kw), dy(N, Cout,
  // oH, oW)
  if (dX && ddW_in) {
    Tensor ddW;
    ddW.ShareDataWith(*ddW_in).Resize(filter_matrix_shape);
    dev_ctx.template Alloc<T>(dX);

    DenseTensor transformed_dX(dX->type());

    if (channel_last) {
      ResizeToChannelFirst<Context, T>(dev_ctx, dX, &transformed_dX);

    } else {
      transformed_dX = *dX;
    }
    // if is_expand is false, the operation of set_zero is unnecessary
    // because math::matmul will reset dx
    if (is_expand) {
      set_zero(dev_ctx, &transformed_dX, static_cast<T>(0));
    }
    phi::funcs::Col2ImFunctor<phi::funcs::ColFormat::kCFO, Context, T> col2im;
    phi::funcs::Col2VolFunctor<Context, T> col2vol;

    for (int i = 0; i < batch_size; i++) {
      DenseTensor dy_batch =
          transformed_dY.Slice(i, i + 1).Resize(output_matrix_shape);
      DenseTensor dx_batch = transformed_dX.Slice(i, i + 1).Resize(input_shape);
      for (int g = 0; g < groups; g++) {
        // gemm
        DenseTensor dy_slice = dy_batch.Slice(g * out_step, (g + 1) * out_step);
        DenseTensor ddw_slice = ddW.Slice(g * out_step, (g + 1) * out_step);
        DenseTensor dx_slice = dx_batch.Slice(g * in_step, (g + 1) * in_step);
        if (!is_expand) {
          col_matrix.ShareDataWith(dx_slice);
          col_matrix.Resize(col_matrix_shape);
        }
        blas.MatMul(
            ddw_slice, true, dy_slice, false, T(1.0), &col_matrix, T(0.0));

        if (is_expand && data_dim == 2U) {
          col2im(dev_ctx,
                 col,
                 dilations,
                 strides,
                 std::vector<int>{
                     paddings[0], paddings[2], paddings[1], paddings[3]},
                 &dx_slice);
        } else if (is_expand && data_dim == 3U) {
          col2vol(dev_ctx, col, dilations, strides, paddings, &dx_slice);
        }
      }
    }
    if (channel_last) {
      TransToChannelLast<Context, T>(dev_ctx, &transformed_dX, dX);
    }
  }

  // dw = ddx * dy  ==> dw(Cout, Cin, kh, kw), ddx(N, Cin, H, W), dy(N, Cout,
  // oH, oW)
  // dw convolution double grad:  im2col(vol2col) + gemm
  if (dW && ddX) {
    dev_ctx.template Alloc<T>(dW);
    set_zero(dev_ctx, dW, static_cast<T>(0));
    DenseTensor dW_arr = *dW;
    dW_arr.Resize(filter_matrix_shape);
    phi::funcs::Im2ColFunctor<phi::funcs::ColFormat::kCFO, Context, T> im2col;
    phi::funcs::Vol2ColFunctor<Context, T> vol2col;
    for (int i = 0; i < batch_size; ++i) {
      DenseTensor dy_batch =
          transformed_dY.Slice(i, i + 1).Resize(output_matrix_shape);
      Tensor ddx_batch = transformed_ddX.Slice(i, i + 1).Resize(input_shape);
      for (int g = 0; g < groups; ++g) {
        // im2col
        DenseTensor dy_slice = dy_batch.Slice(g * out_step, (g + 1) * out_step);
        DenseTensor ddx_slice = ddx_batch.Slice(g * in_step, (g + 1) * in_step);
        if (!is_expand) {
          col.ShareDataWith(ddx_slice);
          col_matrix.ShareDataWith(col);
          col_matrix.Resize(col_matrix_shape);
        } else if (data_dim == 2U) {
          im2col(dev_ctx,
                 ddx_slice,
                 dilations,
                 strides,
                 std::vector<int>{
                     paddings[0], paddings[2], paddings[1], paddings[3]},
                 &col);
        } else if (data_dim == 3U) {
          vol2col(dev_ctx, ddx_slice, dilations, strides, paddings, &col);
        }

        DenseTensor dw_slice = dW_arr.Slice(g * out_step, (g + 1) * out_step);
        blas.MatMul(
            dy_slice, false, col_matrix, true, T(1.0), &dw_slice, T(1.0));
      }
    }
  }

  // ddy = w * ddx + x * ddw ==> ddy(N, Cout, oH, oW), x/ddx(N, Cin, H, W),
  // w/ddw(Cout, Cin, kh, kw)
  // ddy convolution double grad: im2col(vol2col) + gemm
  if (ddY) {
    dev_ctx.template Alloc<T>(ddY);

    DenseTensor transformed_ddY(ddY->type());
    if (channel_last) {
      ResizeToChannelFirst<Context, T>(dev_ctx, ddY, &transformed_ddY);
    } else {
      transformed_ddY = *ddY;
    }

    set_zero(dev_ctx, &transformed_ddY, static_cast<T>(0));
    phi::funcs::Im2ColFunctor<phi::funcs::ColFormat::kCFO, Context, T> im2col;
    phi::funcs::Vol2ColFunctor<Context, T> vol2col;
    for (int i = 0; i < batch_size; ++i) {
      DenseTensor ddy_batch =
          transformed_ddY.Slice(i, i + 1).Resize(output_matrix_shape);
      for (int g = 0; g < groups; ++g) {
        // gemm
        DenseTensor ddy_slice =
            ddy_batch.Slice(g * out_step, (g + 1) * out_step);

        if (ddX) {
          DenseTensor ddx_batch =
              transformed_ddX.Slice(i, i + 1).Resize(input_shape);
          DenseTensor ddx_slice =
              ddx_batch.Slice(g * in_step, (g + 1) * in_step);
          if (!is_expand) {
            col.ShareDataWith(ddx_slice);
            col_matrix.ShareDataWith(col);
            col_matrix.Resize(col_matrix_shape);
          } else if (data_dim == 2U) {
            im2col(dev_ctx,
                   ddx_slice,
                   dilations,
                   strides,
                   std::vector<int>{
                       paddings[0], paddings[2], paddings[1], paddings[3]},
                   &col);
          } else if (data_dim == 3U) {
            vol2col(dev_ctx, ddx_slice, dilations, strides, paddings, &col);
          }
          DenseTensor w_slice = W.Slice(g * out_step, (g + 1) * out_step);
          blas.MatMul(
              w_slice, false, col_matrix, false, T(1.0), &ddy_slice, T(0.0));
        }

        if (ddW_in) {
          DenseTensor x_batch =
              transformed_X.Slice(i, i + 1).Resize(input_shape);
          DenseTensor x_slice = x_batch.Slice(g * in_step, (g + 1) * in_step);

          DenseTensor ddW;
          ddW.ShareDataWith(*ddW_in).Resize(filter_matrix_shape);
          if (!is_expand) {
            col.ShareDataWith(x_slice);
            col_matrix.ShareDataWith(col);
            col_matrix.Resize(col_matrix_shape);
          } else if (data_dim == 2U) {
            im2col(dev_ctx,
                   x_slice,
                   dilations,
                   strides,
                   std::vector<int>{
                       paddings[0], paddings[2], paddings[1], paddings[3]},
                   &col);
          } else if (data_dim == 3U) {
            vol2col(dev_ctx, x_slice, dilations, strides, paddings, &col);
          }

          // gemm
          DenseTensor ddw_slice = ddW.Slice(g * out_step, (g + 1) * out_step);
          blas.MatMul(
              ddw_slice, false, col_matrix, false, T(1.0), &ddy_slice, T(1.0));
        }
      }
    }
    if (channel_last) {
      TransToChannelLast<Context, T>(dev_ctx, &transformed_ddY, ddY);
    }
  }
}

}  // namespace phi
