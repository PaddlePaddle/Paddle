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
#include "paddle/phi/kernels/conv_kernel.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void ConvGradGradKernel(const Context& dev_ctx,
                        const DenseTensor& input,
                        const DenseTensor& filter,
                        const DenseTensor& out_grad,
                        paddle::optional<const DenseTensor&> input_grad_grad,
                        paddle::optional<const DenseTensor&> filter_grad_grad,
                        const std::vector<int>& strides_t,
                        const std::vector<int>& paddings_t,
                        const std::string& padding_algorithm,
                        int groups,
                        const std::vector<int>& dilations_t,
                        const std::string& data_format,
                        bool use_addto,
                        int workspace_size_MB,
                        bool exhaustive_search,
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
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  const int batch_size = static_cast<int>(transformed_X.dims()[0]);
  std::vector<int64_t> filter_shape_vec(vectorize(W.dims()));
  std::vector<int64_t> output_shape_vec(vectorize(transformed_dY.dims()));

  size_t data_dim = filter_shape_vec.size() - 2;
  std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
  // col_shape [in_channel/group, kh, kw, oh, ow]
  col_shape_vec[0] = transformed_X.dims()[1] / groups;
  for (size_t j = 0; j < data_dim; ++j) {
    col_shape_vec[j + 1] = filter_shape_vec[j + 2];
    col_shape_vec[j + data_dim + 1] = output_shape_vec[j + 2];
  }
  DDim col_shape(make_ddim(col_shape_vec));
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
    paddle::operators::math::Col2VolFunctor<Context, T> col2vol;
    paddle::operators::math::
        Col2ImFunctor<paddle::operators::math::ColFormat::kCFO, Context, T>
            col2im;

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
    paddle::operators::math::
        Im2ColFunctor<paddle::operators::math::ColFormat::kCFO, Context, T>
            im2col;
    paddle::operators::math::Vol2ColFunctor<Context, T> vol2col;
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
    paddle::operators::math::
        Im2ColFunctor<paddle::operators::math::ColFormat::kCFO, Context, T>
            im2col;
    paddle::operators::math::Vol2ColFunctor<Context, T> vol2col;
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
