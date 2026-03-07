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

#include "paddle/common/flags.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/cumprod_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/flip_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/impl/reduce_grad.h"
#include "paddle/phi/kernels/nonzero_kernel.h"
#include "paddle/phi/kernels/prod_grad_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {

// Uses exclusive forward and reverse cumulative products to avoid division
// by zero. See:
// input:                        [    a,     b,     c]
// cumprod(exclusive, normal):   [1    ,     a, a * b]
// cumprod(exclusive, reverse):  [b * c,     c,     1]
// product:                      [b * c, a * c, a * b]
template <typename T, typename Context>
DenseTensor ProdSafeZerosBackward(const Context& dev_ctx,
                                  const DenseTensor& grad,
                                  const DenseTensor& inp,
                                  int dim) {
  if (inp.numel() == 0) {
    DenseTensor result;
    result.Resize(inp.dims());
    dev_ctx.template Alloc<T>(&result);
    return result;
  }

  int64_t dim_size = inp.dims()[dim];
  if (dim_size == 1) {
    DenseTensor result;
    result.Resize(grad.dims());
    dev_ctx.template Alloc<T>(&result);
    phi::Copy(dev_ctx, grad, dev_ctx.GetPlace(), false, &result);
    return result;
  }

  // ones: shape same as inp but with size 1 along dim
  auto ones_dims = common::vectorize(inp.dims());
  ones_dims[dim] = 1;
  DenseTensor ones =
      phi::Full<T, Context>(dev_ctx, IntArray(ones_dims), static_cast<T>(1));

  // exclusive_normal = cat([ones, inp[:dim_size-1]], dim).cumprod(dim)
  DenseTensor inp_head = phi::Slice<T, Context>(
      dev_ctx, inp, {static_cast<int64_t>(dim)}, {0}, {dim_size - 1});
  DenseTensor exclusive_normal_input =
      phi::Concat<T, Context>(dev_ctx, {&ones, &inp_head}, dim);

  DenseTensor exclusive_normal;
  exclusive_normal.Resize(exclusive_normal_input.dims());
  dev_ctx.template Alloc<T>(&exclusive_normal);
  phi::CumprodKernel<T, Context>(
      dev_ctx, exclusive_normal_input, dim, false, false, &exclusive_normal);

  // exclusive_reverse = cat([ones, flip(inp[1:], dim)], dim).cumprod(dim)
  //                     .flip(dim)
  DenseTensor inp_tail = phi::Slice<T, Context>(
      dev_ctx, inp, {static_cast<int64_t>(dim)}, {1}, {dim_size});

  DenseTensor inp_tail_flipped;
  inp_tail_flipped.Resize(inp_tail.dims());
  dev_ctx.template Alloc<T>(&inp_tail_flipped);
  phi::FlipKernel<T, Context>(dev_ctx, inp_tail, {dim}, &inp_tail_flipped);

  DenseTensor exclusive_reverse_input =
      phi::Concat<T, Context>(dev_ctx, {&ones, &inp_tail_flipped}, dim);

  DenseTensor exclusive_reverse_cumprod;
  exclusive_reverse_cumprod.Resize(exclusive_reverse_input.dims());
  dev_ctx.template Alloc<T>(&exclusive_reverse_cumprod);
  phi::CumprodKernel<T, Context>(dev_ctx,
                                 exclusive_reverse_input,
                                 dim,
                                 false,
                                 false,
                                 &exclusive_reverse_cumprod);

  DenseTensor exclusive_reverse;
  exclusive_reverse.Resize(exclusive_reverse_cumprod.dims());
  dev_ctx.template Alloc<T>(&exclusive_reverse);
  phi::FlipKernel<T, Context>(
      dev_ctx, exclusive_reverse_cumprod, {dim}, &exclusive_reverse);

  // result = grad * (exclusive_normal * exclusive_reverse)
  DenseTensor product =
      phi::Multiply<T, Context>(dev_ctx, exclusive_normal, exclusive_reverse);
  return phi::Multiply<T, Context>(dev_ctx, grad, product);
}

template <typename T, typename Context>
void ProdGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out,
                    const DenseTensor& out_grad,
                    const IntArray& dims,
                    bool keep_dim,
                    bool reduce_all,
                    DenseTensor* x_grad) {
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  reduce_all = recompute_reduce_all(x, dims, reduce_all);

  if (FLAGS_use_accuracy_compatible_kernel) {
    if (reduce_all) {
      if (x.dims().size() == 0) {
        dev_ctx.template Alloc<T>(x_grad);
        phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
        return;
      }

      // Detect zeros: create (x == 0) mask and count via nonzero
      DenseTensor zeros_like_x = phi::Full<T, Context>(
          dev_ctx, IntArray(common::vectorize(x.dims())), static_cast<T>(0));

      DenseTensor eq_mask;
      eq_mask.Resize(x.dims());
      dev_ctx.template Alloc<bool>(&eq_mask);
      phi::EqualKernel<T, Context>(dev_ctx, x, zeros_like_x, &eq_mask);

      DenseTensor zero_indices;
      phi::NonZeroKernel<bool, Context>(dev_ctx, eq_mask, &zero_indices);
      int64_t num_zeros = zero_indices.dims()[0];

      dev_ctx.template Alloc<T>(x_grad);

      if (num_zeros == 0) {
        // No zeros: grad * (result / input)
        DenseTensor result_div_input = phi::Divide<T, Context>(dev_ctx, out, x);
        DenseTensor grad_result =
            phi::Multiply<T, Context>(dev_ctx, out_grad, result_div_input);
        phi::Copy(dev_ctx, grad_result, dev_ctx.GetPlace(), false, x_grad);
      } else if (num_zeros > 1) {
        // More than one zero: gradient is all zeros
        phi::FullLikeKernel<T, Context>(dev_ctx,
                                        x,
                                        Scalar(static_cast<T>(0)),
                                        phi::CppTypeToDataType<T>::Type(),
                                        x_grad);
      } else {
        // Exactly one zero (or meta tensor): use safe cumprod backward
        // Flatten to 1D, apply along dim=0, reshape back
        DenseTensor x_flat = phi::Reshape<T, Context>(
            dev_ctx, x, {static_cast<int64_t>(x.numel())});
        DenseTensor grad_result =
            ProdSafeZerosBackward<T, Context>(dev_ctx, out_grad, x_flat, 0);
        auto x_shape = common::vectorize<int64_t>(x.dims());
        phi::Reshape<T, Context>(dev_ctx, grad_result, x_shape, x_grad);
      }
    } else {
      auto dim_vec = dims.GetData();

      if (x.dims().size() == 0) {
        dev_ctx.template Alloc<T>(x_grad);
        phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
        return;
      }

      if (dim_vec.size() == 1) {
        int64_t dim = dim_vec[0];
        if (dim < 0) {
          dim += x.dims().size();
        }

        // Unsqueeze grad and result if !keepdim, to match input rank
        DenseTensor grad_expanded = out_grad;
        DenseTensor result_expanded = out;
        if (!keep_dim) {
          auto grad_shape = common::vectorize<int64_t>(out_grad.dims());
          grad_shape.insert(grad_shape.begin() + dim, 1);
          grad_expanded =
              phi::Reshape<T, Context>(dev_ctx, out_grad, grad_shape);

          auto result_shape = common::vectorize<int64_t>(out.dims());
          result_shape.insert(result_shape.begin() + dim, 1);
          result_expanded =
              phi::Reshape<T, Context>(dev_ctx, out, result_shape);
        }

        // Detect zeros
        DenseTensor zeros_like_x = phi::Full<T, Context>(
            dev_ctx, IntArray(common::vectorize(x.dims())), static_cast<T>(0));

        DenseTensor eq_mask;
        eq_mask.Resize(x.dims());
        dev_ctx.template Alloc<bool>(&eq_mask);
        phi::EqualKernel<T, Context>(dev_ctx, x, zeros_like_x, &eq_mask);

        DenseTensor zero_indices;
        phi::NonZeroKernel<bool, Context>(dev_ctx, eq_mask, &zero_indices);
        int64_t total_zeros = zero_indices.dims()[0];

        dev_ctx.template Alloc<T>(x_grad);

        if (total_zeros == 0) {
          // No zeros: grad * (result / input) with broadcasting
          DenseTensor result_div_input =
              phi::Divide<T, Context>(dev_ctx, result_expanded, x);
          DenseTensor grad_result = phi::Multiply<T, Context>(
              dev_ctx, grad_expanded, result_div_input);
          phi::Copy(dev_ctx, grad_result, dev_ctx.GetPlace(), false, x_grad);
        } else {
          // Has zeros: use safe cumprod backward
          DenseTensor grad_result = ProdSafeZerosBackward<T, Context>(
              dev_ctx, grad_expanded, x, static_cast<int>(dim));
          phi::Copy(dev_ctx, grad_result, dev_ctx.GetPlace(), false, x_grad);
        }
      } else {
        // Multiple dims: fall back to original Eigen-based implementation
        ReduceGradKernel<Context, T, funcs::ProdGradFunctor>(dev_ctx,
                                                             x,
                                                             out,
                                                             out_grad,
                                                             dims.GetData(),
                                                             keep_dim,
                                                             reduce_all,
                                                             x_grad);
      }
    }
  } else {
    ReduceGradKernel<Context, T, funcs::ProdGradFunctor>(dev_ctx,
                                                         x,
                                                         out,
                                                         out_grad,
                                                         dims.GetData(),
                                                         keep_dim,
                                                         reduce_all,
                                                         x_grad);
  }
}

}  // namespace phi
