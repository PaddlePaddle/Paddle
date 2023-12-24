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

#include "paddle/phi/kernels/layer_norm_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/layer_norm_util.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void LayerNormGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& scale_opt,
                         const paddle::optional<DenseTensor>& bias_opt UNUSED,
                         const DenseTensor& mean,
                         const DenseTensor& variance,
                         const DenseTensor& out_grad,
                         float epsilon,
                         int begin_norm_axis,
                         DenseTensor* x_grad,
                         DenseTensor* scale_grad,
                         DenseTensor* bias_grad) {
  auto* scale = scale_opt.get_ptr();
  auto d_y = out_grad;

  // init output
  auto* d_x = x_grad;
  auto* d_scale = scale_grad;
  auto* d_bias = bias_grad;

  const auto& x_dims = x.dims();
  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  DDim matrix_shape({left, right});
  DDim var_shape({left});

  d_y.Resize(matrix_shape);
  // resize mean and var to match the shape of resized d_y for broadcast (Resize
  // will not modify the underline data)
  auto mean_tmp = mean;
  mean_tmp.Resize(var_shape);
  auto variance_tmp = variance;
  variance_tmp.Resize(var_shape);

  funcs::ColwiseSum2D<phi::CPUContext, T> colwise_sum(left, right, dev_ctx);
  DenseTensor x_tmp = x;

  DenseTensor temp;
  DenseTensor temp_norm;
  if (d_scale || d_x) {
    x_tmp.Resize(matrix_shape);
    temp.Resize(matrix_shape);
    dev_ctx.template Alloc<T>(&temp);

    temp_norm.Resize(matrix_shape);
    dev_ctx.template Alloc<T>(&temp_norm);
    // get x_norm
    phi::funcs::ElementwiseCompute<funcs::SubtractFunctor<T>, T>(
        dev_ctx, x_tmp, mean_tmp, funcs::SubtractFunctor<T>(), &temp_norm, 0);
    phi::funcs::ElementwiseCompute<funcs::DivAndSqrtFunctor<T>, T>(
        dev_ctx,
        temp_norm,
        variance_tmp,
        funcs::DivAndSqrtFunctor<T>(static_cast<T>(epsilon)),
        &temp_norm,
        0);
  }

  if (d_bias) {
    dev_ctx.template Alloc<T>(d_bias);
    colwise_sum(dev_ctx, d_y, d_bias);
  }
  if (d_scale) {
    dev_ctx.template Alloc<T>(d_scale);
    phi::funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
        dev_ctx, temp_norm, d_y, funcs::MultiplyFunctor<T>(), &temp, 0);
    colwise_sum(dev_ctx, temp, d_scale);
  }

  if (d_x) {
    DDim vec_shape({left});
    dev_ctx.template Alloc<T>(d_x);
    auto dx_dim = d_x->dims();
    DenseTensor temp_vec;
    temp_vec.Resize(vec_shape);
    dev_ctx.template Alloc<T>(&temp_vec);

    funcs::RowwiseMean2D<phi::CPUContext, T> row_mean(left, right, dev_ctx);

    if (d_scale) {
      // dy_dx
      phi::funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
          dev_ctx, d_y, *scale, funcs::MultiplyFunctor<T>(), &temp, 1);
      phi::Copy<Context>(dev_ctx, temp, dev_ctx.GetPlace(), false, d_x);

      // dy_dmean_dx
      row_mean(dev_ctx, temp, &temp_vec);
      phi::funcs::ElementwiseCompute<funcs::SubtractFunctor<T>, T>(
          dev_ctx, *d_x, temp_vec, funcs::SubtractFunctor<T>(), d_x, 0);

      // dy_var_dx
      phi::funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
          dev_ctx, temp, temp_norm, funcs::MultiplyFunctor<T>(), &temp, 0);
    } else {
      // dy_dx
      phi::Copy<Context>(dev_ctx, d_y, dev_ctx.GetPlace(), false, d_x);

      // dy_dmean_dx
      row_mean(dev_ctx, d_y, &temp_vec);
      phi::funcs::ElementwiseCompute<funcs::SubtractFunctor<T>, T>(
          dev_ctx, *d_x, temp_vec, funcs::SubtractFunctor<T>(), d_x, 0);

      // dy_var_dx
      phi::funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
          dev_ctx, d_y, temp_norm, funcs::MultiplyFunctor<T>(), &temp, 0);
    }
    // dy_var_dx
    row_mean(dev_ctx, temp, &temp_vec);
    phi::funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
        dev_ctx, temp_norm, temp_vec, funcs::MultiplyFunctor<T>(), &temp, 0);
    phi::funcs::ElementwiseCompute<funcs::SubtractFunctor<T>, T>(
        dev_ctx, *d_x, temp, funcs::SubtractFunctor<T>(), d_x, 0);

    phi::funcs::ElementwiseCompute<funcs::DivAndSqrtFunctor<T>, T>(
        dev_ctx,
        *d_x,
        variance_tmp,
        funcs::DivAndSqrtFunctor<T>(static_cast<T>(epsilon)),
        d_x,
        0);
    d_x->Resize(dx_dim);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    layer_norm_grad, CPU, ALL_LAYOUT, phi::LayerNormGradKernel, float, double) {
}
