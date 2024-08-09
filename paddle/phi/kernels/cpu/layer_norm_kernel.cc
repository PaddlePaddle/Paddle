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

#include "paddle/phi/kernels/layer_norm_kernel.h"

#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/funcs/layer_norm_util.h"
#if !defined(PADDLE_WITH_CUDA) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__)
#include "paddle/phi/kernels/funcs/jit/kernels.h"
#endif
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void LayerNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale_opt,
                     const paddle::optional<DenseTensor>& bias_opt,
                     float epsilon,
                     int begin_norm_axis,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* var) {
  const auto x_dims = x.dims();
  auto* scale = scale_opt.get_ptr();
  auto* bias = bias_opt.get_ptr();

  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<T>(mean);
  dev_ctx.template Alloc<T>(var);

  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  DDim matrix_shape({left, right});
  DDim normalized_shape({left});

  auto x_tmp = x;
  x_tmp.Resize(matrix_shape);
  DenseTensor out;
  out.ShareDataWith(*y);
  out.Resize(matrix_shape);
  // resize mean and var to match the shape of resized x_tmp for broadcast
  DenseTensor mean_tmp;
  mean_tmp.ShareDataWith(*mean);
  mean_tmp.Resize(normalized_shape);
  DenseTensor var_tmp;
  var_tmp.ShareDataWith(*var);
  var_tmp.Resize(normalized_shape);

#if defined(PADDLE_WITH_CUDA) || defined(_WIN32) || defined(__APPLE__) || \
    defined(__OSX__)

  funcs::RowwiseMean2D<phi::CPUContext, T> row_mean(left, right, dev_ctx);

  // get mean
  row_mean(dev_ctx, x_tmp, &mean_tmp);

  // get variance

  phi::funcs::ElementwiseCompute<funcs::SubAndSquareFunctor<T>, T>(
      dev_ctx, x_tmp, mean_tmp, funcs::SubAndSquareFunctor<T>(), &out, 0);

  row_mean(dev_ctx, out, &var_tmp);

  // get x_norm
  phi::funcs::ElementwiseCompute<funcs::SubtractFunctor<T>, T>(
      dev_ctx, x_tmp, mean_tmp, funcs::SubtractFunctor<T>(), &out, 0);

  phi::funcs::ElementwiseCompute<funcs::DivAndSqrtFunctor<T>, T>(
      dev_ctx,
      out,
      var_tmp,
      funcs::DivAndSqrtFunctor<T>(static_cast<T>(epsilon)),
      &out,
      0);

  if (scale) {
    phi::funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
        dev_ctx, out, *scale, funcs::MultiplyFunctor<T>(), &out, 1);
  }
  if (bias) {
    phi::funcs::ElementwiseCompute<funcs::AddFunctor<T>, T>(
        dev_ctx, out, *bias, funcs::AddFunctor<T>(), &out, 1);
  }
#else
  PADDLE_ENFORCE_EQ(mean_tmp.numel(),
                    left,
                    common::errors::InvalidArgument(
                        "mean's length (%d) is not equal with expected (%d).",
                        mean_tmp.numel(),
                        left));
  PADDLE_ENFORCE_EQ(var_tmp.numel(),
                    left,
                    common::errors::InvalidArgument(
                        "var's length (%d) is not equal with expected (%d).",
                        var_tmp.numel(),
                        left));
  if (scale) {
    PADDLE_ENFORCE_EQ(
        scale->numel(),
        right,
        common::errors::InvalidArgument(
            "scale's length (%d) is not equal with expected (%d).",
            scale->numel(),
            right));
  }
  if (bias) {
    PADDLE_ENFORCE_EQ(bias->numel(),
                      right,
                      common::errors::InvalidArgument(
                          "bias's length (%d) is not equal with expected (%d).",
                          bias->numel(),
                          right));
  }

  auto ker =
      phi::jit::KernelFuncs<phi::jit::LayerNormTuple<T>, phi::CPUPlace>::Cache()
          .At(right);
  ker(x_tmp.data<T>(),
      out.data<T>(),
      mean_tmp.data<T>(),
      var_tmp.data<T>(),
      scale ? scale->data<T>() : nullptr,
      bias ? bias->data<T>() : nullptr,
      static_cast<int>(left),
      static_cast<float>(epsilon),
      right);
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(
    layer_norm, CPU, ALL_LAYOUT, phi::LayerNormKernel, float, double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
}
