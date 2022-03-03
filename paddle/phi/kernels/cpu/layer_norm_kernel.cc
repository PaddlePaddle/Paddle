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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/layer_norm_util.h"

namespace phi {

template <typename T, typename Context>
void LayerNormKernel(const Context& ctx,
                     const DenseTensor& x,
                     paddle::optional<const DenseTensor&> scale,
                     paddle::optional<const DenseTensor&> bias,
                     float epsilon,
                     int begin_norm_axis,
                     bool is_test,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* var) {
  auto* mean = ctx.Output<Tensor>("Mean");
  const auto x_dims = x.dims();

  y->mutable_data<T>(ctx.GetPlace());
  mean->mutable_data<T>(ctx.GetPlace());
  var->mutable_data<T>(ctx.GetPlace());

  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  framework::DDim matrix_shape({left, right});

  x.Resize(matrix_shape);
  DenseTensor out;
  out.ShareDataWith(*y);
  out.Resize(matrix_shape);

#if defined(PADDLE_WITH_CUDA) || defined(_WIN32) || defined(__APPLE__) || \
    defined(__OSX__)
  auto& dev_ctx = ctx.template device_context<DeviceContext>();
  funcs::RowwiseMean2D<DeviceContext, T> row_mean(
      left, right, ctx.device_context());

  // get mean
  row_mean(dev_ctx, x, mean);

  // get variance
  ElementwiseComputeEx<SubAndSquareFunctor<T>, DeviceContext, T>(
      ctx, &x, mean, /*axis*/ 0, SubAndSquareFunctor<T>(), &out);

  row_mean(dev_ctx, out, var);

  // get x_norm
  ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
      ctx, &x, mean, /*axis*/ 0, SubFunctor<T>(), &out);
  ElementwiseComputeEx<DivAndSqrtFunctor<T>, DeviceContext, T>(
      ctx,
      &out,
      var,
      /*axis*/ 0,
      DivAndSqrtFunctor<T>(static_cast<T>(epsilon)),
      &out);

  if (scale) {
    ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
        ctx, &out, scale, /*axis*/ 1, MulFunctor<T>(), &out);
  }
  if (bias) {
    ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
        ctx, &out, bias, /*axis*/ 1, AddFunctor<T>(), &out);
  }
#else
  PADDLE_ENFORCE_EQ(mean->numel(),
                    left,
                    platform::errors::InvalidArgument(
                        "mean's length (%d) is not equal with expected (%d).",
                        mean->numel(),
                        left));
  PADDLE_ENFORCE_EQ(var->numel(),
                    left,
                    platform::errors::InvalidArgument(
                        "var's length (%d) is not equal with expected (%d).",
                        var->numel(),
                        left));
  if (scale) {
    PADDLE_ENFORCE_EQ(
        scale->numel(),
        right,
        platform::errors::InvalidArgument(
            "scale's length (%d) is not equal with expected (%d).",
            scale->numel(),
            right));
  }
  if (bias) {
    PADDLE_ENFORCE_EQ(bias->numel(),
                      right,
                      platform::errors::InvalidArgument(
                          "bias's length (%d) is not equal with expected (%d).",
                          bias->numel(),
                          right));
  }

  auto ker =
      jit::KernelFuncs<jit::LayerNormTuple<T>, platform::CPUPlace>::Cache().At(
          right);
  ker(x.data<T>(),
      out.data<T>(),
      mean->data<T>(),
      var->data<T>(),
      scale ? scale->data<T>() : nullptr,
      bias ? bias->data<T>() : nullptr,
      static_cast<int>(left),
      static_cast<const float>(epsilon),
      right);
#endif
}

}  // namespace phi
