/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#if !defined(PADDLE_WITH_CUDA) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__)
#include "paddle/fluid/operators/jit/kernels.h"
#endif
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace platform {
class CPUDeviceContext;
class CUDADeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LayerNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

template <typename DeviceContext, typename T>
class LayerNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    auto x = *ctx.Input<Tensor>("X");
    auto* mean = ctx.Input<Tensor>("Mean");
    auto* var = ctx.Input<Tensor>("Variance");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto d_y = *ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");

    // init output
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    const auto& x_dims = x.dims();
    auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);
    framework::DDim matrix_shape({left, right});

    d_y.Resize(matrix_shape);
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    ColwiseSum2D<DeviceContext, T> colwise_sum(left, right,
                                               ctx.device_context());

    Tensor temp;
    Tensor temp_norm;
    if (d_scale || d_x) {
      x.Resize(matrix_shape);
      temp.mutable_data<T>(matrix_shape, ctx.GetPlace());

      temp_norm.mutable_data<T>(matrix_shape, ctx.GetPlace());
      // get x_norm
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &x, mean, /*axis*/ 0, SubFunctor<T>(), &temp_norm);
      ElementwiseComputeEx<DivAndSqrtFunctor<T>, DeviceContext, T>(
          ctx, &temp_norm, var, /*axis*/ 0,
          DivAndSqrtFunctor<T>(static_cast<T>(epsilon)), &temp_norm);
    }

    if (d_bias) {
      d_bias->mutable_data<T>(ctx.GetPlace());
      colwise_sum(dev_ctx, d_y, d_bias);
    }
    if (d_scale) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
          ctx, &temp_norm, &d_y, /*axis*/ 0, MulFunctor<T>(), &temp);
      colwise_sum(dev_ctx, temp, d_scale);
    }

    if (d_x) {
      framework::DDim vec_shape({left});
      d_x->mutable_data<T>(ctx.GetPlace());
      auto dx_dim = d_x->dims();
      Tensor temp_vec;
      temp_vec.mutable_data<T>(vec_shape, ctx.GetPlace());

      RowwiseMean2D<DeviceContext, T> row_mean(left, right,
                                               ctx.device_context());

      if (d_scale) {
        // dy_dx
        ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
            ctx, &d_y, scale, /*axis*/ 1, MulFunctor<T>(), &temp);
        framework::TensorCopy(temp, ctx.GetPlace(), ctx.device_context(), d_x);

        // dy_dmean_dx
        row_mean(dev_ctx, temp, &temp_vec);
        ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
            ctx, d_x, &temp_vec, /*axis*/ 0, SubFunctor<T>(), d_x);

        // dy_var_dx
        ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
            ctx, &temp, &temp_norm, /*axis*/ 0, MulFunctor<T>(), &temp);
      } else {
        // dy_dx
        framework::TensorCopy(d_y, ctx.GetPlace(), ctx.device_context(), d_x);

        // dy_dmean_dx
        row_mean(dev_ctx, d_y, &temp_vec);
        ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
            ctx, d_x, &temp_vec, /*axis*/ 0, SubFunctor<T>(), d_x);

        // dy_var_dx
        ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
            ctx, &d_y, &temp_norm, /*axis*/ 0, MulFunctor<T>(), &temp);
      }
      // dy_var_dx
      row_mean(dev_ctx, temp, &temp_vec);
      ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
          ctx, &temp_norm, &temp_vec, /*axis*/ 0, MulFunctor<T>(), &temp);
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, d_x, &temp, /*axis*/ 0, SubFunctor<T>(), d_x);

      ElementwiseComputeEx<DivAndSqrtFunctor<T>, DeviceContext, T>(
          ctx, d_x, var, /*axis*/ 0,
          DivAndSqrtFunctor<T>(static_cast<T>(epsilon)), d_x);
      d_x->Resize(dx_dim);
    }
  }
};

}  // namespace operators
}  // namespace paddle
