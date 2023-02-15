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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SquaredL2DistanceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<phi::DenseTensor>("X");
    auto* in1 = context.Input<phi::DenseTensor>("Y");
    auto* out0 = context.Output<phi::DenseTensor>("sub_result");
    auto* out1 = context.Output<phi::DenseTensor>("Out");

    auto in0_dims = in0->dims();
    auto in1_dims = in1->dims();

    int cols = in0->numel() / in0_dims[0];
    // reduce dimensions except the first
    auto x = framework::EigenMatrix<T>::From(
        *in0, phi::make_ddim({in0_dims[0], cols}));
    auto y = framework::EigenMatrix<T>::From(
        *in1, phi::make_ddim({in1_dims[0], cols}));

    out0->mutable_data<T>(context.GetPlace());
    out1->mutable_data<T>(context.GetPlace());
    auto sub_result = framework::EigenMatrix<T>::From(*out0);
    auto z = framework::EigenVector<T>::Flatten(*out1);

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto x_dims = x.dimensions();
    auto y_dims = y.dimensions();
    // buffer the substraction result
    if (y_dims[0] == 1 && x_dims[0] > y_dims[0]) {
      sub_result.device(place) =
          x -
          y.broadcast(Eigen::array<int, 2>({{static_cast<int>(x_dims[0]), 1}}));
    } else {
      sub_result.device(place) = x - y;
    }
    auto sub_res_pow2 = sub_result * sub_result;
    z.device(place) = sub_res_pow2.sum(Eigen::array<int, 1>({{1}}));
  }
};

template <typename DeviceContext, typename T>
class SquaredL2DistanceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<phi::DenseTensor>("sub_result");
    auto* in1 = context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* x_g = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* y_g = context.Output<phi::DenseTensor>(framework::GradVarName("Y"));

    PADDLE_ENFORCE_NOT_NULL(
        x_g,
        platform::errors::NotFound(
            "variable(%s) cannot be found "
            "in scope for operator 'squared_l2_distance_grad'.",
            framework::GradVarName("X")));
    PADDLE_ENFORCE_NOT_NULL(
        y_g,
        platform::errors::NotFound(
            "variable(%s) cannot be found "
            "in scope for operator 'squared_l2_distance_grad'.",
            framework::GradVarName("Y")));

    auto sub_result = framework::EigenMatrix<T>::From(*in0);
    auto out_grad = framework::EigenMatrix<T>::From(*in1);

    auto x_dims = x_g->dims();
    auto y_dims = y_g->dims();

    int cols = x_g->numel() / x_dims[0];
    // calculate gradient
    auto grad_mat = 2 *
                    (out_grad.broadcast(Eigen::array<int, 2>({{1, cols}}))) *
                    sub_result;

    // propagate back to input
    auto& eigen_place =
        *context.template device_context<DeviceContext>().eigen_device();

    x_g->mutable_data<T>(context.GetPlace());
    // eigen matrix
    auto x_grad = framework::EigenMatrix<T>::From(
        *x_g, phi::make_ddim({x_dims[0], cols}));
    // dimensions are same with subResult
    x_grad.device(eigen_place) = grad_mat;

    y_g->mutable_data<T>(context.GetPlace());

    PADDLE_ENFORCE_GE(sub_result.dimensions()[0],
                      y_dims[0],
                      platform::errors::InvalidArgument(
                          "First dimension of gradient must be greater or "
                          "equal than first dimension of target. But received "
                          "gradient dimension = %d and target dimension is %d.",
                          sub_result.dimensions()[0],
                          y_dims[0]));

    if (sub_result.dimensions()[0] == y_dims[0]) {
      auto y_grad = framework::EigenMatrix<T>::From(
          *y_g, phi::make_ddim({y_dims[0], cols}));
      y_grad.device(eigen_place) = -1 * grad_mat;
    } else {
      auto col_sum_res = -1 * (grad_mat.sum(Eigen::array<int, 1>({{0}})));
      auto y_grad = framework::EigenVector<T>::Flatten(*y_g);
      y_grad.device(eigen_place) = col_sum_res;
    }
  }
};

}  // namespace operators
}  // namespace paddle
