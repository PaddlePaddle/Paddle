/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/tensor_util.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class ElementwiseModKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    // only support Tensor % Scalar
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());

    auto x_dims = x->dims();
    auto y_dims = y->dims();

    PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                      "Rank of first input must >= rank of second input.");
    PADDLE_ENFORCE_EQ(
        y_dims[0] == 1,
        " ElementwiseModOp Input(Y) must be scalar, shape equal 1.");

    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto y_e = framework::Scalar<T, 1>::From(*y);
    auto z_e = framework::EigenVector<T>::Flatten(*z);
    z_e.device(ctx.GetEigenDevice<Place>()) = x_e % static_cast<int>(5);
  }
};

template <typename Place, typename T>
class ElementwiseModGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    // at integer point, the gradient is undefined.
    // https://math.stackexchange.com/questions/1849280/derivative-of-remainder-function-wrt-denominator
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dz = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto y_e = framework::EigenVector<T>::Flatten(*y);
    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto dz_e = framework::EigenVector<T>::Flatten(*dz);

    auto place = ctx.GetEigenDevice<Place>();

    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      auto dx_e = framework::EigenVector<T>::Flatten(*dx);
      dx_e.device(place) = dz_e;
    }

    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      auto dy_e = framework::EigenVector<T>::Flatten(*dy);
      auto floor_div = x_e / y_e;
      dy_e.device(place) = -1.0 * dz_e * floor_div.floor();
    }
  }
};

}  // namespace operators
}  // namespace paddle
