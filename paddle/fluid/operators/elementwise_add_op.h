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
#include "paddle/fluid/operators/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a + b; }
};

template <typename DeviceContext, typename T>
class ElementwiseAddKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    const auto x = ctx.Input<Tensor>("X");
    const auto y = ctx.Input<Tensor>("Y");
    auto z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");

    auto dims_equal = x->dims() == y->dims();
    if (platform::is_cpu_place(ctx.GetPlace()) && dims_equal) {
      auto eigen_x = framework::EigenVector<T>::Flatten(*x);
      auto eigen_y = framework::EigenVector<T>::Flatten(*y);
      auto eigen_z = framework::EigenVector<T>::Flatten(*z);

      auto blas = math::GetBlas<DeviceContext, T>(ctx);
      blas.VADD(x->numel(), eigen_x.data(), eigen_y.data(), eigen_z.data());
    } else {
      ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                            AddFunctor<T>(), z);
    }
  }
};

template <typename T>
struct IdentityGrad {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename DeviceContext, typename T>
class ElementwiseAddGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    ElemwiseGradCompute<DeviceContext, T, IdentityGrad<T>, IdentityGrad<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, IdentityGrad<T>(),
        IdentityGrad<T>());
  }
};

}  // namespace operators
}  // namespace paddle
