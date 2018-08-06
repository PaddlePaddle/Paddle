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
void default_elementwise_add(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y, framework::Tensor* z) {
  int axis = ctx.Attr<int>("axis");
  ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                        AddFunctor<T>(), z);
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_add(const framework::ExecutionContext& ctx,
                const framework::Tensor* x, const framework::Tensor* y,
                framework::Tensor* z) {
  auto eigen_x = framework::EigenVector<T>::Flatten(*x);
  auto eigen_y = framework::EigenVector<T>::Flatten(*y);
  auto eigen_z = framework::EigenVector<T>::Flatten(*z);

  auto blas = math::GetBlas<DeviceContext, T>(ctx);
  blas.VADD(x->numel(), eigen_x.data(), eigen_y.data(), eigen_z.data());
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    !std::is_floating_point<T>::value ||
    !std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_add(const framework::ExecutionContext& ctx,
                const framework::Tensor* x, const framework::Tensor* y,
                framework::Tensor* z) {
  default_elementwise_add<DeviceContext, T>(ctx, x, y, z);
}

template <typename DeviceContext, typename T>
class ElementwiseAddKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    const auto x = ctx.Input<Tensor>("X");
    const auto y = ctx.Input<Tensor>("Y");
    auto z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());

    auto dims_equal = x->dims() == y->dims();
    if (dims_equal) {
      elementwise_add<DeviceContext, T>(ctx, x, y, z);
    } else {
      default_elementwise_add<DeviceContext, T>(ctx, x, y, z);
    }
  }
};

template <typename T>
struct IdentityGrad {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename DeviceContext, typename T>
void default_elementwise_add_grad(const framework::ExecutionContext& ctx,
                                  const framework::Tensor* x,
                                  const framework::Tensor* y,
                                  const framework::Tensor* out,
                                  const framework::Tensor* dout,
                                  framework::Tensor* dx,
                                  framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");

  ElemwiseExplicitGradCompute<DeviceContext, T, IdentityGrad<T>,
                              IdentityGrad<T>>(ctx, *x, *y, *out, *dout, axis,
                                               dx, dy, IdentityGrad<T>(),
                                               IdentityGrad<T>());
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_add_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  auto blas = math::GetBlas<DeviceContext, T>(ctx);

  if (dx) {
    blas.VCOPY(dout->numel(), dout->data<T>(),
               dx->mutable_data<T>(ctx.GetPlace()));
  }

  if (dy) {
    blas.VCOPY(dout->numel(), dout->data<T>(),
               dy->mutable_data<T>(ctx.GetPlace()));
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    !std::is_floating_point<T>::value ||
    !std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_add_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  default_elementwise_add_grad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
}

template <typename DeviceContext, typename T>
class ElementwiseAddGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    if (dx != nullptr) dx->ShareDataWith(*dout);
    if (dy == nullptr) return;

    if (x->dims() == y->dims()) {
      dy->ShareDataWith(*dout);
    } else {
      dy->mutable_data<T>(ctx.GetPlace());
      // Perform reduction to dout to calculate dy
      const framework::DDim& x_dim = x->dims();
      framework::DDim y_dim = y->dims();
      int axis = ctx.Attr<int>("axis");
      axis = (axis == -1 ? x_dim.size() - y_dim.size() : axis);
      y_dim = trim_trailing_singular_dims(y_dim);
      axis = (y_dim.size() == 0) ? x_dim.size() : axis;

      auto* device =
          ctx.template device_context<DeviceContext>().eigen_device();
      int pre, n, post;
      get_mid_dims(x_dim, y_dim, axis, &pre, &n, &post);
      auto eigen_dout = framework::EigenTensor<T, 3>::From(
          *dout, framework::make_ddim({pre, n, post}));
      auto eigen_dy =
          framework::EigenTensor<T, 1>::From(*dy, framework::make_ddim({n}));
      eigen_dy.device(*device) = eigen_dout.sum(
          framework::EigenDim<2>::From(framework::make_ddim({0, 2})));
    }
  }
};

}  // namespace operators
}  // namespace paddle
