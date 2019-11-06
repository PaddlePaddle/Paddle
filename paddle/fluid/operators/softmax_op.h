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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/softmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

static inline int SizeToAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeFromAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename DeviceContext, typename T>
class SoftmaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");
    const int rank = X->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = X->dims()[axis];

    // allocate memory on device.
    Out->mutable_data<T>(context.GetPlace());

    const int n = SizeToAxis(axis, X->dims());
    const int d = SizeFromAxis(axis, X->dims());
    Tensor X_2d, Out_2d;
    X_2d.ShareDataWith(*X).Resize({n, d});
    Out_2d.ShareDataWith(*Out).Resize({n, d});

#ifdef PADDLE_ON_INFERENCE
    math::SoftmaxFunctor<DeviceContext, T, true>()(
        context.template device_context<DeviceContext>(), axis_dim, &X_2d,
        &Out_2d);
#else
    math::SoftmaxFunctor<DeviceContext, T, false>()(
        context.template device_context<DeviceContext>(), axis_dim, &X_2d,
        &Out_2d);
#endif
  }
};

template <typename DeviceContext, typename T>
class SoftmaxGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* Out = context.Input<Tensor>("Out");
    auto* dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));
    const int rank = dX->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int axis_dim = dX->dims()[axis];

    // allocate memory on device.
    dX->mutable_data<T>(context.GetPlace());

    const int n = SizeToAxis(axis, dX->dims());
    const int d = SizeFromAxis(axis, dX->dims());
    Tensor dX_2d, Out_2d, dOut_2d;
    dX_2d.ShareDataWith(*dX).Resize({n, d});
    Out_2d.ShareDataWith(*Out).Resize({n, d});
    dOut_2d.ShareDataWith(*dOut).Resize({n, d});

    math::SoftmaxGradFunctor<DeviceContext, T>()(
        context.template device_context<DeviceContext>(), axis_dim, &Out_2d,
        &dOut_2d, &dX_2d);
  }
};

}  // namespace operators
}  // namespace paddle
