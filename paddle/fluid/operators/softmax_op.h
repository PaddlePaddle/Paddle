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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/transpose_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static inline void CalcTransPermAndShapeByAxis(const Tensor& x, const int axis,
                                               std::vector<int>* perm,
                                               std::vector<int>* shape) {
  auto dim_x = x.dims();
  int rank = dim_x.size();

  if (axis == -1 || axis == rank - 1) {
    return;
  }

  for (int i = 0; i < rank - 1; i++) {
    if (i == axis) {
      perm->push_back(rank - 1);
      shape->push_back(dim_x[rank - 1]);
    } else {
      perm->push_back(i);
      shape->push_back(dim_x[i]);
    }
  }
  perm->push_back(axis);
  shape->push_back(dim_x[axis]);
}

template <typename DeviceContext, typename T>
class SoftmaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");
    const int axis = context.Attr<int>("axis");
    int rank = X->dims().size();

    // allocate memory on device.
    Out->mutable_data<T>(context.GetPlace());

    std::vector<int> perm, shape;
    CalcTransPermAndShapeByAxis(*X, axis, &perm, &shape);

    Tensor X_2d, Out_2d;
    Tensor X_trans, Out_trans;
    if (axis != -1 && axis != rank - 1) {
      X_trans.mutable_data<T>(framework::make_ddim(shape), context.GetPlace());
      Out_trans.mutable_data<T>(framework::make_ddim(shape),
                                context.GetPlace());
      TransCompute<DeviceContext, T>(rank, dev_ctx, *X, &X_trans, perm);
      TransCompute<DeviceContext, T>(rank, dev_ctx, *Out, &Out_trans, perm);
      X_2d = framework::ReshapeToMatrix(X_trans, rank - 1);
      Out_2d = framework::ReshapeToMatrix(Out_trans, rank - 1);
    } else {
      X_2d = framework::ReshapeToMatrix(*X, rank - 1);
      Out_2d = framework::ReshapeToMatrix(*Out, rank - 1);
    }

#ifdef PADDLE_ON_INFERENCE
    math::SoftmaxFunctor<DeviceContext, T, true>()(
        context.template device_context<DeviceContext>(), &X_2d, &Out_2d);
#else
    math::SoftmaxFunctor<DeviceContext, T, false>()(
        context.template device_context<DeviceContext>(), &X_2d, &Out_2d);
#endif

    if (axis != -1 && axis != rank - 1) {
      TransCompute<DeviceContext, T>(rank, dev_ctx, Out_trans, Out, perm);
    }
  }
};

template <typename DeviceContext, typename T>
class SoftmaxGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto* Out = context.Input<Tensor>("Out");
    auto* dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));
    const int axis = context.Attr<int>("axis");
    int rank = Out->dims().size();

    // allocate memory on device.
    dX->mutable_data<T>(context.GetPlace());

    std::vector<int> perm, shape;
    CalcTransPermAndShapeByAxis(*dX, axis, &perm, &shape);

    Tensor dX_2d, Out_2d, dOut_2d;
    Tensor dX_trans, Out_trans, dOut_trans;
    if (axis != -1 && axis != rank - 1) {
      dX_trans.mutable_data<T>(framework::make_ddim(shape), context.GetPlace());
      Out_trans.mutable_data<T>(framework::make_ddim(shape),
                                context.GetPlace());
      dOut_trans.mutable_data<T>(framework::make_ddim(shape),
                                 context.GetPlace());
      TransCompute<DeviceContext, T>(rank, dev_ctx, *dX, &dX_trans, perm);
      TransCompute<DeviceContext, T>(rank, dev_ctx, *Out, &Out_trans, perm);
      TransCompute<DeviceContext, T>(rank, dev_ctx, *dOut, &dOut_trans, perm);
      dX_2d = framework::ReshapeToMatrix(dX_trans, rank - 1);
      Out_2d = framework::ReshapeToMatrix(Out_trans, rank - 1);
      dOut_2d = framework::ReshapeToMatrix(dOut_trans, rank - 1);
    } else {
      dX_2d = framework::ReshapeToMatrix(*dX, rank - 1);
      Out_2d = framework::ReshapeToMatrix(*Out, rank - 1);
      dOut_2d = framework::ReshapeToMatrix(*dOut, rank - 1);
    }

    math::SoftmaxGradFunctor<DeviceContext, T>()(
        context.template device_context<DeviceContext>(), &Out_2d, &dOut_2d,
        &dX_2d);

    if (axis != -1 && axis != rank - 1) {
      TransCompute<DeviceContext, T>(rank, dev_ctx, dX_trans, dX, perm);
    }
  }
};

}  // namespace operators
}  // namespace paddle
