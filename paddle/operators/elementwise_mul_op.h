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

#include "paddle/operators/math/math_function.h"

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class ElemWiseMulKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* z = context.Output<Tensor>("Out");
    z->mutable_data<T>(context.GetPlace());

    auto* input_axis = context.InputVar("axis");
    auto* input_broadcast = context.InputVar("broadcast");

    int axis = -1;
    if (nullptr != input_axis) {
      axis = input_axis->Get<int>();
    }

    int broadcast = 0;
    if (nullptr != input_broadcast) {
      PADDLE_ENFORCE_NOT_NULL(input_axis);
      broadcast = input_broadcast->Get<int>();
    }

    printf("axis:%d broadcast:%d\n", axis, broadcast);
    if (x->dims() == y->dims() || product(y->dims()) == 1) {
      auto x_e = framework::EigenVector<T>::Flatten(*x);
      auto y_e = framework::EigenVector<T>::Flatten(*y);
      auto z_e = framework::EigenVector<T>::Flatten(*z);

      z_e.device(context.GetEigenDevice<Place>()) = x_e * y_e;
      return
    }

    // TODO(gongweibao):
  }
};

template <typename Place, typename T>
class ElemWiseMulGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    dx->mutable_data<T>(ctx.GetPlace());
    dy->mutable_data<T>(ctx.GetPlace());

    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto y_e = framework::EigenVector<T>::Flatten(*y);
    auto dx_e = framework::EigenVector<T>::Flatten(*dx);
    auto dy_e = framework::EigenVector<T>::Flatten(*dy);
    auto dout_e = framework::EigenVector<T>::Flatten(*dout);

    dx_e.device(ctx.GetEigenDevice<Place>()) = dout_e * y_e;
    dy_e.device(ctx.GetEigenDevice<Place>()) = x_e * dout_e;
  }
};

}  // namespace operators
}  // namespace paddle
