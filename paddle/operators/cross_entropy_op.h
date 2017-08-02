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
#include "paddle/operators/type_alias.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class OnehotCrossEntropyOpKernel : public OpKernel {
public:
  constexpr T kLOG_THRESHOLD() const { return static_cast<T>(1e-20); }

  void Compute(const ExecutionContext& ctx) const override {
    auto X = ctx.Input<Tensor>("X");
    const T* Xdata = X->data<T>();
    const int* label_data = ctx.Input<Tensor>(1)->data<int>();
    auto Y = ctx.Output<Tensor>("Y");

    Y->mutable_data<T>(ctx.GetPlace());

    T* Ydata = Y->data<T>();

    int batch_size = X->dims()[0];
    int class_num = X->dims()[1];

    // Y[i] = -log(X[i][j])
    for (int i = 0; i < batch_size; ++i) {
      Ydata[i] = -std::log(
          std::max(Xdata[i * class_num + label_data[i]], kLOG_THRESHOLD()));
    }
  }
};

template <typename Place, typename T>
class OnehotCrossEntropyGradientOpKernel : public OpKernel {
public:
  constexpr T kLOG_THRESHOLD() const { return static_cast<T>(1e-20); }

  void Compute(const ExecutionContext& ctx) const override {
    using framework::op_helpers::GenGradName;
    auto X = ctx.Input<Tensor>("X");
    auto dX = ctx.Output<Tensor>(GenGradName("X"));
    auto dY = ctx.Input<Tensor>(GenGradName("Y"));
    auto label = ctx.Input<Tensor>("label");

    auto* dXdata = dX->template mutable_data<T>(ctx.GetPlace());
    auto* dYdata = dY->template data<T>();
    auto* Xdata = X->template data<T>();
    auto* label_data = label->data<int>();

    const int batch_size = X->dims()[0];
    const int class_num = X->dims()[1];

    for (int i = 0; i < batch_size; ++i) {
      dXdata[i * class_num + label_data[i]] =
          -dYdata[i] /
          std::max(Xdata[i * class_num + label_data[i]], kLOG_THRESHOLD());
    }
  }
};

}  // namespace operators
}  // namespace paddle
