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
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
inline T tolerable_value(const T x) {
  static_assert(std::is_floating_point<T>::value,
                "tolerable_value works only on float, "
                "double and double double.");

  const T kApproInf = 1e20;

  if (x == INFINITY) {
    return kApproInf;
  }

  if (x == -INFINITY) {
    return -kApproInf;
  }

  return x;
}

template <typename T>
class OnehotCrossEntropyOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto X = ctx.Input<Tensor>("X");
    const T* Xdata = X->data<T>();
    const int* label_data = ctx.Input<Tensor>("label")->data<int>();
    auto Y = ctx.Output<Tensor>("Y");

    Y->mutable_data<T>(ctx.GetPlace());

    T* Ydata = Y->data<T>();

    int batch_size = X->dims()[0];
    int class_num = X->dims()[1];

    for (int i = 0; i < batch_size; ++i) {
      int index = i * class_num + label_data[i];
      Ydata[i] = -tolerable_value(std::log(Xdata[index]));
    }
  }
};

template <typename T>
class OnehotCrossEntropyGradientOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto X = ctx.Input<Tensor>("X");
    auto dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto dY = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto label = ctx.Input<Tensor>("label");

    auto* dXdata = dX->template mutable_data<T>(ctx.GetPlace());
    auto* dYdata = dY->template data<T>();
    auto* Xdata = X->template data<T>();
    auto* label_data = label->data<int>();

    const int batch_size = X->dims()[0];
    const int class_num = X->dims()[1];

    // TODO(qingqing): make zero setting an common function.
    memset(dXdata, 0, sizeof(T) * batch_size * class_num);
    for (int i = 0; i < batch_size; ++i) {
      int index = i * class_num + label_data[i];
      dXdata[index] = -tolerable_value(dYdata[i] / Xdata[index]);
    }
  }
};

}  // namespace operators
}  // namespace paddle
