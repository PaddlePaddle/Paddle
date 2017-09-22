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
#include "paddle/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
HOSTDEVICE T tolerable_value(const T x) {
  PADDLE_ASSERT(std::is_floating_point<T>::value);
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
class CrossEntropyOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto x = ctx.Input<Tensor>("X");
    auto y = ctx.Output<Tensor>("Y");

    auto* x_data = x->data<T>();
    y->mutable_data<T>(ctx.GetPlace());
    auto* y_data = y->data<T>();

    int batch_size = x->dims()[0];
    int class_num = x->dims()[1];

    if (ctx.Attr<int>("soft_label") == 1) {
      auto* label_data = ctx.Input<Tensor>("Label")->data<T>();
      int index = 0;
      for (int i = 0; i < batch_size; ++i) {
        T sum = static_cast<T>(0);
        for (int j = 0; j < class_num; ++j) {
          sum += label_data[index] * tolerable_value(std::log(x_data[index]));
          y_data[i] = -sum;
          index++;
        }
      }
    } else {
      auto* label_data = ctx.Input<Tensor>("Label")->data<int>();
      for (int i = 0; i < batch_size; ++i) {
        int index = i * class_num + label_data[i];
        y_data[i] = -tolerable_value(std::log(x_data[index]));
      }
    }
  }
};

template <typename T>
class CrossEntropyGradientOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto x = ctx.Input<Tensor>("X");
    auto dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto label = ctx.Input<Tensor>("Label");

    auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    auto* dy_data = dy->data<T>();
    auto* x_data = x->data<T>();

    int batch_size = x->dims()[0];
    int class_num = x->dims()[1];

    // TODO(qingqing): make zero setting an common function.
    if (ctx.Attr<int>("soft_label") == 1) {
      auto* label_data = ctx.Input<Tensor>("Label")->data<T>();
      int index = 0;
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < class_num; ++j) {
          dx_data[index] = -label_data[index] * dy_data[i] / x_data[index];
          index++;
        }
      }
    } else {
      auto* label_data = label->data<int>();
      memset(dx_data, 0, sizeof(T) * batch_size * class_num);
      for (int i = 0; i < batch_size; ++i) {
        PADDLE_ASSERT(label_data[i] >= 0 || label_data[i] < class_num);
        int index = i * class_num + label_data[i];
        dx_data[index] = -dy_data[i] / x_data[index];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
