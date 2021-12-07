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
#include <algorithm>
#include <limits>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
const int kIgnoreIndex = -100;

// Out = max(X, 0) - X * Labels + log(1 + exp(-abs(X)))
template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Label");
    Tensor *Out = context.Output<Tensor>("Out");
    int ignore_index = context.Attr<int>("ignore_index");
    auto out_data = Out->mutable_data<T>(context.GetPlace());
    int limit = Out->numel();
    auto x_data = X->data<T>();
    auto label_data = Labels->data<T>();
    for (int idx = 0; idx < limit; ++idx) {
      T x = x_data[idx];
      T label = label_data[idx];
      if (static_cast<int>(label) == ignore_index) {
        out_data[idx] = static_cast<T>(0.);
      } else {
        T term1 = (x > 0) ? x : 0;
        T term2 = x * label;
        T term3 = std::log(static_cast<T>(1) + std::exp(-std::abs(x)));
        out_data[idx] = term1 - term2 + term3;
      }
    }
    bool normalize = context.Attr<bool>("normalize");
    if (normalize) {
      int norm = 0;
      T eps = static_cast<T>(1e-6);
      for (int idx = 0; idx < limit; ++idx) {
        T diff = label_data[idx] - static_cast<T>(ignore_index);
        if ((diff < -eps) || (diff > eps)) {
          norm += 1;
        }
      }
      eps = static_cast<T>(1e-5);
      norm = norm > eps ? norm : eps;
      std::for_each(out_data, out_data + limit, [norm](T &v) { v = v / norm; });
    }
  }
};

// dX = sigmoid(X) - labels
template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Label");
    const Tensor *dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *dX = context.Output<Tensor>(framework::GradVarName("X"));
    auto dx_data = dX->mutable_data<T>(context.GetPlace());

    int ignore_index = context.Attr<int>("ignore_index");
    int limit = dX->numel();
    auto x_data = X->data<T>();
    auto label_data = Labels->data<T>();
    auto dout_data = dOut->data<T>();
    for (int idx = 0; idx < limit; ++idx) {
      T x = x_data[idx];
      T label = label_data[idx];
      T dout = dout_data[idx];
      if (static_cast<int>(label) == ignore_index) {
        dx_data[idx] = static_cast<T>(0.);
      } else {
        T simoid_x = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
        T diff = simoid_x - label;
        dx_data[idx] = dout * diff;
      }
    }
    bool normalize = context.Attr<bool>("normalize");
    if (normalize) {
      int norm = 0;
      T eps = static_cast<T>(1e-6);
      for (int idx = 0; idx < limit; ++idx) {
        T diff = label_data[idx] - static_cast<T>(ignore_index);
        if ((diff < -eps) || (diff > eps)) {
          norm += 1;
        }
      }
      eps = static_cast<T>(1e-5);
      norm = norm > eps ? norm : eps;
      std::for_each(dx_data, dx_data + limit, [norm](T &v) { v = v / norm; });
    }
  }
};

}  // namespace operators
}  // namespace paddle
