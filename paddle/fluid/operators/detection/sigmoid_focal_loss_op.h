/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <cfloat>
#include <limits>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class SigmoidFocalLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<phi::DenseTensor>("X");
    const Tensor *Labels = context.Input<phi::DenseTensor>("Label");
    const Tensor *FgNum = context.Input<phi::DenseTensor>("FgNum");
    Tensor *Out = context.Output<phi::DenseTensor>("Out");
    T gamma = static_cast<T>(context.Attr<float>("gamma"));
    T alpha = static_cast<T>(context.Attr<float>("alpha"));
    auto out_data = Out->mutable_data<T>(context.GetPlace());
    int limit = Out->numel();
    auto x_data = X->data<T>();
    auto label_data = Labels->data<int>();
    auto fg_num_data = FgNum->data<int>();
    auto x_dims = X->dims();
    int num_classes = static_cast<int>(x_dims[1]);

    for (int idx = 0; idx < limit; ++idx) {
      T x = x_data[idx];
      int a = idx / num_classes;  // current sample
      int d = idx % num_classes;  // current class
      int g = label_data[a];      // target

      // Check whether the input data is positive or negative
      // The target classes are in range 1-81
      // and the d is in range 0-80
      T c_pos = static_cast<T>(g == (d + 1));
      T c_neg = static_cast<T>((g != -1) & (g != (d + 1)));
      T fg_num = static_cast<T>((fg_num_data[0] > 1) ? fg_num_data[0] : 1);
      T s_neg = (1.0 - alpha) / fg_num;
      T s_pos = alpha / fg_num;

      // p = 1. / 1. + expf(-x)
      T p = 1. / (1. + std::exp(-x));

      // (1 - p)**gamma * log(p) where
      T term_pos = std::pow(static_cast<T>(1. - p), gamma) *
                   std::log(p > FLT_MIN ? p : FLT_MIN);
      // p**gamma * log(1 - p)
      T term_neg =
          std::pow(p, gamma) *
          (-1. * x * (x >= 0) - std::log(1. + std::exp(x - 2. * x * (x >= 0))));

      out_data[idx] = 0.0;
      out_data[idx] += -c_pos * term_pos * s_pos;
      out_data[idx] += -c_neg * term_neg * s_neg;
    }
  }
};

template <typename DeviceContext, typename T>
class SigmoidFocalLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<phi::DenseTensor>("X");
    const Tensor *Labels = context.Input<phi::DenseTensor>("Label");
    const Tensor *FgNum = context.Input<phi::DenseTensor>("FgNum");
    const Tensor *dOut =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    Tensor *dX = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto dx_data = dX->mutable_data<T>(context.GetPlace());
    T gamma = static_cast<T>(context.Attr<float>("gamma"));
    T alpha = static_cast<T>(context.Attr<float>("alpha"));
    auto x_dims = X->dims();
    int num_classes = static_cast<int>(x_dims[1]);

    int limit = dX->numel();
    auto x_data = X->data<T>();
    auto label_data = Labels->data<int>();
    auto fg_num_data = FgNum->data<int>();
    auto dout_data = dOut->data<T>();
    for (int idx = 0; idx < limit; ++idx) {
      T x = x_data[idx];
      int a = idx / num_classes;  // current sample
      int d = idx % num_classes;  // current class

      T fg_num = static_cast<T>((fg_num_data[0] > 1) ? fg_num_data[0] : 1);
      T s_neg = static_cast<T>((1.0 - alpha) / fg_num);
      T s_pos = alpha / fg_num;
      int g = label_data[a];

      T c_pos = static_cast<T>(g == (d + 1));
      T c_neg = static_cast<T>((g != -1) & (g != (d + 1)));
      T p = 1. / (1. + std::exp(-x));

      // (1-p)**g * (1 - p - g*p*log(p))
      T term_pos = std::pow(static_cast<T>(1. - p), gamma) *
                   (1. - p - (p * gamma * std::log(p > FLT_MIN ? p : FLT_MIN)));
      // (p**g) * (g*(1-p)*log(1-p) - p)
      T term_neg = std::pow(p, gamma) *
                   ((-1. * x * (x >= 0) -
                     std::log(1. + std::exp(x - 2. * x * (x >= 0)))) *
                        (1. - p) * gamma -
                    p);
      dx_data[idx] = 0.0;
      dx_data[idx] += -c_pos * s_pos * term_pos;
      dx_data[idx] += -c_neg * s_neg * term_neg;
      dx_data[idx] = dx_data[idx] * dout_data[idx];
    }
  }
};

}  // namespace operators
}  // namespace paddle
