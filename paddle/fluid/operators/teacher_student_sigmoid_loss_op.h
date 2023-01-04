/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class TeacherStudentSigmoidLossOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    phi::DenseTensor* y = context.Output<phi::DenseTensor>("Y");
    const phi::DenseTensor* x = context.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* labels = context.Input<phi::DenseTensor>("Label");
    T* y_data = y->mutable_data<T>(context.GetPlace());
    const T* x_data = x->data<T>();
    const T* label_data = labels->data<T>();
    int64_t batch_size = x->dims()[0];
    // loss = max(x, 0) - x * z + log(1 + exp(-abs(x))) + max(x, 0) - x * z' +
    // log(1 + exp(-abs(x)))
    // z is click or not
    // z' is value q of feed_fine
    // label = {-2, -1, [0, 2]}
    // when z' is not exist, clk = 0 : label = -2;
    // when z' is not exist, clk = 1 : label = -1;
    // when z' is exist    , clk = 0 : label = 0 + z';
    // when z' is exist    , clk = 1 : label = 1 + z';
    for (int i = 0; i < batch_size; ++i) {
      if (label_data[i] < -1.0) {
        y_data[i] = (x_data[i] > 0 ? x_data[i] : 0.0) +
                    log(1.0 + exp(-fabs(x_data[i])));
      } else if (label_data[i] < 0.0) {
        y_data[i] = (x_data[i] > 0 ? x_data[i] : 0.0) - x_data[i] +
                    log(1.0 + exp(-fabs(x_data[i])));
      } else if (label_data[i] < 1.0) {
        y_data[i] = (x_data[i] > 0 ? x_data[i] : 0.0) +
                    log(1.0 + exp(-fabs(x_data[i]))) +
                    (x_data[i] > 0 ? x_data[i] : 0.0) -
                    x_data[i] * label_data[i] +
                    log(1.0 + exp(-fabs(x_data[i])));
      } else {
        y_data[i] = (x_data[i] > 0 ? x_data[i] : 0.0) - x_data[i] +
                    log(1.0 + exp(-fabs(x_data[i]))) +
                    (x_data[i] > 0 ? x_data[i] : 0.0) -
                    x_data[i] * (label_data[i] - 1.0) +
                    log(1.0 + exp(-fabs(x_data[i])));
      }
    }
  }
};

template <typename T>
class TeacherStudentSigmoidLossGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const phi::DenseTensor* x = context.Input<phi::DenseTensor>("X");
    const T* x_data = x->data<T>();

    phi::DenseTensor* dx =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    const phi::DenseTensor* labels = context.Input<phi::DenseTensor>("Label");
    const T* label_data = labels->data<T>();

    T soft_max_up_bound =
        static_cast<T>(context.Attr<float>("soft_max_up_bound"));
    T soft_max_lower_bound =
        static_cast<T>(context.Attr<float>("soft_max_lower_bound"));

    int64_t batch_size = x->dims()[0];

    const phi::DenseTensor* dOut =
        context.Input<phi::DenseTensor>(framework::GradVarName("Y"));

    const T* dout_data = dOut->data<T>();

    for (int i = 0; i < batch_size; ++i) {
      T sum_val = x_data[i];
      if (sum_val > soft_max_up_bound) {
        sum_val = soft_max_up_bound;
      } else {
        if (sum_val < soft_max_lower_bound) {
          sum_val = soft_max_lower_bound;
        }
      }

      T pred = 1.0 / (1.0 + exp(-sum_val));
      if (label_data[i] < -1.0) {
        dx_data[i] = 0.0 - pred;
      } else if (label_data[i] < 0.0) {
        dx_data[i] = 1.0 - pred;
      } else {
        dx_data[i] = label_data[i] - 2.0 * pred;
      }
      if (sum_val >= soft_max_up_bound || sum_val <= soft_max_lower_bound) {
        dx_data[i] = 0;
      }
      dx_data[i] *= dout_data[i] * -1;
    }
  }
};
}  // namespace operators
}  // namespace paddle
