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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct TolerableValue {
  HOSTDEVICE T operator()(const T& x) const {
    PADDLE_ASSERT(std::is_floating_point<T>::value);
    const T kApproInf = 1e20;
    if (x == INFINITY) return kApproInf;
    if (x == -INFINITY) return -kApproInf;
    return x;
  }
};

template <typename DeviceContext, typename T>
class BprLossOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* labels_Pos = ctx.Input<Tensor>("Label_Pos");
    auto* y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());
    int rank = x->dims().size();

    Tensor x_2d = framework::ReshapeToMatrix(*x, rank - 1);
    Tensor labels_Pos_2d = framework::ReshapeToMatrix(*labels_Pos, rank - 1);
    Tensor y_2d = framework::ReshapeToMatrix(*y, rank - 1);

    const framework::Tensor* prob = &x_2d;
    const framework::Tensor* labels_pos = &labels_Pos_2d;
    framework::Tensor* out = &y_2d;

    const int step_size = prob->dims()[0];
    const int class_num = prob->dims()[1];
    const T* prob_data = prob->data<T>();
    T* loss_data = out->data<T>();

    const int64_t* label_pos_data = labels_pos->data<int64_t>();
    for (int i = 0; i < step_size; ++i) {
      int lbl_pos = label_pos_data[i];
      PADDLE_ENFORCE_GE(lbl_pos, 0);
      PADDLE_ENFORCE_LT(lbl_pos, class_num);
      int index_pos = i * class_num + lbl_pos;
      T sum = static_cast<T>(0);
      for (int j = 0; j < class_num; j++) {
        if (j == lbl_pos) continue;
        int index_neg = i * class_num + j;
        sum += TolerableValue<T>()(-std::log(
            1.0f + TolerableValue<T>()(
                       std::exp(prob_data[index_neg] - prob_data[index_pos]))));
      }
      loss_data[i] = -sum / (class_num - 1);
    }
  }
};

template <typename T>
class XeGradFunctor {
 public:
  XeGradFunctor(T* dx,
                const T* dy,               // NOLINT
                const T* x,                // NOLINT
                const int64_t* label_pos,  // NOLINT
                size_t num_classes)
      : dx_(dx),
        dy_(dy),
        x_(x),
        label_pos_(label_pos),
        num_classes_(num_classes) {}

  HOSTDEVICE void operator()(size_t sample_id) {
    for (size_t x_offset = sample_id * num_classes_;
         x_offset < (sample_id + 1) * num_classes_; ++x_offset) {
      dx_[x_offset] = static_cast<T>(0);
    }
    auto p_index = sample_id * num_classes_ + label_pos_[sample_id];
    for (size_t ni = 0; ni < num_classes_; ni++) {
      if (label_pos_[sample_id] == ni) continue;
      auto n_index = sample_id * num_classes_ + ni;
      auto grad_ =
          -dy_[sample_id] /
          ((num_classes_ - 1) *
           (1.0f + TolerableValue<T>()(std::exp(x_[p_index] - x_[n_index]))));
      dx_[p_index] += grad_;
      dx_[n_index] -= grad_;
    }
  }

 private:
  T* dx_;
  const T* dy_;
  const T* x_;
  const int64_t* label_pos_;
  size_t num_classes_;
};

template <typename DeviceContext, typename T>
class BprLossGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* label_pos = ctx.Input<Tensor>("Label_Pos");
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(ctx.GetPlace());

    int rank = x->dims().size();
    int64_t class_num = x->dims()[rank - 1];
    XeGradFunctor<T> functor(dx_data, dy->data<T>(), x->data<T>(),
                             label_pos->data<int64_t>(),
                             static_cast<size_t>(class_num));
    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(),
        static_cast<size_t>(dy->numel()));
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle
