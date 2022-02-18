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
#include "paddle/fluid/platform/for_range.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
/*Todo:
 *Find a way to adapt TolerableValue, using blas or eigen.
 */
template <typename T>
struct TolerableValue {
  HOSTDEVICE T operator()(const T& x) const {
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
    auto* label = ctx.Input<Tensor>("Label");
    auto* y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());
    int rank = x->dims().size();

    Tensor x_2d = framework::ReshapeToMatrix(*x, rank - 1);
    Tensor labels_2d = framework::ReshapeToMatrix(*label, rank - 1);
    Tensor y_2d = framework::ReshapeToMatrix(*y, rank - 1);

    const framework::Tensor* logits = &x_2d;
    const framework::Tensor* labels = &labels_2d;
    framework::Tensor* out = &y_2d;

    const int step_size = logits->dims()[0];
    const int class_num = logits->dims()[1];
    const T* logits_data = logits->data<T>();
    T* loss_data = out->data<T>();

    const int64_t* label_data = labels->data<int64_t>();
    for (int i = 0; i < step_size; ++i) {
      int lbl_pos = label_data[i];
      PADDLE_ENFORCE_GE(lbl_pos, 0, platform::errors::InvalidArgument(
                                        "label data %d is illegal.", lbl_pos));
      PADDLE_ENFORCE_LT(lbl_pos, class_num,
                        platform::errors::InvalidArgument(
                            "label data %d is illegal.", lbl_pos));
      int index_pos = i * class_num + lbl_pos;
      T sum = static_cast<T>(0);
      for (int j = 0; j < class_num; j++) {
        if (j == lbl_pos) continue;
        int index_neg = i * class_num + j;
        sum += TolerableValue<T>()(-std::log(
            1.0f + TolerableValue<T>()(std::exp(logits_data[index_neg] -
                                                logits_data[index_pos]))));
      }
      loss_data[i] = -sum / (class_num - 1);
    }
  }
};

template <typename DeviceContext, typename T>
class BprLossGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* label = ctx.Input<Tensor>("Label");
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    const size_t step_size = static_cast<size_t>(x->dims()[0]);
    const size_t num_classes = static_cast<size_t>(x->dims()[1]);
    T* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    const T* dy_data = dy->data<T>();
    const T* x_data = x->data<T>();
    const int64_t* label_data = label->data<int64_t>();

    for (size_t sample_id = 0; sample_id < step_size; sample_id++) {
      for (size_t x_offset = sample_id * num_classes;
           x_offset < (sample_id + 1) * num_classes; x_offset++) {
        dx_data[x_offset] = static_cast<T>(0);
      }
      auto p_index = sample_id * num_classes + label_data[sample_id];
      for (size_t ni = 0; ni < num_classes; ni++) {
        if (label_data[sample_id] == static_cast<int>(ni)) continue;
        auto n_index = sample_id * num_classes + ni;
        auto grad_ = -dy_data[sample_id] /
                     ((num_classes - 1) *
                      (1.0f + TolerableValue<T>()(std::exp(x_data[p_index] -
                                                           x_data[n_index]))));
        dx_data[p_index] += grad_;
        dx_data[n_index] -= grad_;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
