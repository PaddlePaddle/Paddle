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

#include <cmath>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct CrossEntropyBackwardFunctor {
  CrossEntropyBackwardFunctor(T *dx, const T *y, const T *dy,
                              const int64_t *label, int64_t ignore_index,
                              int64_t feature_size)
      : dx_(dx),
        y_(y),
        dy_(dy),
        label_(label),
        ignore_index_(ignore_index),
        feature_size_(feature_size) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    auto row_idx = idx / feature_size_;
    auto col_idx = idx % feature_size_;
    auto label = label_[row_idx];
    if (label == col_idx && label != ignore_index_) {
      dx_[idx] = -dy_[row_idx] * real_exp(y_[row_idx]);
    } else {
      dx_[idx] = 0;
    }
  }

  T *dx_;
  const T *y_;
  const T *dy_;
  const int64_t *label_;
  int64_t ignore_index_;
  int64_t feature_size_;
};

template <typename DeviceContext, typename T>
class CrossEntropyOpKernel2 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x_original = ctx.Input<Tensor>("X");
    int rank = x_original->dims().size();

    auto x = framework::ReshapeToMatrix(*x_original, rank - 1);
    auto label =
        framework::ReshapeToMatrix(*ctx.Input<Tensor>("Label"), rank - 1);
    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    auto ignore_index = ctx.Attr<int>("ignore_index");

    math::CrossEntropyFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), y, &x, &label, false,
        ignore_index);
  }
};

template <typename DeviceContext, typename T>
class CrossEntropyGradientOpKernel2 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *y = ctx.Input<Tensor>("Y");
    auto *dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto *label = ctx.Input<Tensor>("Label");

    auto *p_dx = dx->mutable_data<T>(ctx.GetPlace());
    auto *p_y = y->data<T>();
    auto *p_dy = dy->data<T>();
    auto *p_label = label->data<int64_t>();

    int64_t ignore_index = ctx.Attr<int>("ignore_index");
    int rank = dx->dims().size();
    int64_t feature_size = dx->dims()[rank - 1];
    int64_t batch_size = framework::product(dx->dims()) / feature_size;

    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(),
        batch_size * feature_size);
    for_range(CrossEntropyBackwardFunctor<T>(p_dx, p_y, p_dy, p_label,
                                             ignore_index, feature_size));
  }
};

}  // namespace operators
}  // namespace paddle
