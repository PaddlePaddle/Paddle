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
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CrossEntropyOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    int rank = x->dims().size();
    Tensor x_2d = framework::ReshapeToMatrix(*x, rank - 1);
    Tensor labels_2d = framework::ReshapeToMatrix(*labels, rank - 1);
    Tensor y_2d = framework::ReshapeToMatrix(*y, rank - 1);

    math::CrossEntropyFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), &y_2d, &x_2d, &labels_2d,
        ctx.Attr<bool>("soft_label"), ctx.Attr<int>("ignore_index"));
  }
};

template <typename T>
class XeSoftlabelGradFunctor {
 public:
  XeSoftlabelGradFunctor(T* dx,
                         const T* dy,     // NOLINT
                         const T* x,      // NOLINT
                         const T* label,  // NOLINT
                         size_t num_classes)
      : dx_(dx), dy_(dy), x_(x), label_(label), num_classes_(num_classes) {}

  HOSTDEVICE void operator()(size_t i) {
    auto row_ids = i / num_classes_;
    dx_[i] = -label_[i] * dy_[row_ids] / x_[i];
  }

 private:
  T* dx_;
  const T* dy_;
  const T* x_;
  const T* label_;
  size_t num_classes_;
};

template <typename T>
class XeGradFunctor {
 public:
  XeGradFunctor(T* dx,
                const T* dy,           // NOLINT
                const T* x,            // NOLINT
                const int64_t* label,  // NOLINT
                size_t num_classes, size_t ignore_index)
      : dx_(dx),
        dy_(dy),
        x_(x),
        label_(label),
        num_classes_(num_classes),
        ignore_index_(ignore_index) {}

  HOSTDEVICE void operator()(size_t sample_id) {
    auto x_is_true_offset = sample_id * num_classes_ + label_[sample_id];
    for (size_t x_offset = sample_id * num_classes_;
         x_offset < (sample_id + 1) * num_classes_; ++x_offset) {
      dx_[x_offset] = (x_offset != x_is_true_offset ||
                       label_[sample_id] == static_cast<int64_t>(ignore_index_))
                          ? static_cast<T>(0)
                          : -dy_[sample_id] / x_[x_offset];
    }
  }

 private:
  T* dx_;
  const T* dy_;
  const T* x_;
  const int64_t* label_;
  size_t num_classes_;
  size_t ignore_index_;
};

template <typename DeviceContext, typename T>
class CrossEntropyGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* label = ctx.Input<Tensor>("Label");
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(ctx.GetPlace());

    // Following computation only depends on the last dimension size. So it's
    // unnecessary to convert tensors to 2-D views.
    int rank = x->dims().size();
    int64_t class_num = x->dims()[rank - 1];
    int64_t ignore_index = ctx.Attr<int>("ignore_index");
    if (ctx.Attr<bool>("soft_label")) {
      XeSoftlabelGradFunctor<T> functor(dx_data, dy->data<T>(), x->data<T>(),
                                        label->data<T>(),
                                        static_cast<size_t>(class_num));
      platform::ForRange<DeviceContext> for_range(
          ctx.template device_context<DeviceContext>(),
          static_cast<size_t>(dx->numel()));
      for_range(functor);
    } else {
      XeGradFunctor<T> functor(
          dx_data, dy->data<T>(), x->data<T>(), label->data<int64_t>(),
          static_cast<size_t>(class_num), static_cast<size_t>(ignore_index));
      platform::ForRange<DeviceContext> for_range(
          ctx.template device_context<DeviceContext>(),
          static_cast<size_t>(dy->numel()));
      for_range(functor);
    }
  }
};

template <typename T>
struct HardLabelCrossEntropyBackwardFunctor {
  HardLabelCrossEntropyBackwardFunctor(T* dx, const T* y, const T* dy,
                                       const int64_t* label,
                                       int64_t ignore_index,
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

  T* dx_;
  const T* y_;
  const T* dy_;
  const int64_t* label_;
  int64_t ignore_index_;
  int64_t feature_size_;
};

template <typename DeviceContext, typename T>
class CrossEntropyOpKernel2 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x_original = ctx.Input<Tensor>("X");
    int rank = x_original->dims().size();

    auto x = framework::ReshapeToMatrix(*x_original, rank - 1);
    auto label =
        framework::ReshapeToMatrix(*ctx.Input<Tensor>("Label"), rank - 1);
    auto* y = ctx.Output<Tensor>("Y");
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
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* y = ctx.Input<Tensor>("Y");
    auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* label = ctx.Input<Tensor>("Label");

    auto* p_dx = dx->mutable_data<T>(ctx.GetPlace());
    auto* p_y = y->data<T>();
    auto* p_dy = dy->data<T>();
    auto* p_label = label->data<int64_t>();

    int64_t ignore_index = ctx.Attr<int>("ignore_index");
    int rank = dx->dims().size();
    int64_t feature_size = dx->dims()[rank - 1];
    int64_t batch_size = framework::product(dx->dims()) / feature_size;

    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(),
        batch_size * feature_size);
    for_range(HardLabelCrossEntropyBackwardFunctor<T>(
        p_dx, p_y, p_dy, p_label, ignore_index, feature_size));
  }
};

}  // namespace operators
}  // namespace paddle
