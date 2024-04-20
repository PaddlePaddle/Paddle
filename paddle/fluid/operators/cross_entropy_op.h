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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CrossEntropyOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* labels = ctx.Input<phi::DenseTensor>("Label");
    auto* y = ctx.Output<phi::DenseTensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    int rank = x->dims().size();
    auto label_dims = labels->dims();
    phi::DenseTensor x_2d = phi::ReshapeToMatrix(*x, rank - 1);
    phi::DenseTensor labels_2d, y_2d;
    if (label_dims.size() < rank) {
      labels_2d.ShareDataWith(*labels);
      labels_2d.Resize({common::product(label_dims), 1});

      y_2d.ShareDataWith(*y);
      y_2d.Resize({common::product(y->dims()), 1});

    } else {
      labels_2d = phi::ReshapeToMatrix(*labels, rank - 1);
      y_2d = phi::ReshapeToMatrix(*y, rank - 1);
    }

    int axis_dim = x->dims()[rank - 1];
    phi::funcs::CrossEntropyFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(),
        &y_2d,
        &x_2d,
        &labels_2d,
        ctx.Attr<bool>("soft_label"),
        ctx.Attr<int>("ignore_index"),
        axis_dim);
  }
};

template <typename T>
class XeSoftLabelGradFunctor {
 public:
  XeSoftLabelGradFunctor(T* dx,
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
                size_t num_classes,
                size_t ignore_index)
      : dx_(dx),
        dy_(dy),
        x_(x),
        label_(label),
        num_classes_(num_classes),
        ignore_index_(ignore_index) {}

  HOSTDEVICE void operator()(size_t sample_id) {
    auto x_is_true_offset = sample_id * num_classes_ + label_[sample_id];
    for (size_t x_offset = sample_id * num_classes_;
         x_offset < (sample_id + 1) * num_classes_;
         ++x_offset) {
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

template <typename T, typename DeviceContext>
class CrossEntropyGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* dy = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    auto* label = ctx.Input<phi::DenseTensor>("Label");
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(ctx.GetPlace());

    // Following computation only depends on the last dimension size. So it's
    // unnecessary to convert tensors to 2-D views.
    int rank = x->dims().size();
    int64_t class_num = x->dims()[rank - 1];
    int64_t ignore_index = ctx.Attr<int>("ignore_index");
    if (ctx.Attr<bool>("soft_label")) {
      XeSoftLabelGradFunctor<T> functor(dx_data,
                                        dy->data<T>(),
                                        x->data<T>(),
                                        label->data<T>(),
                                        static_cast<size_t>(class_num));
      platform::ForRange<DeviceContext> for_range(
          ctx.template device_context<DeviceContext>(),
          static_cast<size_t>(dx->numel()));
      for_range(functor);
    } else {
      XeGradFunctor<T> functor(dx_data,
                               dy->data<T>(),
                               x->data<T>(),
                               label->data<int64_t>(),
                               static_cast<size_t>(class_num),
                               static_cast<size_t>(ignore_index));
      platform::ForRange<DeviceContext> for_range(
          ctx.template device_context<DeviceContext>(),
          static_cast<size_t>(dy->numel()));
      for_range(functor);
    }
  }
};

template <typename T>
struct HardLabelCrossEntropyForwardFunctor {
  HardLabelCrossEntropyForwardFunctor(const T* x,
                                      T* y,
                                      T* match_x,
                                      const int64_t* label,
                                      int64_t ignore_index,
                                      int64_t feature_size)
      : x_(x),
        y_(y),
        match_x_(match_x),
        label_(label),
        ignore_index_(ignore_index),
        feature_size_(feature_size) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    auto label = label_[idx];
    if (label != ignore_index_) {
      // don't update to PADDLE_ENFORCE_GE and PADDLE_ENFORCE_LT cause
      // can't use phi::errors::InvalidArgument in HOSTDEVICE
      PADDLE_ENFORCE(label >= 0 && label < feature_size_,
                     "Variable value (label) of "
                     "OP(fluid.layers.cross_entropy) expected >= 0 "
                     "and < %ld, but got %ld. Please check label value.",
                     feature_size_,
                     label);

      auto match_x = x_[idx * feature_size_ + label];
      y_[idx] = -phi::funcs::TolerableValue<T>()(phi::funcs::real_log(match_x));
      match_x_[idx] = match_x;
    } else {
      y_[idx] = 0;
      match_x_[idx] = 0;  // any value is ok
    }
  }

  const T* x_;
  T* y_;
  T* match_x_;
  const int64_t* label_;
  int64_t ignore_index_;
  int64_t feature_size_;
};

template <typename T>
struct HardLabelCrossEntropyBackwardFunctor {
  HardLabelCrossEntropyBackwardFunctor(T* dx,
                                       const T* dy,
                                       const T* match_x,
                                       const int64_t* label,
                                       int64_t ignore_index,
                                       int64_t feature_size)
      : dx_(dx),
        dy_(dy),
        match_x_(match_x),
        label_(label),
        ignore_index_(ignore_index),
        feature_size_(feature_size) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    auto row_idx = idx / feature_size_;
    auto col_idx = idx % feature_size_;
    auto label = label_[row_idx];
    if (label == col_idx && label != ignore_index_) {
      dx_[idx] = -dy_[row_idx] / match_x_[row_idx];
    } else {
      dx_[idx] = 0;
    }
  }

  T* dx_;
  const T* dy_;
  const T* match_x_;
  const int64_t* label_;
  int64_t ignore_index_;
  int64_t feature_size_;
};

template <typename T, typename DeviceContext>
class CrossEntropyOpKernel2 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* label = ctx.Input<phi::DenseTensor>("Label");
    auto* y = ctx.Output<phi::DenseTensor>("Y");
    auto* match_x = ctx.Output<phi::DenseTensor>("MatchX");

    auto& x_dims = x->dims();
    auto feature_size = x_dims[x_dims.size() - 1];
    auto batch_size = common::product(x->dims()) / feature_size;

    auto* p_x = x->data<T>();
    auto* p_label = label->data<int64_t>();
    auto* p_y = y->mutable_data<T>(ctx.GetPlace());
    auto* p_match_x = match_x->mutable_data<T>(ctx.GetPlace());

    auto ignore_index = ctx.Attr<int>("ignore_index");

    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(), batch_size);
    for_range(HardLabelCrossEntropyForwardFunctor<T>(
        p_x, p_y, p_match_x, p_label, ignore_index, feature_size));
  }
};

template <typename T, typename DeviceContext>
class CrossEntropyGradientOpKernel2 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    auto* match_x = ctx.Input<phi::DenseTensor>("MatchX");
    auto* label = ctx.Input<phi::DenseTensor>("Label");

    auto* p_dx = dx->mutable_data<T>(ctx.GetPlace());
    auto* p_dy = dy->data<T>();
    auto* p_match_x = match_x->data<T>();
    auto* p_label = label->data<int64_t>();

    int64_t ignore_index = ctx.Attr<int>("ignore_index");
    int rank = dx->dims().size();
    int64_t feature_size = dx->dims()[rank - 1];
    int64_t batch_size = common::product(dx->dims()) / feature_size;

    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(),
        batch_size * feature_size);
    for_range(HardLabelCrossEntropyBackwardFunctor<T>(
        p_dx, p_dy, p_match_x, p_label, ignore_index, feature_size));
  }
};

}  // namespace operators
}  // namespace paddle
