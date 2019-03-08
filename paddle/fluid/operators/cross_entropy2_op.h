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
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

HOSTDEVICE inline platform::float16 RealLog(platform::float16 x) {
#ifdef __NVCC__
  return static_cast<platform::float16>(logf(static_cast<float>(x)));
#else
  return static_cast<platform::float16>(std::log(static_cast<float>(x)));
#endif
}

HOSTDEVICE inline float RealLog(float x) {
#ifdef __NVCC__
  return logf(x);
#else
  return std::log(x);
#endif
}

HOSTDEVICE inline double RealLog(double x) {
#ifdef __NVCC__
  return log(x);
#else
  return std::log(x);
#endif
}

HOSTDEVICE inline platform::float16 RealExp(platform::float16 x) {
#ifdef __NVCC__
  return static_cast<platform::float16>(expf(static_cast<float>(x)));
#else
  return static_cast<platform::float16>(std::exp(static_cast<float>(x)));
#endif
}

HOSTDEVICE inline float RealExp(float x) {
#ifdef __NVCC__
  return expf(x);
#else
  return std::exp(x);
#endif
}

HOSTDEVICE inline double RealExp(double x) {
#ifdef __NVCC__
  return exp(x);
#else
  return std::exp(x);
#endif
}

template <typename T>
struct CrossEntropyForwardFunctor {
  CrossEntropyForwardFunctor(const T *x, T *y, const int64_t *label,
                             int64_t ignore_index, int64_t feature_size)
      : x_(x),
        y_(y),
        label_(label),
        ignore_index_(ignore_index),
        feature_size_(feature_size) {}

  HOSTDEVICE void operator()(int64_t row_idx) const {
    auto col_idx = label_[row_idx];
    if (col_idx != ignore_index_) {
      y_[row_idx] = -math::TolerableValue<T>()(
          RealLog(x_[row_idx * feature_size_ + col_idx]));
    } else {
      y_[row_idx] = 0;
    }
  }

  const T *x_;
  T *y_;
  const int64_t *label_;
  int64_t ignore_index_;
  int64_t feature_size_;
};

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
      dx_[idx] = -dy_[row_idx] * RealExp(y_[row_idx]);
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
    auto *x = ctx.Input<Tensor>("X");
    auto *label = ctx.Input<Tensor>("Label");
    auto *y = ctx.Output<Tensor>("Y");

    auto *p_y = y->mutable_data<T>(ctx.GetPlace());
    auto *p_x = x->data<T>();
    auto *p_label = label->data<int64_t>();

    int rank = x->dims().size();
    int64_t feature_size = x->dims()[rank - 1];
    int64_t batch_size = framework::product(x->dims()) / feature_size;

    int64_t ignore_index = ctx.Attr<int>("ignore_index");

    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(), batch_size);
    for_range(CrossEntropyForwardFunctor<T>(p_x, p_y, p_label, ignore_index,
                                            feature_size));
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
