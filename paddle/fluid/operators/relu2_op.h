// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct Relu2ForwardFunctor {
 public:
  Relu2ForwardFunctor(const T *x, T *y, uint8_t *mask, int64_t limit)
      : x_(x), y_(y), mask_(mask), limit_(limit) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    uint8_t mask = 0;
    int64_t start_idx = idx * 8;
    int64_t end_idx = start_idx + 8;
    if (end_idx > limit_) end_idx = limit_;

    for (int64_t i = start_idx; i < end_idx; ++i) {
      auto x = x_[i];
      if (x >= static_cast<T>(0)) {
        y_[i] = x;
        mask |= (1 << (i - start_idx));
      } else {
        y_[i] = static_cast<T>(0);
      }
    }
    mask_[idx] = mask;
  }

 private:
  const T *x_;
  T *y_;
  uint8_t *mask_;
  int64_t limit_;
};

template <typename T>
struct Relu2BackwardFunctor {
 public:
  Relu2BackwardFunctor(T *dx, const T *dy, const uint8_t *mask, int64_t limit)
      : dx_(dx), dy_(dy), mask_(mask), limit_(limit) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    uint8_t mask = mask_[idx];
    int64_t start_idx = idx * 8;
    int64_t end_idx = start_idx + 8;
    if (end_idx > limit_) end_idx = limit_;

    for (int64_t i = start_idx; i < end_idx; ++i) {
      if (mask & (1 << (i - start_idx))) {
        dx_[i] = dy_[i];
      } else {
        dx_[i] = static_cast<T>(0);
      }
    }
  }

 private:
  T *dx_;
  const T *dy_;
  const uint8_t *mask_;
  int64_t limit_;
};

template <typename DeviceContext, typename T>
class Relu2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<framework::Tensor>("X");
    auto *mask = ctx.Output<framework::Tensor>("Mask");
    auto *y = ctx.Output<framework::Tensor>("Out");

    auto x_numel = x->numel();
    auto mask_size = (x_numel + 8 - 1) / 8;
    mask->Resize({mask_size});

    auto mask_data = mask->mutable_data<uint8_t>(ctx.GetPlace());
    auto y_data = y->mutable_data<T>(ctx.GetPlace());
    auto x_data = x->data<T>();

    Relu2ForwardFunctor<T> functor(x_data, y_data, mask_data, x_numel);
    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(), mask_size);
    for_range(functor);
  }
};

template <typename DeviceContext, typename T>
class Relu2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *mask = ctx.Input<framework::Tensor>("Mask");
    auto *dy = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto *mask_data = mask->data<uint8_t>();
    auto *dy_data = dy->data<T>();
    auto dx_data = dx->mutable_data<T>(ctx.GetPlace());

    auto limit = dy->numel();
    auto mask_size = mask->numel();

    Relu2BackwardFunctor<T> functor(dx_data, dy_data, mask_data, limit);
    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(), mask_size);
    for_range(functor);
  }
};
}  // namespace operators
}  // namespace paddle
