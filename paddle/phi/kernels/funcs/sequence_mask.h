/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {
namespace funcs {

template <typename Tx, typename Ty>
struct SequenceMaskForRangeFunctor {
  HOSTDEVICE SequenceMaskForRangeFunctor(const Tx *x, Ty *y, int maxlen)
      : x_(x), y_(y), maxlen_(maxlen) {}

  HOSTDEVICE void operator()(int y_idx) const {
    int x_idx = y_idx / maxlen_;
    int j = y_idx % maxlen_;
    y_[y_idx] = static_cast<Ty>(j < x_[x_idx] ? 1 : 0);
  }

 private:
  const Tx *x_;
  Ty *y_;
  int maxlen_;
};

template <typename DeviceContext, typename Tx>
struct SequenceMaskFunctor {
  SequenceMaskFunctor(const DeviceContext &ctx,
                      const Tx *x,
                      phi::DenseTensor *y,
                      int limits,
                      int maxlen)
      : ctx_(ctx), x_(x), y_(y), limits_(limits), maxlen_(maxlen) {}
  template <typename Ty>
  void apply() const {
    ctx_.template Alloc<Ty>(y_);
    auto *y_data = y_->data<Ty>();
    phi::funcs::ForRange<DeviceContext> for_range(ctx_, limits_);
    for_range(SequenceMaskForRangeFunctor<Tx, Ty>(x_, y_data, maxlen_));
  }

 private:
  const DeviceContext &ctx_;
  const Tx *x_;
  phi::DenseTensor *y_;
  int limits_;
  int maxlen_;
};

}  // namespace funcs
}  // namespace phi
