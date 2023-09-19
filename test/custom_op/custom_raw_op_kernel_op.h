// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"

namespace custom_raw_op {

struct ReluFunctor {
  explicit ReluFunctor(const phi::DenseTensor &x, phi::DenseTensor *y)
      : x_(x), y_(y) {}

  template <typename U>
  struct Impl {
    Impl(const U *x, U *y) : x_(x), y_(y) {}

    HOSTDEVICE void operator()(size_t i) const {
      y_[i] = (x_[i] > static_cast<U>(0) ? x_[i] : static_cast<U>(0));
    }

   private:
    const U *x_;
    U *y_;
  };

  template <typename T>
  void apply() {
    auto n = x_.numel();
    auto place = x_.place();
    const auto *x_data = x_.data<T>();

    y_->Resize(x_.dims());
    auto *y_data = y_->mutable_data<T>(place);

    const auto &dev_ctx =
        *paddle::platform::DeviceContextPool::Instance().Get(place);

#define LAUNCH_RELU_KERNEL(DevCtxT)                              \
  do {                                                           \
    auto &__dev_ctx = dynamic_cast<const DevCtxT &>(dev_ctx);    \
    paddle::platform::ForRange<DevCtxT> for_range(__dev_ctx, n); \
    Impl<T> functor(x_data, y_data);                             \
    for_range(functor);                                          \
  } while (0)

#if defined(__NVCC__) || defined(__HIPCC__)
    if (paddle::platform::is_gpu_place(place)) {
      LAUNCH_RELU_KERNEL(phi::GPUContext);
      return;
    }
#endif
    LAUNCH_RELU_KERNEL(phi::CPUContext);

#undef LAUNCH_RELU_KERNEL
  }

 private:
  const phi::DenseTensor &x_;
  phi::DenseTensor *y_;
};

inline void ReluForward(const phi::DenseTensor &x, phi::DenseTensor *y) {
  custom_raw_op::ReluFunctor functor(x, y);
  paddle::framework::VisitDataType(
      paddle::framework::TransToProtoVarType(x.dtype()), functor);
}

}  // namespace custom_raw_op
