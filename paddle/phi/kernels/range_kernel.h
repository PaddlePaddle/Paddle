
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

#include "paddle/phi/core/dense_tensor.h"

#include "paddle/fluid/framework/tensor_util.h"

namespace phi {

template <typename T>
void GetSize(T start, T end, T step, int64_t* size) {
  PADDLE_ENFORCE_NE(
      step,
      0,
      phi::errors::InvalidArgument("The step of range op should not be 0."));

  if (start < end) {
    PADDLE_ENFORCE_GT(
        step,
        0,
        phi::errors::InvalidArgument(
            "The step should be greater than 0 while start < end."));
  }

  if (start > end) {
    PADDLE_ENFORCE_LT(step,
                      0,
                      phi::errors::InvalidArgument(
                          "The step should be less than 0 while start > end."));
  }

  *size = std::is_integral<T>::value
              ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
              : std::ceil(std::abs((end - start) / step));
}

template <typename T>
inline T GetValue(const DenseTensor& x) {
  T value = static_cast<T>(0);
  if (!paddle::platform::is_cpu_place(x.place())) {
    DenseTensor cpu_x;
    paddle::framework::TensorCopy(x, phi::CPUPlace(), &cpu_x);
#ifdef PADDLE_WITH_ASCEND_CL
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    const phi::DeviceContext* dev_ctx = pool.Get(x.place());
    dev_ctx->Wait();
#endif
    value = cpu_x.data<T>()[0];
  } else {
    value = x.data<T>()[0];
  }
  return value;
}

template <typename T, typename Context>
void RangeKernel(const Context& dev_ctx,
                 const DenseTensor& start,
                 const DenseTensor& end,
                 const DenseTensor& step,
                 DenseTensor* out);
}  // namespace phi
