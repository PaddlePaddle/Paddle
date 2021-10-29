/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"
namespace pten {

using CUDAContext = paddle::platform::CUDADeviceContext;

void Copy(const CUDAContext& dev_ctx, const DenseTensor& src, DenseTensor* dst);

template <typename T = int32_t>
inline std::vector<T> GetDataFromTensor(const CUDAContext& dev_ctx,
                                        const DenseTensor* x) {
  std::vector<T> vec_new_data;
  if (x->data_type() == DataType::INT32) {
    auto* data = x->data<int>();

    if (!paddle::platform::is_cpu_place(x->place())) {
      const auto allocator =
          std::make_shared<paddle::experimental::DefaultAllocator>(
              paddle::platform::CPUPlace());
      DenseTensor cpu_attr_tensor(allocator, x->meta());

      pten::Copy(dev_ctx, *x, &cpu_attr_tensor);
      data = cpu_attr_tensor.data<int>();
    }

    vec_new_data = std::vector<T>(data, data + x->numel());
  } else if (x->data_type() == DataType::INT64) {
    auto* data = x->data<int64_t>();

    if (!paddle::platform::is_cpu_place(x->place())) {
      const auto allocator =
          std::make_shared<paddle::experimental::DefaultAllocator>(
              paddle::platform::CPUPlace());
      DenseTensor cpu_attr_tensor(allocator, x->meta());

      pten::Copy(dev_ctx, *x, &cpu_attr_tensor);
      data = cpu_attr_tensor.data<int64_t>();
    }

    // NOTE: Converting int64 to int32 may cause data overflow.
    vec_new_data = std::vector<T>(data, data + x->numel());
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "The dtype of Tensor must be int32 or int64, but received: %s",
        x->data_type()));
  }
  return vec_new_data;
}

}  // namespace pten
