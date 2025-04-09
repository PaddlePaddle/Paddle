// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/number_count_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {
template <typename T, typename Context>
void NumberCountKernel(const Context& dev_ctx,
                       const DenseTensor& numbers_in,
                       int upper_range,
                       DenseTensor* out) {
  auto numbers = &numbers_in;
  auto number_count = out;
  number_count->Resize({upper_range});
  dev_ctx.template Alloc<T>(number_count);
  phi::DenseTensor cpu_tensor;
  phi::Copy(dev_ctx, *numbers, phi::CPUPlace(), true, &cpu_tensor);
  std::vector<T> count(upper_range);
  for (auto i = 0; i < cpu_tensor.numel(); ++i) {
    auto idx = static_cast<int64_t>(cpu_tensor.data<T>()[i]);
    if (idx >= 0 && idx < upper_range) {
      count[idx] += 1;
    }
  }
  phi::TensorFromVector<T>(count, dev_ctx, number_count);
  number_count->Resize({upper_range});
}
}  // namespace phi

PD_REGISTER_KERNEL(
    number_count, Custom, ALL_LAYOUT, phi::NumberCountKernel, int64_t) {}

#endif
