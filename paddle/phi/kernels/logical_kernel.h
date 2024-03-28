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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

#define DECLARE_LOGICAL_BINARY_KERNEL(type)          \
  template <typename T, typename Context>            \
  void Logical##type##Kernel(const Context& dev_ctx, \
                             const DenseTensor& x,   \
                             const DenseTensor& y,   \
                             DenseTensor* out);

DECLARE_LOGICAL_BINARY_KERNEL(And)
DECLARE_LOGICAL_BINARY_KERNEL(Or)
DECLARE_LOGICAL_BINARY_KERNEL(Xor)
#undef DECLARE_LOGICAL_BINARY_KERNEL

template <typename T, typename Context>
void LogicalNotKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      DenseTensor* out);

}  // namespace phi
