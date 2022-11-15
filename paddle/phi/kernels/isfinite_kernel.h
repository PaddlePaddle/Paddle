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

#define DEFINE_ISFINITE_KERNEL(isfinite_kernel) \
  template <typename T, typename Context>       \
  void isfinite_kernel(                         \
      const Context& ctx, const DenseTensor& x, DenseTensor* out);

DEFINE_ISFINITE_KERNEL(IsinfKernel)
DEFINE_ISFINITE_KERNEL(IsnanKernel)
DEFINE_ISFINITE_KERNEL(IsfiniteKernel)
#undef DEFINE_ISFINITE_KERNEL

}  // namespace phi
