/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#define DECALRE_COMPARE_KERNEL(compare_kernel) \
  template <typename T, typename Context>      \
  void compare_kernel(const Context& ctx,      \
                      const DenseTensor& x,    \
                      const DenseTensor& y,    \
                      int axis,                \
                      DenseTensor* out);

DECALRE_COMPARE_KERNEL(LessThanKernel)
DECALRE_COMPARE_KERNEL(LessEqualKernel)
DECALRE_COMPARE_KERNEL(GreaterThanKernel)
DECALRE_COMPARE_KERNEL(GreaterEqualKernel)
DECALRE_COMPARE_KERNEL(EqualKernel)
DECALRE_COMPARE_KERNEL(NotEqualKernel)
#undef DECALRE_COMPARE_KERNEL

#define DECALRE_COMPARE_ALL_KERNEL(compare_all_kernel) \
  template <typename T, typename Context>              \
  void compare_all_kernel(const Context& ctx,          \
                          const DenseTensor& x,        \
                          const DenseTensor& y,        \
                          DenseTensor* out);

DECALRE_COMPARE_ALL_KERNEL(EqualAll)
#undef DECALRE_COMPARE_KERNEL

}  // namespace phi
