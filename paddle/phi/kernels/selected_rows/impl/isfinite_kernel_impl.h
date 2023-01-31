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

#include "paddle/phi/kernels/funcs/isfinite_functor.h"
#include "paddle/phi/kernels/selected_rows/isfinite_kernel.h"

namespace phi {

#define DEFINE_ISFINITE_SR(isfinite)                                  \
  template <typename T, typename Context>                             \
  void isfinite##SR(                                                  \
      const Context& ctx, const SelectedRows& x, SelectedRows* out) { \
    ctx.template Alloc<bool>(out);                                    \
    Isinf##Kernel<T, Context>(ctx, x.value(), out->mutable_value());  \
  }

DEFINE_ISFINITE_SR(Isinf)
DEFINE_ISFINITE_SR(Isnan)
DEFINE_ISFINITE_SR(Isfinite)
#undef DEFINE_ISFINITE_SR

}  // namespace phi
