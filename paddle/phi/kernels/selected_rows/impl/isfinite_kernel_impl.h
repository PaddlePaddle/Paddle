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

template <typename T, typename Context, typename Functor>
inline void IsfiniteSRImpl(const Context& ctx,
                           const SelectedRows& x,
                           SelectedRows* out);

#define DEFINE_ISFINITE_SR(isfinite_sr, functor)                      \
  template <typename T, typename Context>                             \
  void isfinite_sr(                                                   \
      const Context& ctx, const SelectedRows& x, SelectedRows* out) { \
    IsfiniteSRImpl<T, Context, functor>(ctx, x, out);                 \
  }

DEFINE_ISFINITE_SR(IsinfSR, funcs::InfinityV2Functor)
DEFINE_ISFINITE_SR(IsnanSR, funcs::NANV2Functor)
DEFINE_ISFINITE_SR(IsfiniteSR, funcs::IsfiniteV2Functor)
#undef DEFINE_ISFINITE_SR

}  // namespace phi
