// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU

#include <vector>
#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

DECLARE_bool(limited_idle_chunk);
namespace phi {
template <typename T>
T* Alloc_l3(xpu::ctx_guard* RAII_GUARD, const int64_t n) {
  VLOG(4) << "Alloc XPU L3 with size = " << n * sizeof(T) << " and len = " << n;
  T* ret = RAII_GUARD->alloc_l3<T>(n);
  return ret;
}

template <typename T, typename Context>
T* Alloc_gm(const Context& dev_ctx, const int64_t n) {
  DenseTensor ret_tensor;
  VLOG(4) << "Alloc XPU GM with size = " << n * sizeof(T) << " and len = " << n;
  DDim d({n});
  ret_tensor.Resize(d);
  T* ret = dev_ctx.template Alloc<T>(&ret_tensor);
  return ret;
}

template <typename Context, typename T, typename Tx>
Tx* Alloc_l3_or_gm(const Context& dev_ctx,
                   const int64_t n,
                   xpu::ctx_guard* RAII_GUARD) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (std::is_same<Tx, XPUType>::value) {
    Tx* ret = Alloc_l3<Tx>(RAII_GUARD, n);
    if (ret != nullptr) {
      return ret;
    }
    ret = reinterpret_cast<XPUType*>(Alloc_gm<T, Context>(dev_ctx, n));
    return ret;
  } else {
    PADDLE_ENFORCE_EQ(0,
                      1,
                      errors::PreconditionNotMet(
                          "XPU Alloc_l3_or_gm retrun ",
                          "type should be XPUType, and be same as casted T"));
  }
}

}  // namespace phi
#endif
