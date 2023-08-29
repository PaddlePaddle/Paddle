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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"

namespace phi {

//////// Sum Functor ///////
struct SumFunctor {
  template <typename DeviceContext, typename X, typename Y>
  void operator()(const DeviceContext& ctx,
                  const X* x,
                  Y* y,
                  const std::vector<int>& xdims,
                  const std::vector<int>& reduce_dims) {
    using XPUType = typename XPUTypeTrait<X>::Type;
#ifndef PADDLE_WITH_XPU_PLUGIN
    int r = xpu::reduce_sum<XPUType>(ctx,
                                     reinterpret_cast<const XPUType*>(x),
                                     reinterpret_cast<XPUType*>(y),
                                     xdims,
                                     reduce_dims);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
#else
    int r = xpu::plugin::fast_reduce_sum<XPUType>(
        ctx,
        reinterpret_cast<const XPUType*>(x),
        reinterpret_cast<XPUType*>(y),
        xdims,
        reduce_dims);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "fast_reduce_sum");
#endif
  }
};
}  // namespace phi
