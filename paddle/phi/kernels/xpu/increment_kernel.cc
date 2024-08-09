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

#include "paddle/phi/kernels/increment_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void IncrementKernel(const Context& ctx,
                     const DenseTensor& x,
                     float value,
                     DenseTensor* out) {
  // check input
  PADDLE_ENFORCE_EQ(x.numel(),
                    1,
                    common::errors::InvalidArgument(
                        "input tensor x's numel should be EXACTLY 1."));

  const T* x_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);

  // allocation for "value" on xpu
  T value_as_t = static_cast<T>(value);
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  T* value_xpu = RAII_GUARD.alloc_l3_or_gm<T>(1);
  memory_utils::Copy(ctx.GetPlace(),
                     value_xpu,
                     phi::CPUPlace(),
                     reinterpret_cast<void*>(&value_as_t),
                     sizeof(T));

  // int add(Context* ctx, const T* x, const T* y, T* z, int64_t len);
  int ret = xpu::add(ctx.x_context(), x_data, value_xpu, out_data, 1);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "add");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    increment, XPU, ALL_LAYOUT, phi::IncrementKernel, float, int, int64_t) {}
