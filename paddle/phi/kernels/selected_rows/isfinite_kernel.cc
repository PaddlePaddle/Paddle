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

#include "paddle/phi/kernels/selected_rows/isfinite_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/selected_rows/impl/isfinite_kernel_impl.h"

namespace phi {

template <typename T, typename Context, typename Functor>
inline void IsfiniteSRImpl(const Context& dev_ctx,
                           const SelectedRows& x,
                           SelectedRows* out) {
  dev_ctx.template Alloc<T>(out);
  Functor functor;
  functor(x.value(), out->mutable_value());
}
}  // namespace phi

PD_REGISTER_KERNEL(isinf_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::IsinfSR,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(isnan_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::IsnanSR,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(isfinite_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::IsfiniteSR,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(isinf_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::IsinfSR,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(isnan_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::IsnanSR,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(isfinite_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::IsfiniteSR,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {}
#endif
