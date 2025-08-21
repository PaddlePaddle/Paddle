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

#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/range_function.h"

namespace phi {

template <typename T, typename Context>
void ArangeTensorKernel(const Context& dev_ctx,
                        const DenseTensor& start,
                        const DenseTensor& end,
                        const DenseTensor& step,
                        DenseTensor* out) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  using XPUType = typename XPUTypeTrait<T>::Type;
  MPType start_value =
      static_cast<MPType>(GetValue<T, Context>(dev_ctx, start));
  MPType end_value = static_cast<MPType>(GetValue<T, Context>(dev_ctx, end));
  MPType step_value = static_cast<MPType>(GetValue<T, Context>(dev_ctx, step));

  int64_t size = 0;
  phi::funcs::GetSize(start_value, end_value, step_value, &size);
  out->Resize(common::make_ddim({size}));
  XPUType* out_data =
      reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out));

  int ret = xpu::range(dev_ctx.x_context(),
                       out_data,
                       static_cast<XPUType>(start_value),
                       static_cast<XPUType>(step_value),
                       size);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "range");
}

}  // namespace phi

PD_REGISTER_KERNEL(arange_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::ArangeTensorKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}
