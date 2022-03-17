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

#include "paddle/phi/kernels/selected_rows/copy_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
namespace phi {
namespace sr {

template <typename Context>
void Copy(const Context& dev_ctx,
          const SelectedRows& src,
          Place dst_place,
          bool blocking,
          SelectedRows* dst) {
  if (src.value().Holder() != dst->value().Holder() ||
      src.value().data() != dst->value().data()) {
    dst->set_rows(src.rows());
    dst->set_height(src.height());
  }
  phi::Copy<Context>(
      dev_ctx, src.value(), dst_place, blocking, dst->mutable_value());
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(
    copy_sr, CPU, ALL_LAYOUT, phi::sr::Copy<phi::CPUContext>, ALL_DTYPE) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(
    copy_sr, GPU, ALL_LAYOUT, phi::sr::Copy<phi::GPUContext>, ALL_DTYPE) {}
#endif
