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

#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context>
void TransposeStrideKernel(const Context& ctx,
                           const DenseTensor& x,
                           const std::vector<int>& axis,
                           DenseTensor* out) {
  auto meta = x.meta();
  auto in_strides = x.strides();
  auto in_dims = x.dims();
  for (size_t i = 0; i < axis.size(); i++) {
    meta.strides[i] = in_strides[axis[i]];
    meta.dims[i] = in_dims[axis[i]];
  }

  out->set_meta(meta);
  out->ResetHolder(x.Holder());
}

}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(transpose,
                           CPU,
                           STRIDED,
                           phi::TransposeStrideKernel<phi::CPUContext>,
                           ALL_DTYPE) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(transpose,
                           GPU,
                           STRIDED,
                           phi::TransposeStrideKernel<phi::GPUContext>,
                           ALL_DTYPE) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_GENERAL_KERNEL(transpose,
                           XPU,
                           STRIDED,
                           phi::TransposeStrideKernel<phi::XPUContext>,
                           ALL_DTYPE) {}
#endif
