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

#include "paddle/phi/kernels/strings/strings_empty_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi::strings {

template <typename Context>
void EmptyKernel(const Context& dev_ctx,
                 const IntArray& shape,
                 StringTensor* out) {
  out->Resize(common::make_ddim(shape.GetData()));
  dev_ctx.template Alloc<dtype::pstring>(out);
}

template <typename Context>
void EmptyLikeKernel(const Context& dev_ctx, StringTensor* out) {
  dev_ctx.template Alloc<dtype::pstring>(out);
}

}  // namespace phi::strings

using pstring = ::phi::dtype::pstring;

PD_REGISTER_KERNEL_FOR_ALL_DTYPE(strings_empty,
                                 CPU,
                                 ALL_LAYOUT,
                                 phi::strings::EmptyKernel<phi::CPUContext>) {}

PD_REGISTER_KERNEL_FOR_ALL_DTYPE(
    strings_empty_like,
    CPU,
    ALL_LAYOUT,
    phi::strings::EmptyLikeKernel<phi::CPUContext>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL_FOR_ALL_DTYPE(strings_empty,
                                 GPU,
                                 ALL_LAYOUT,
                                 phi::strings::EmptyKernel<phi::GPUContext>) {}

PD_REGISTER_KERNEL_FOR_ALL_DTYPE(
    strings_empty_like,
    GPU,
    ALL_LAYOUT,
    phi::strings::EmptyLikeKernel<phi::GPUContext>) {}
#endif
