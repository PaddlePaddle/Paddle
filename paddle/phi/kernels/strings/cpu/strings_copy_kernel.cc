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

#include "paddle/phi/kernels/strings/strings_copy_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace strings {

template <typename Context>
void Copy(const Context& dev_ctx,
          const StringTensor& src,
          bool blocking,
          StringTensor* dst) {
  auto* src_ptr = src.data();
  const auto& src_place = src.place();

  VLOG(3) << "StringTensorCopy " << src.dims() << " from " << src.place()
          << " to " << src_place;

  dst->Resize(src.dims());
  dtype::pstring* dst_ptr = dev_ctx.template Alloc<dtype::pstring>(dst);

  if (src_ptr == dst_ptr) {
    VLOG(3) << "Skip copy the same string data async from " << src_place
            << " to " << src_place;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;
  int64_t numel = src.numel();

  if (src_place.GetType() == phi::AllocationType::CPU) {
    for (int64_t i = 0; i < numel; ++i) {
      dst_ptr[i] = src_ptr[i];
    }
  }
}

}  // namespace strings
}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(strings_copy,
                           CPU,
                           ALL_LAYOUT,
                           phi::strings::Copy<phi::CPUContext>,
                           pstring) {}
