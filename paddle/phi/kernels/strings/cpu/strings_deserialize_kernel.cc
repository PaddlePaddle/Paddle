/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/strings/strings_deserialize_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace strings {

template <typename Context>
void Deserialize(const Context& dev_ctx,
                 const DenseTensor& src,
                 StringTensor* dst) {
  auto* strings_data = reinterpret_cast<const char*>(src.data<uint8_t>());
  auto* strings_offset = reinterpret_cast<const int*>(strings_data);
  int numel = strings_offset[0] / sizeof(int) - 1;
  dst->Resize({numel});
  dtype::pstring* dst_str = dev_ctx.template Alloc<dtype::pstring>(dst);
  for (int i = 0; i < numel; ++i) {
    // -1 not include '\0'
    auto len = strings_offset[i + 1] - strings_offset[i] - 1;
    dst_str[i] = phi::dtype::pstring(strings_data + strings_offset[i], len);
  }
}

}  // namespace strings
}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(strings_deserialize,
                           CPU,
                           ALL_LAYOUT,
                           phi::strings::Deserialize<phi::CPUContext>,
                           pstring) {}
