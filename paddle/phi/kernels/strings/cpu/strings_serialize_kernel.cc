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

#include "paddle/phi/kernels/strings/strings_serialize_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace strings {

template <typename Context>
void Serialize(const Context& dev_ctx,
               const StringTensor& src,
               DenseTensor* dst) {
  int64_t numel = src.numel();
  int64_t num = sizeof(int) * (numel + 1);
  auto* src_str = src.data();
  for (int64_t i = 0; i < numel; ++i) {
    num += src_str[i].length() + 1;
  }
  dst->Resize({num});
  uint8_t* strings_data = dev_ctx.template Alloc<uint8_t>(dst);
  auto* strings_offset = reinterpret_cast<int*>(strings_data);
  int start_offset = sizeof(int) * (numel + 1);
  for (int64_t i = 0; i <= numel; ++i) {
    if (i == 0) {
      strings_offset[i] = start_offset;
    } else {
      strings_offset[i] = strings_offset[i - 1] + src_str[i - 1].length() + 1;
    }
  }
  for (int64_t i = 0; i < numel; ++i) {
    memcpy(strings_data + strings_offset[i],
           src_str[i].data(),
           src_str[i].length() + 1);
  }
}

}  // namespace strings
}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(strings_serialize,
                           CPU,
                           ALL_LAYOUT,
                           phi::strings::Serialize<phi::CPUContext>,
                           pstring) {}
