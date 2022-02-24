// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <math.h>

#include "paddle/phi/kernels/vocab_kernel.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FindKernel(const Context& dev_ctx,
                const VocabTensor& vocab_tensor,
                const DenseTensor& x,
                DenseTensor* out) {
  auto input_data = x.data<T>();
  auto vocab = vocab_tensor.data();

  auto size = x.numel();
  DDim out_dim{size};
  out->Resize(out_dim);
  T* out_data = dev_ctx.template Alloc<T>(out);
  for (auto i = 0; i < size; i++) {
    auto it = vocab.find(input_data[0]);
    if (it != vocab.end()) {
      out_data[i] = it->second;
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(vocab_find, CPU, ALL_LAYOUT, phi::FindKernel, int32_t) {}
