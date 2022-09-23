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

#include "paddle/phi/kernels/masked_select_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MaskedSelectKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& mask,
                        DenseTensor* out) {
  auto* mask_data = mask.data<bool>();
  auto input_data = x.data<T>();

  auto mask_size = mask.numel();

  auto input_dim = x.dims();
  auto mask_dim = mask.dims();
  PADDLE_ENFORCE_EQ(input_dim,
                    mask_dim,
                    phi::errors::InvalidArgument(
                        "The dim size of input and mask in OP(masked_selected) "
                        "must be equal, but got input dim:(%ld), mask dim: "
                        "(%ld). Please check input "
                        "value.",
                        input_dim,
                        mask_dim));

  int out_size = 0;
  for (int i = 0; i < mask_size; i++) {
    if (mask_data[i]) out_size++;
  }

  DDim out_dim{out_size};
  out->Resize(out_dim);
  auto out_data = out->mutable_data<T>(phi::CPUPlace());

  int index = 0;
  for (int i = 0; i < mask_size; i++) {
    if (mask_data[i]) {
      out_data[index] = input_data[i];
      index++;
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_select,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaskedSelectKernel,
                   float,
                   double,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
