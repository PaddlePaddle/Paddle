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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>

#include "paddle/phi/kernels/funcs/select_impl.cu.h"
#include "paddle/phi/kernels/masked_select_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename MT, typename InT, typename OutT>
struct MaskedSelectFunctor {
  HOSTDEVICE MaskedSelectFunctor() {}

  HOSTDEVICE inline void operator()(OutT* out,
                                    const MT* mask,
                                    const InT* value,
                                    int num) {
    int store_fix = 0;
    for (int idx = 0; idx < num; idx++) {
      if (mask[idx]) {
        out[store_fix++] = value[idx];
      }
    }
  }
};

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
  using Functor = MaskedSelectFunctor<bool, T, T>;
  phi::funcs::SelectKernel<bool, T, T, 1, Functor>(
      dev_ctx, mask, x, out, Functor());
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_select,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaskedSelectKernel,
                   float,
                   double,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
