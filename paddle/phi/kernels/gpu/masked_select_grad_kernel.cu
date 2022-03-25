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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/select_impl.cu.h"
#include "paddle/phi/kernels/masked_select_grad_kernel.h"

namespace phi {

template <typename MT, typename InT, typename OutT>
struct MaskedSelectGradFunctor {
  HOSTDEVICE MaskedSelectGradFunctor() {}

  HOSTDEVICE inline void operator()(OutT* out,
                                    const MT* mask,
                                    const InT* value,
                                    int num) {
    int read_fix = 0;
    for (int idx = 0; idx < num; idx++) {
      if (mask[idx]) {
        out[idx] = value[read_fix++];
      } else {
        out[idx] = 0;
      }
    }
  }
};

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const DenseTensor& out_grad,
                            const DenseTensor& x,
                            const DenseTensor& mask,
                            DenseTensor* x_grad) {
  auto mask_size = mask.numel();
  dev_ctx.template Alloc<T>(x_grad);
  if (mask_size <= 0) return;
  using Functor = MaskedSelectGradFunctor<bool, T, T>;
  phi::funcs::SelectKernel<bool, T, T, 2, Functor>(
      dev_ctx, mask, out_grad, x_grad, Functor());
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_select_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaskedSelectGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
