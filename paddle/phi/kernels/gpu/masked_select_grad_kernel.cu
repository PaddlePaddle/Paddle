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

#include "paddle/phi/kernels/masked_select_grad_kernel.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_grad_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/funcs/select_impl.cu.h"

namespace phi {

template <typename MT, typename InT, typename OutT>
struct MaskedSelectGradFunctor {
  HOSTDEVICE MaskedSelectGradFunctor() = default;

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
                            const DenseTensor& x,
                            const DenseTensor& mask,
                            const DenseTensor& out_grad,
                            DenseTensor* x_grad) {
  // x_grad.size() == x.size()
  // x.size() == mask.size(), no broadcast, expand_mask = false, expand_x =
  // false x.size() < mask.size(), x broadcast to mask, expand_mask = false,
  // expand_x = true x.size() > mask.size(), mask broadcast to x, expand_mask =
  // true, expand_x = false
  DenseTensor mask_expand;
  DenseTensor x_grad_expand;
  bool expand_x = false;

  auto expanded_size = funcs::MatrixGetBroadcastBatchPortion(
      common::vectorize(x_grad->dims()), common::vectorize(mask.dims()));
  auto expanded_dims = common::make_ddim(expanded_size);

  if (mask.dims() != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  if (x_grad->dims() != expanded_dims) {
    x_grad_expand = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
    expand_x = true;
  } else {
    expand_x = false;
  }

  dev_ctx.template Alloc<T>(x_grad);
  auto mask_size = mask_expand.numel();
  if (mask_size <= 0) return;

  using Functor = MaskedSelectGradFunctor<bool, T, T>;

  DenseTensor* x_grad_tmp = x_grad;
  if (expand_x) {
    x_grad_tmp = &x_grad_expand;
  }

  phi::funcs::SelectKernel<bool, T, T, 2, Functor>(
      dev_ctx, mask_expand, out_grad, x_grad_tmp, Functor());

  if (expand_x) {
    ExpandGradKernel<T, Context>(
        dev_ctx, x, x_grad_expand, IntArray(expanded_size), x_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_select_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaskedSelectGradKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
