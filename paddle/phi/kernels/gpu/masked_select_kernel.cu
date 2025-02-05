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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/select_impl.cu.h"

namespace phi {

template <typename MT, typename InT, typename OutT>
struct MaskedSelectFunctor {
  HOSTDEVICE MaskedSelectFunctor() = default;

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
  DenseTensor mask_expand;
  DenseTensor x_expand;

  auto expanded_size = funcs::MatrixGetBroadcastBatchPortion(
      common::vectorize(x.dims()), common::vectorize(mask.dims()));

  DDim expand_dims = common::make_ddim(expanded_size);
  if (mask.dims() != expand_dims) {
    phi::ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  if (x.dims() != expand_dims) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, x, IntArray(expanded_size), &x_expand);
  } else {
    x_expand = x;
  }

  auto input_dim = x_expand.dims();
  auto mask_dim = mask_expand.dims();
  PADDLE_ENFORCE_EQ(input_dim,
                    mask_dim,
                    common::errors::InvalidArgument(
                        "The dim size of input and mask in OP(masked_selected) "
                        "must be equal, but got input dim:(%ld), mask dim: "
                        "(%ld). Please check input "
                        "value.",
                        input_dim,
                        mask_dim));

  using Functor = MaskedSelectFunctor<bool, T, T>;
  phi::funcs::SelectKernel<bool, T, T, 1, Functor>(
      dev_ctx, mask_expand, x_expand, out, Functor());
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_select,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaskedSelectKernel,
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
                   phi::dtype::complex<double>) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
