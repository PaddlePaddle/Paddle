// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/stack_grad_kernel.h"
#include <algorithm>
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/funcs/strided_reshape_utils.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename T, typename Context>
void StackGradStrideKernel(const Context& dev_ctx,
                           const DenseTensor& out_grad,
                           int axis,
                           std::vector<DenseTensor*> x_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  if (axis < 0) {
    axis = axis + out_grad.dims().size();
  }

  if (!out_grad.IsInitialized()) {
    return;
  }

  DenseTensor out_grad_;
  if (!FLAGS_use_stride_compute_kernel || IsComplexType(out_grad.dtype())) {
    if (!out_grad.meta().is_contiguous()) {
      phi::MetaTensor meta_input(out_grad);
      phi::MetaTensor meta_out(&out_grad_);
      UnchangedInferMeta(meta_input, &meta_out);
      phi::ContiguousKernel<T, Context>(dev_ctx, out_grad, &out_grad_);
    } else {
      out_grad_ = out_grad;
    }

    for (int i = 0; i < x_grad.size(); i++) {
      if (x_grad[i]) {
        auto meta = x_grad[i]->meta();
        meta.strides = meta.calc_strides(x_grad[i]->dims());
        x_grad[i]->set_meta(meta);
      }
    }
    phi::StackGradKernel<T, Context>(dev_ctx, out_grad_, axis, x_grad);
    return;
  }

  for (int i = 0; i < x_grad.size(); i++) {
    int64_t index = static_cast<int64_t>(i);
    int64_t size = out_grad.dims()[axis];

    std::vector<int64_t> sizes = common::vectorize<int64_t>(out_grad.dims());
    std::vector<int64_t> strides =
        common::vectorize<int64_t>(out_grad.strides());

    auto storage_offset = out_grad.offset() +
                          index * strides[axis] * phi::SizeOf(out_grad.dtype());

    sizes.erase(sizes.begin() + axis);
    strides.erase(strides.begin() + axis);

    auto meta = x_grad[i]->meta();
    meta.dims = common::make_ddim(sizes);
    meta.strides = common::make_ddim(strides);
    meta.offset = storage_offset;
    x_grad[i]->set_meta(meta);
    x_grad[i]->ResetHolder(out_grad.Holder());
    x_grad[i]->ShareInplaceVersionCounterWith(out_grad);
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(stack_grad,
                   CPU,
                   STRIDED,
                   phi::StackGradStrideKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int16_t,
                   int64_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(stack_grad,
                   GPU,
                   STRIDED,
                   phi::StackGradStrideKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   uint8_t,
                   int16_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2,
                   phi::complex64,
                   phi::complex128) {}
#endif

#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_CUDA)
PD_REGISTER_KERNEL(stack_grad,
                   XPU,
                   STRIDED,
                   phi::StackGradStrideKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int64_t,
                   int,
                   int16_t,
                   int8_t,
                   uint8_t) {}
#endif
