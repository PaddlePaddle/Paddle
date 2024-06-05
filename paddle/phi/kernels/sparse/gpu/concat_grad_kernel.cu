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

#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/sparse/concat_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"
namespace phi {
namespace sparse {

template <typename T, typename Context>
void ConcatCooGradKernel(const Context& dev_ctx,
                         const std::vector<const SparseCooTensor*>& x,
                         const SparseCooTensor& out_grad,
                         const Scalar& axis_scalar,
                         std::vector<SparseCooTensor*> x_grad) {
  PADDLE_ENFORCE_NOT_NULL(
      x[0], phi::errors::NotFound("The first input tensor is not initalized."));

  auto axis = axis_scalar.to<int64_t>();
  axis = phi::funcs::ComputeAxis(static_cast<int64_t>(axis),
                                 static_cast<int64_t>(x[0]->dims().size()));
  const size_t num_split = x_grad.size();
  for (size_t i = 0; i < num_split; i++) {
    if (x_grad[i] && x_grad[i]->numel() != 0UL) {
      // Handle of crows and cols
      EmptyLikeCooKernel<T, Context>(dev_ctx, *x[i], x_grad[i]);
    } else {
      x_grad[i] = nullptr;
    }
  }

  int64_t cumulative_offset = 0;
  for (size_t i = 0; i < x_grad.size(); ++i) {
    phi::sparse::SliceCooKernel<T, Context>(
        dev_ctx,
        out_grad,
        {axis},
        {cumulative_offset},
        {x[i]->dims()[axis] + cumulative_offset},
        x_grad[i]);
    cumulative_offset += x[i]->dims()[axis];
  }
}

template <typename T, typename Context>
void ConcatCsrGradKernel(const Context& dev_ctx,
                         const std::vector<const SparseCsrTensor*>& x,
                         const SparseCsrTensor& out_grad,
                         const Scalar& axis_scalar,
                         std::vector<SparseCsrTensor*> x_grad) {
  PADDLE_ENFORCE_NOT_NULL(
      x[0], phi::errors::NotFound("The first input tensor is not initalized."));
  const size_t num_split = x_grad.size();
  for (size_t i = 0; i < num_split; i++) {
    if (x_grad[i] && x_grad[i]->numel() != 0UL) {
      EmptyLikeCsrKernel<T, Context>(dev_ctx, *x[i], x_grad[i]);

    } else {
      x_grad[i] = nullptr;
    }
  }
  auto axis = axis_scalar.to<int>();
  axis = phi::funcs::ComputeAxis(static_cast<int64_t>(axis),
                                 static_cast<int64_t>(x[0]->dims().size()));
  PADDLE_ENFORCE_GE(
      axis,
      0,
      phi::errors::InvalidArgument("concat_grad: axis should be larger than or "
                                   "equal to 0, but received axis is %d.",
                                   axis));
  int64_t cumulative_offset = 0;
  for (size_t i = 0; i < num_split; ++i) {
    phi::sparse::SliceCsrKernel<T, Context>(
        dev_ctx,
        out_grad,
        {axis},
        {cumulative_offset},
        {x[i]->dims()[axis] + cumulative_offset},
        x_grad[i]);
    cumulative_offset += x[i]->dims()[axis];
  }
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(concat_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ConcatCooGradKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t) {}

PD_REGISTER_KERNEL(concat_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ConcatCsrGradKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t) {}
