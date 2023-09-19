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

#pragma once

#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/abs_grad_kernel.h"
#include "paddle/phi/kernels/activation_grad_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"

namespace phi {
namespace sparse {

#define DEFINE_SPARSE_UNARY_GRAD_KERNEL(prefix)                           \
  template <typename T, typename Context>                                 \
  void prefix##CooGradKernel(const Context& dev_ctx,                      \
                             const SparseCooTensor& x_or_out,             \
                             const SparseCooTensor& dout,                 \
                             SparseCooTensor* dx) {                       \
    EmptyLikeCooKernel<T, Context>(dev_ctx, x_or_out, dx);                \
    phi::prefix##GradKernel<T, Context>(dev_ctx,                          \
                                        x_or_out.non_zero_elements(),     \
                                        dout.non_zero_elements(),         \
                                        dx->mutable_non_zero_elements()); \
  }                                                                       \
                                                                          \
  template <typename T, typename Context>                                 \
  void prefix##CsrGradKernel(const Context& dev_ctx,                      \
                             const SparseCsrTensor& x_or_out,             \
                             const SparseCsrTensor& dout,                 \
                             SparseCsrTensor* dx) {                       \
    EmptyLikeCsrKernel<T, Context>(dev_ctx, x_or_out, dx);                \
    phi::prefix##GradKernel<T, Context>(dev_ctx,                          \
                                        x_or_out.non_zero_elements(),     \
                                        dout.non_zero_elements(),         \
                                        dx->mutable_non_zero_elements()); \
  }

#define DEFINE_SPARSE_UNARY_GRAD_KERNEL_WITH_ONE_ATTR(prefix, attr)       \
  template <typename T, typename Context>                                 \
  void prefix##CooGradKernel(const Context& dev_ctx,                      \
                             const SparseCooTensor& x_or_out,             \
                             const SparseCooTensor& dout,                 \
                             float attr,                                  \
                             SparseCooTensor* dx) {                       \
    EmptyLikeCooKernel<T, Context>(dev_ctx, x_or_out, dx);                \
    phi::prefix##GradKernel<T, Context>(dev_ctx,                          \
                                        x_or_out.non_zero_elements(),     \
                                        dout.non_zero_elements(),         \
                                        attr,                             \
                                        dx->mutable_non_zero_elements()); \
  }                                                                       \
                                                                          \
  template <typename T, typename Context>                                 \
  void prefix##CsrGradKernel(const Context& dev_ctx,                      \
                             const SparseCsrTensor& x_or_out,             \
                             const SparseCsrTensor& dout,                 \
                             float attr,                                  \
                             SparseCsrTensor* dx) {                       \
    EmptyLikeCsrKernel<T, Context>(dev_ctx, x_or_out, dx);                \
    phi::prefix##GradKernel<T, Context>(dev_ctx,                          \
                                        x_or_out.non_zero_elements(),     \
                                        dout.non_zero_elements(),         \
                                        attr,                             \
                                        dx->mutable_non_zero_elements()); \
  }

DEFINE_SPARSE_UNARY_GRAD_KERNEL(Sin)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Tan)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Asin)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Atan)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Sinh)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Tanh)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Asinh)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Atanh)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Sqrt)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Square)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Log1p)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Relu)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Abs)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Expm1)
DEFINE_SPARSE_UNARY_GRAD_KERNEL(Relu6)
DEFINE_SPARSE_UNARY_GRAD_KERNEL_WITH_ONE_ATTR(Pow, factor)
DEFINE_SPARSE_UNARY_GRAD_KERNEL_WITH_ONE_ATTR(LeakyRelu, alpha)

template <typename T, typename Context>
void CastCooGradKernel(const Context& dev_ctx,
                       const SparseCooTensor& x,
                       const SparseCooTensor& dout,
                       DataType value_dtype,
                       SparseCooTensor* dx) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, dx);
  if (value_dtype == DataType::UNDEFINED) {
    phi::Copy(dev_ctx,
              dout.non_zero_elements(),
              dev_ctx.GetPlace(),
              false,
              dx->mutable_non_zero_elements());
  } else {
    phi::CastKernel<T, Context>(dev_ctx,
                                dout.non_zero_elements(),
                                x.non_zero_elements().dtype(),
                                dx->mutable_non_zero_elements());
  }
}

template <typename T, typename Context>
void CastCsrGradKernel(const Context& dev_ctx,
                       const SparseCsrTensor& x,
                       const SparseCsrTensor& dout,
                       DataType value_dtype,
                       SparseCsrTensor* dx) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, dx);
  if (value_dtype == DataType::UNDEFINED) {
    phi::Copy(dev_ctx,
              dout.non_zero_elements(),
              dev_ctx.GetPlace(),
              false,
              dx->mutable_non_zero_elements());
  } else {
    phi::CastKernel<T, Context>(dev_ctx,
                                dout.non_zero_elements(),
                                x.non_zero_elements().dtype(),
                                dx->mutable_non_zero_elements());
  }
}

}  // namespace sparse
}  // namespace phi
