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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace phi {
namespace sparse {

#define DECLARE_SPARSE_UNARY_GRAD_KERNEL(prefix)              \
  template <typename T, typename Context>                     \
  void prefix##CooGradKernel(const Context& dev_ctx,          \
                             const SparseCooTensor& x_or_out, \
                             const SparseCooTensor& dout,     \
                             SparseCooTensor* dx);            \
                                                              \
  template <typename T, typename Context>                     \
  void prefix##CsrGradKernel(const Context& dev_ctx,          \
                             const SparseCsrTensor& x_or_out, \
                             const SparseCsrTensor& dout,     \
                             SparseCsrTensor* dx);

#define DECLARE_SPARSE_UNARY_GRAD_KERNEL_WITH_ONE_ATTR(prefix, attr) \
  template <typename T, typename Context>                            \
  void prefix##CooGradKernel(const Context& dev_ctx,                 \
                             const SparseCooTensor& x_or_out,        \
                             const SparseCooTensor& dout,            \
                             float attr,                             \
                             SparseCooTensor* dx);                   \
                                                                     \
  template <typename T, typename Context>                            \
  void prefix##CsrGradKernel(const Context& dev_ctx,                 \
                             const SparseCsrTensor& x_or_out,        \
                             const SparseCsrTensor& dout,            \
                             float attr,                             \
                             SparseCsrTensor* dx);

DECLARE_SPARSE_UNARY_GRAD_KERNEL(Sin)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Tan)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Asin)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Atan)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Sinh)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Asinh)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Atanh)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Relu)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Tanh)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Square)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Sqrt)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Log1p)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Abs)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Expm1)
DECLARE_SPARSE_UNARY_GRAD_KERNEL(Relu6)
DECLARE_SPARSE_UNARY_GRAD_KERNEL_WITH_ONE_ATTR(Pow, factor)
DECLARE_SPARSE_UNARY_GRAD_KERNEL_WITH_ONE_ATTR(LeakyRelu, alpha)

template <typename T, typename Context>
void CastCooGradKernel(const Context& dev_ctx,
                       const SparseCooTensor& x,
                       const SparseCooTensor& dout,
                       DataType value_dtype,
                       SparseCooTensor* dx);

template <typename T, typename Context>
void CastCsrGradKernel(const Context& dev_ctx,
                       const SparseCsrTensor& x,
                       const SparseCsrTensor& dout,
                       DataType value_dtype,
                       SparseCsrTensor* dx);

template <typename T, typename Context>
void TransposeCooGradKernel(const Context& dev_ctx,
                            const SparseCooTensor& dout,
                            const std::vector<int>& perm,
                            SparseCooTensor* dx);

template <typename T, typename Context>
void TransposeCsrGradKernel(const Context& dev_ctx,
                            const SparseCsrTensor& dout,
                            const std::vector<int>& perm,
                            SparseCsrTensor* dx);

template <typename T, typename Context>
void SumCooGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const SparseCooTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCooTensor* dx);

template <typename T, typename Context>
void SumCsrGradKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const SparseCsrTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCsrTensor* dx);

template <typename T, typename Context>
void ReshapeCooGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          const SparseCooTensor& dout,
                          SparseCooTensor* dx);

template <typename T, typename Context>
void ReshapeCsrGradKernel(const Context& dev_ctx,
                          const SparseCsrTensor& x,
                          const SparseCsrTensor& dout,
                          SparseCsrTensor* dx);

template <typename T, typename Context>
void SliceCooGradKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const SparseCooTensor& out_grad,
                        const phi::IntArray& axes,
                        const phi::IntArray& starts,
                        const phi::IntArray& ends,
                        SparseCooTensor* x_grad);

template <typename T, typename Context>
void SliceCsrGradKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const SparseCsrTensor& out_grad,
                        const phi::IntArray& axes,
                        const phi::IntArray& starts,
                        const phi::IntArray& ends,
                        SparseCsrTensor* x_grad);
}  // namespace sparse
}  // namespace phi
