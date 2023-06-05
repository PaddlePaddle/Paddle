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
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace phi {
namespace sparse {

#define DECLARE_SPARSE_UNARY_KERNEL(prefix)                                    \
  template <typename T, typename Context>                                      \
  void prefix##CooKernel(                                                      \
      const Context& dev_ctx, const SparseCooTensor& x, SparseCooTensor* out); \
                                                                               \
  template <typename T, typename Context>                                      \
  void prefix##CsrKernel(                                                      \
      const Context& dev_ctx, const SparseCsrTensor& x, SparseCsrTensor* out);

#define DECLARE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(prefix, attr) \
  template <typename T, typename Context>                       \
  void prefix##CooKernel(const Context& dev_ctx,                \
                         const SparseCooTensor& x,              \
                         float attr,                            \
                         SparseCooTensor* out);                 \
                                                                \
  template <typename T, typename Context>                       \
  void prefix##CsrKernel(const Context& dev_ctx,                \
                         const SparseCsrTensor& x,              \
                         float attr,                            \
                         SparseCsrTensor* out);

DECLARE_SPARSE_UNARY_KERNEL(Sin)
DECLARE_SPARSE_UNARY_KERNEL(Tan)
DECLARE_SPARSE_UNARY_KERNEL(Asin)
DECLARE_SPARSE_UNARY_KERNEL(Atan)
DECLARE_SPARSE_UNARY_KERNEL(Sinh)
DECLARE_SPARSE_UNARY_KERNEL(Asinh)
DECLARE_SPARSE_UNARY_KERNEL(Atanh)
DECLARE_SPARSE_UNARY_KERNEL(Relu)
DECLARE_SPARSE_UNARY_KERNEL(Isnan)
DECLARE_SPARSE_UNARY_KERNEL(Tanh)
DECLARE_SPARSE_UNARY_KERNEL(Square)
DECLARE_SPARSE_UNARY_KERNEL(Sqrt)
DECLARE_SPARSE_UNARY_KERNEL(Log1p)
DECLARE_SPARSE_UNARY_KERNEL(Abs)
DECLARE_SPARSE_UNARY_KERNEL(Expm1)
DECLARE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(Pow, factor)
DECLARE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(Relu6Raw, threshold)
DECLARE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(LeakyRelu, alpha)

template <typename T, typename Context>
void Relu6CooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    SparseCooTensor* out);

template <typename T, typename Context>
void Relu6CsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    SparseCsrTensor* out);

template <typename T, typename Context>
void ScaleCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    float scale,
                    float bias,
                    bool bias_after_scale,
                    SparseCooTensor* out);

template <typename T, typename Context>
void ScaleCsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    float scale,
                    float bias,
                    bool bias_after_scale,
                    SparseCsrTensor* out);

template <typename T, typename Context>
void DivScalarCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        float scalar,
                        SparseCooTensor* out);

template <typename T, typename Context>
void DivScalarCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        float scalar,
                        SparseCsrTensor* out);

template <typename T, typename Context>
void CastCooKernel(const Context& dev_ctx,
                   const SparseCooTensor& x,
                   DataType index_dtype,
                   DataType value_dtype,
                   SparseCooTensor* out);

template <typename T, typename Context>
void CastCsrKernel(const Context& dev_ctx,
                   const SparseCsrTensor& x,
                   DataType index_dtype,
                   DataType value_dtype,
                   SparseCsrTensor* out);

template <typename T, typename Context>
void TransposeCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const std::vector<int>& perm,
                        SparseCooTensor* out);

template <typename T, typename Context>
void TransposeCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const std::vector<int>& perm,
                        SparseCsrTensor* out);

template <typename T, typename Context>
SparseCooTensor TransposeCoo(const Context& dev_ctx,
                             const SparseCooTensor& x,
                             const std::vector<int>& perm) {
  PADDLE_ENFORCE_EQ(x.sparse_dim(),
                    perm.size(),
                    phi::errors::InvalidArgument(
                        "size of perm must be equal than the x.sparse_dim()"));
  SparseCooTensor coo;
  TransposeCooKernel<T, Context>(dev_ctx, x, perm, &coo);
  return coo;
}

template <typename T, typename Context>
SparseCsrTensor TransposeCsr(const Context& dev_ctx,
                             const SparseCsrTensor& x,
                             const std::vector<int>& perm) {
  PADDLE_ENFORCE_LE(
      2,
      perm.size(),
      phi::errors::InvalidArgument("size of perm must be equal to 2 or 3"));
  PADDLE_ENFORCE_GE(
      3,
      perm.size(),
      phi::errors::InvalidArgument("size of perm must be equal to 2 or 3"));
  SparseCsrTensor csr;
  TransposeCsrKernel<T, Context>(dev_ctx, x, perm, &csr);
  return csr;
}

template <typename T, typename Context>
void SumCooKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCooTensor* out);

template <typename T, typename Context>
void SumCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCsrTensor* out);

template <typename T, typename Context>
SparseCooTensor ReluCoo(const Context& dev_ctx, const SparseCooTensor& x) {
  SparseCooTensor coo;
  ReluCooKernel<T, Context>(dev_ctx, x, &coo);
  return coo;
}

template <typename T, typename Context>
SparseCooTensor ReluCsr(const Context& dev_ctx, const SparseCooTensor& x) {
  SparseCooTensor csr;
  ReluCsrKernel<T, Context>(dev_ctx, x, &csr);
  return csr;
}

template <typename T, typename Context>
void ReshapeCooKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const phi::IntArray& shape,
                      SparseCooTensor* out);

template <typename T, typename Context>
void ReshapeCsrKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const phi::IntArray& shape,
                      SparseCsrTensor* out);

template <typename T, typename Context>
SparseCooTensor ReshapeCoo(const Context& dev_ctx,
                           const SparseCooTensor& x,
                           const phi::IntArray& shape) {
  SparseCooTensor coo;
  ReshapeCooKernel<T, Context>(dev_ctx, x, shape, &coo);
  return coo;
}

template <typename T, typename Context>
SparseCsrTensor ReshapeCsr(const Context& dev_ctx,
                           const SparseCsrTensor& x,
                           const phi::IntArray& shape) {
  PADDLE_ENFORCE_LE(
      2,
      shape.size(),
      phi::errors::InvalidArgument("size of shape must be equal to 2 or 3"));
  PADDLE_ENFORCE_GE(
      3,
      shape.size(),
      phi::errors::InvalidArgument("size of shape must be equal to 2 or 3"));
  SparseCsrTensor csr;
  ReshapeCsrKernel<T, Context>(dev_ctx, x, shape, &csr);
  return csr;
}

template <typename T, typename Context>
void SliceCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const phi::IntArray& axes,
                    const phi::IntArray& starts,
                    const phi::IntArray& ends,
                    SparseCooTensor* out);

template <typename T, typename Context>
void SliceCsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    const phi::IntArray& axes,
                    const phi::IntArray& starts,
                    const phi::IntArray& ends,
                    SparseCsrTensor* out);

}  // namespace sparse
}  // namespace phi
