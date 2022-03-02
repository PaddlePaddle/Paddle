/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void MeanRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const std::vector<int64_t>& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DenseTensor* out);

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                DenseTensor* out);

template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DataType out_dtype,
                  DenseTensor* out);

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               DataType out_dtype,
               bool keep_dim,
               DenseTensor* out);

template <typename T, typename Context>
void AddRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  int axis,
                  DenseTensor* out);

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out);

template <typename T, typename Context>
void SubtractRawKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       int axis,
                       DenseTensor* out);

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out);

template <typename T, typename Context>
void DivideRawKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     int axis,
                     DenseTensor* out);

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* out);

template <typename T, typename Context>
void MultiplyRawKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       int axis,
                       DenseTensor* out);

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out);

template <typename T, typename Context>
DenseTensor Add(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y) {
  auto dense_out = phi::Empty<T, Context>(dev_ctx);
  MetaTensor meta_out(&dense_out);
  ElementwiseInferMeta(x, y, &meta_out);
  AddKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Subtract(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y) {
  auto dense_out = phi::Empty<T, Context>(dev_ctx);
  MetaTensor meta_out(&dense_out);
  ElementwiseInferMeta(x, y, &meta_out);
  SubtractKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Divide(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y) {
  auto dense_out = phi::Empty<T, Context>(dev_ctx);
  MetaTensor meta_out(&dense_out);
  ElementwiseInferMeta(x, y, &meta_out);
  DivideKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Multiply(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y) {
  auto dense_out = phi::Empty<T, Context>(dev_ctx);
  MetaTensor meta_out(&dense_out);
  ElementwiseInferMeta(x, y, &meta_out);
  MultiplyKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Mean(const Context& dev_ctx,
                 const DenseTensor& x,
                 const std::vector<int64_t>& axis,
                 bool keep_dim) {
  auto dense_out = phi::Empty<T, Context>(dev_ctx);
  MetaTensor meta_out(&dense_out);
  ReduceInferMetaBase(x, axis, keep_dim, false, x.dtype(), &meta_out);
  MeanKernel<T, Context>(dev_ctx, x, axis, keep_dim, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Sum(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& axis,
                DataType dtype,
                bool keep_dim) {
  auto dense_out = phi::Empty<T, Context>(dev_ctx);
  MetaTensor meta_out(&dense_out);
  SumInferMeta(x, axis, dtype, keep_dim, &meta_out);
  SumKernel<T, Context>(dev_ctx, x, axis, dtype, keep_dim, &dense_out);
  return dense_out;
}

}  // namespace phi
