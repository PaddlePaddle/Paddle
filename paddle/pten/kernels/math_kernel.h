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

#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/include/infermeta.h"
#include "paddle/pten/kernels/empty_kernel.h"

namespace pten {

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                bool reduce_all,
                DenseTensor* out);

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               int axis,
               DenseTensor* out);

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out);

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  int axis,
                  DenseTensor* out);

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out);

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               bool reduce_all,
               DataType out_dtype,
               DenseTensor* out);

template <typename T, typename ContextT>
DenseTensor Add(const ContextT& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                int axis) {
  auto out_meta = ElementwiseInferMeta(x.meta(), y.meta(), axis);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  AddKernel<T, ContextT>(dev_ctx, x, y, axis, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Subtract(const ContextT& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     int axis) {
  auto out_meta = ElementwiseInferMeta(x.meta(), y.meta(), axis);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  SubtractKernel<T, ContextT>(dev_ctx, x, y, axis, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Divide(const ContextT& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   int axis) {
  auto out_meta = ElementwiseInferMeta(x.meta(), y.meta(), axis);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  DivideKernel<T, ContextT>(dev_ctx, x, y, axis, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Multiply(const ContextT& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     int axis) {
  auto out_meta = ElementwiseInferMeta(x.meta(), y.meta(), axis);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  MultiplyKernel<T, ContextT>(dev_ctx, x, y, axis, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Mean(const Context& dev_ctx,
                 const DenseTensor& x,
                 const std::vector<int64_t>& axis,
                 bool keep_dim) {
  auto out_meta = ReduceInferMeta(x.meta(), axis, keep_dim);
  auto dense_out = pten::Empty<T, Context>(dev_ctx, std::move(out_meta));
  bool reduce_all = false;
  MeanKernel<T, Context>(dev_ctx, x, axis, keep_dim, reduce_all, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Sum(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& axis,
                DataType dtype,
                bool keep_dim) {
  auto out_meta = ReduceInferMeta(x.meta(), axis, keep_dim, dtype);
  auto dense_out = pten::Empty<T, Context>(dev_ctx, std::move(out_meta));

  // The real value of reduce_all will be get in kernel
  // so use default value(false) is OK.
  bool reduce_all = false;

  SumKernel<T, Context>(
      dev_ctx, x, axis, keep_dim, reduce_all, out_meta.dtype, &dense_out);
  return dense_out;
}

}  // namespace pten
