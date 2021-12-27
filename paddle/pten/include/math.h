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

// See Note: [ How do we organize the kernel directory ]
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/include/infermeta.h"
#include "paddle/pten/kernels/complex_kernel.h"
#include "paddle/pten/kernels/cpu/math.h"
#include "paddle/pten/kernels/gpu/math.h"
#include "paddle/pten/kernels/scale_kernel.h"

namespace pten {

template <typename T, typename ContextT>
DenseTensor Sign(const ContextT& dev_ctx, const DenseTensor& x) {
  auto out_meta = UnchangedInferMeta(x.meta());
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Sign<T>(dev_ctx, x, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Mean(const ContextT& dev_ctx,
                 const DenseTensor& x,
                 const std::vector<int64_t>& axis,
                 bool keep_dim) {
  auto out_meta = ReduceInferMeta(x.meta(), axis, keep_dim);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  bool reduce_all = false;
  Mean<T>(dev_ctx, x, axis, keep_dim, reduce_all, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Sum(const ContextT& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& axis,
                DataType dtype,
                bool keep_dim) {
  auto out_meta = ReduceInferMeta(x.meta(), axis, keep_dim, dtype);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      out_meta);

  // The real value of reduce_all will be get in kernel
  // so use default value(false) is OK.
  bool reduce_all = false;

  Sum<T>(dev_ctx, x, axis, keep_dim, reduce_all, out_meta.dtype, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Scale(const ContextT& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& scale,
                  float bias,
                  bool bias_after_scale) {
  auto out_meta = UnchangedInferMeta(x.meta());
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Scale<T, ContextT>(dev_ctx, x, scale, bias, bias_after_scale, &dense_out);
  return dense_out;
}

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
  Add<T>(dev_ctx, x, y, axis, &dense_out);
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
  Subtract<T>(dev_ctx, x, y, axis, &dense_out);
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
  Divide<T>(dev_ctx, x, y, axis, &dense_out);
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
  Multiply<T>(dev_ctx, x, y, axis, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Conj(const ContextT& dev_ctx, const DenseTensor& x) {
  auto out_meta = UnchangedInferMeta(x.meta());
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Conj<T>(dev_ctx, x, &dense_out);
  return dense_out;
}

}  // namespace pten
