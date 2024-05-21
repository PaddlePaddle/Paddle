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
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/isfinite_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

#define DEFINE_SPARSE_UNARY_KERNEL(prefix)                                 \
  template <typename T, typename Context>                                  \
  void prefix##CooKernel(const Context& dev_ctx,                           \
                         const SparseCooTensor& x,                         \
                         SparseCooTensor* out) {                           \
    EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);                       \
    phi::prefix##Kernel<T, Context>(                                       \
        dev_ctx, x.non_zero_elements(), out->mutable_non_zero_elements()); \
    out->SetIndicesDict(x.GetIndicesDict());                               \
    out->SetKmaps(x.GetKmaps());                                           \
  }                                                                        \
                                                                           \
  template <typename T, typename Context>                                  \
  void prefix##CsrKernel(const Context& dev_ctx,                           \
                         const SparseCsrTensor& x,                         \
                         SparseCsrTensor* out) {                           \
    EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);                       \
    phi::prefix##Kernel<T, Context>(                                       \
        dev_ctx, x.non_zero_elements(), out->mutable_non_zero_elements()); \
  }

#define DEFINE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(prefix, attr)         \
  template <typename T, typename Context>                              \
  void prefix##CooKernel(const Context& dev_ctx,                       \
                         const SparseCooTensor& x,                     \
                         float attr,                                   \
                         SparseCooTensor* out) {                       \
    EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);                   \
    phi::prefix##Kernel<T, Context>(dev_ctx,                           \
                                    x.non_zero_elements(),             \
                                    attr,                              \
                                    out->mutable_non_zero_elements()); \
  }                                                                    \
                                                                       \
  template <typename T, typename Context>                              \
  void prefix##CsrKernel(const Context& dev_ctx,                       \
                         const SparseCsrTensor& x,                     \
                         float attr,                                   \
                         SparseCsrTensor* out) {                       \
    EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);                   \
    phi::prefix##Kernel<T, Context>(dev_ctx,                           \
                                    x.non_zero_elements(),             \
                                    attr,                              \
                                    out->mutable_non_zero_elements()); \
  }

DEFINE_SPARSE_UNARY_KERNEL(Sin)
DEFINE_SPARSE_UNARY_KERNEL(Tan)
DEFINE_SPARSE_UNARY_KERNEL(Asin)
DEFINE_SPARSE_UNARY_KERNEL(Atan)
DEFINE_SPARSE_UNARY_KERNEL(Sinh)
DEFINE_SPARSE_UNARY_KERNEL(Tanh)
DEFINE_SPARSE_UNARY_KERNEL(Asinh)
DEFINE_SPARSE_UNARY_KERNEL(Atanh)
DEFINE_SPARSE_UNARY_KERNEL(Sqrt)
DEFINE_SPARSE_UNARY_KERNEL(Square)
DEFINE_SPARSE_UNARY_KERNEL(Log1p)
DEFINE_SPARSE_UNARY_KERNEL(Relu)
DEFINE_SPARSE_UNARY_KERNEL(Expm1)
DEFINE_SPARSE_UNARY_KERNEL(Relu6)
DEFINE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(Pow, factor)
DEFINE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(LeakyRelu, alpha)

template <typename T, typename Context>
void AbsCooKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  SparseCooTensor* out) {
  *(out->mutable_indices()) = x.indices();

  DenseTensor* out_values = out->mutable_values();
  const DenseTensor& x_values = x.values();
  out_values->Resize(x_values.dims());
  dev_ctx.template Alloc<T>(out_values);

  phi::AbsKernel<T, Context>(
      dev_ctx, x.non_zero_elements(), out->mutable_non_zero_elements());

  out->SetIndicesDict(x.GetIndicesDict());
}

template <typename T, typename Context>
void AbsCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  SparseCsrTensor* out) {
  *(out->mutable_crows()) = x.crows();
  *(out->mutable_cols()) = x.cols();

  DenseTensor* out_values = out->mutable_values();
  const DenseTensor& x_values = x.values();
  out_values->Resize(x_values.dims());
  dev_ctx.template Alloc<T>(out_values);

  phi::AbsKernel<T, Context>(
      dev_ctx, x.non_zero_elements(), out->mutable_non_zero_elements());
}

template <typename T, typename Context>
void ScaleCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    float scale,
                    float bias,
                    bool bias_after_scale,
                    SparseCooTensor* out) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);
  phi::ScaleKernel<T, Context>(dev_ctx,
                               x.non_zero_elements(),
                               scale,
                               bias,
                               bias_after_scale,
                               out->mutable_non_zero_elements());
  out->SetIndicesDict(x.GetIndicesDict());
  out->SetKmaps(x.GetKmaps());
}

template <typename T, typename Context>
void ScaleCsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    float scale,
                    float bias,
                    bool bias_after_scale,
                    SparseCsrTensor* out) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);
  phi::ScaleKernel<T, Context>(dev_ctx,
                               x.non_zero_elements(),
                               scale,
                               bias,
                               bias_after_scale,
                               out->mutable_non_zero_elements());
}

template <typename T, typename Context>
void CastCooKernel(const Context& dev_ctx,
                   const SparseCooTensor& x,
                   DataType index_dtype,
                   DataType value_dtype,
                   SparseCooTensor* out) {
  const DenseTensor& x_indices = x.indices();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_indices = out->mutable_indices();
  DenseTensor* out_values = out->mutable_non_zero_elements();

  if (index_dtype == DataType::UNDEFINED) {
    *out_indices = x_indices;
  } else {
    phi::MetaTensor meta(out_indices);
    meta.set_dims(x_indices.dims());
    meta.set_dtype(index_dtype);

    PD_VISIT_INTEGRAL_TYPES(x_indices.dtype(), "CastCooKernel", [&] {
      phi::CastKernel<data_t, Context>(
          dev_ctx, x_indices, index_dtype, out_indices);
    });
  }

  if (value_dtype == DataType::UNDEFINED) {
    phi::Copy(dev_ctx, x_values, dev_ctx.GetPlace(), false, out_values);
  } else {
    phi::MetaTensor meta(out_values);
    meta.set_dims(x_values.dims());
    phi::CastKernel<T, Context>(dev_ctx, x_values, value_dtype, out_values);
  }
  out->SetIndicesDict(x.GetIndicesDict());
  out->SetKmaps(x.GetKmaps());
}

template <typename T, typename Context>
void CastCsrKernel(const Context& dev_ctx,
                   const SparseCsrTensor& x,
                   DataType index_dtype,
                   DataType value_dtype,
                   SparseCsrTensor* out) {
  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_cols = x.cols();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_crows = out->mutable_crows();
  DenseTensor* out_cols = out->mutable_cols();
  DenseTensor* out_values = out->mutable_non_zero_elements();

  if (index_dtype == DataType::UNDEFINED) {
    *out_crows = x_crows;
    *out_cols = x_cols;
  } else {
    phi::MetaTensor crows_meta(out_crows);
    crows_meta.set_dims(x_crows.dims());
    crows_meta.set_dtype(index_dtype);

    PD_VISIT_INTEGRAL_TYPES(x_crows.dtype(), "CastCsrKernel", [&] {
      phi::CastKernel<data_t, Context>(
          dev_ctx, x_crows, index_dtype, out_crows);
    });

    phi::MetaTensor cols_meta(out_cols);
    cols_meta.set_dims(x_cols.dims());
    cols_meta.set_dtype(index_dtype);

    PD_VISIT_INTEGRAL_TYPES(x_cols.dtype(), "CastCsrKernel", [&] {
      phi::CastKernel<data_t, Context>(dev_ctx, x_cols, index_dtype, out_cols);
    });
  }

  if (value_dtype == DataType::UNDEFINED) {
    phi::Copy(dev_ctx, x_values, dev_ctx.GetPlace(), false, out_values);
  } else {
    phi::MetaTensor meta(out_values);
    meta.set_dims(x_values.dims());
    phi::CastKernel<T, Context>(dev_ctx, x_values, value_dtype, out_values);
  }
}

template <typename T, typename Context>
void IsnanCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    SparseCooTensor* out) {
  *(out->mutable_indices()) = x.indices();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_values = out->mutable_non_zero_elements();

  phi::MetaTensor meta(out_values);
  meta.set_dims(x_values.dims());
  meta.set_dtype(DataType::BOOL);

  phi::IsnanKernel<T, Context>(
      dev_ctx, x.non_zero_elements(), out->mutable_non_zero_elements());
  out->SetIndicesDict(x.GetIndicesDict());
  out->SetKmaps(x.GetKmaps());
}

template <typename T, typename Context>
void IsnanCsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    SparseCsrTensor* out) {
  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_cols = x.cols();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_crows = out->mutable_crows();
  DenseTensor* out_cols = out->mutable_cols();
  DenseTensor* out_values = out->mutable_non_zero_elements();

  *out_crows = x_crows;
  *out_cols = x_cols;

  phi::MetaTensor meta(out_values);
  meta.set_dims(x_values.dims());
  meta.set_dtype(DataType::BOOL);

  phi::IsnanKernel<T, Context>(
      dev_ctx, x.non_zero_elements(), out->mutable_non_zero_elements());
}

}  // namespace sparse
}  // namespace phi
