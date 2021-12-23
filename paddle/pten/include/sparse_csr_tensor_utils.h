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
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/kernels/sparse/cpu/sparse_csr_tensor_utils.h"
#include "paddle/pten/kernels/sparse/cuda/sparse_csr_tensor_utils.h"

namespace pten {

template <typename T, typename ContextT>
SparseCsrTensor ToSparseCsr(const ContextT& dev_ctx, const DenseTensor& x) {
  DenseTensorMeta crows_meta, cols_meta, values_meta;
  crows_meta.dtype = DataType::INT64;
  cols_meta.dtype = DataType::INT64;
  values_meta.dtype = x.meta().dtype;
  values_meta.layout = x.meta().layout;
  pten::DenseTensor crows(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(crows_meta));
  pten::DenseTensor cols(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(cols_meta));
  pten::DenseTensor values(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(values_meta));
  SparseCsrTensor csr(crows, cols, values, x.dims());
  ToSparseCsr<T>(dev_ctx, x, &csr);
  return csr;
}

template <typename T, typename ContextT>
SparseCsrTensor SparseCooToCsr(const ContextT& dev_ctx,
                               const SparseCooTensor& x) {
  DenseTensorMeta crows_meta, cols_meta, values_meta;
  crows_meta.dtype = DataType::INT64;
  cols_meta.dtype = DataType::INT64;
  values_meta.dtype = x.dtype();
  values_meta.layout = x.layout();
  pten::DenseTensor crows(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(crows_meta));
  pten::DenseTensor cols(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(cols_meta));
  pten::DenseTensor values(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(values_meta));
  SparseCsrTensor csr(crows, cols, values, x.dims());
  SparseCooToCsr<T>(dev_ctx, x, &csr);
  return csr;
}

template <typename T, typename ContextT>
DenseTensor SparseCsrToDense(const ContextT& dev_ctx,
                             const SparseCsrTensor& x) {
  auto dense_meta =
      pten::DenseTensorMeta(x.dtype(), x.dims(), pten::DataLayout::NCHW);

  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          dev_ctx.GetPlace());
  DenseTensor dense_out(allocator, dense_meta);
  SparseCsrToDense<T>(dev_ctx, x, &dense_out);
  return dense_out;
}

}  // namespace pten
