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
#include "paddle/pten/kernels/sparse/cpu/sparse_coo_tensor_util.h"
#include "paddle/pten/kernels/sparse/cuda/sparse_coo_tensor_utils.h"

namespace pten {

template <typename T, typename ContextT>
SparseCooTensor ToSparseCoo(const ContextT& dev_ctx,
                            const DenseTensor& x,
                            const int64_t sparse_dim) {
  DenseTensorMeta indices_meta, values_meta;
  indices_meta.dtype = DataType::INT64;
  values_meta.dtype = x.meta().dtype;
  values_meta.layout = x.meta().layout;
  pten::DenseTensor indices(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(indices_meta));
  pten::DenseTensor values(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(values_meta));
  SparseCooTensor coo(indices, values, x.dims());
  ToSparseCoo<T>(dev_ctx, x, sparse_dim, &coo);
  return coo;
}

}  // namespace pten
