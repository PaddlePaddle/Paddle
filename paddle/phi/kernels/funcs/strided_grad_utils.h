// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"

namespace phi {

// Returns true when two different logical indices of the view described by
// (dims, strides) may map to the same memory location. A view whose memory
// overlaps requires its backward to *accumulate* the incoming gradient instead
// of scattering it, otherwise the writes destroy each other.
//
// The algorithm follows at::detail::_maybe_overlapping_memory: sort the
// dimensions by stride and check that each dimension starts beyond the span
// covered by all smaller-stride dimensions. Dimensions of size <= 1 can only
// ever contribute index 0 and are skipped; a non-positive stride on a
// dimension of size > 1 (broadcast or reversed view) always overlaps.
inline bool MaybeOverlappingStrides(const std::vector<int64_t>& dims,
                                    const std::vector<int64_t>& strides) {
  if (dims.size() != strides.size()) {
    return true;
  }
  std::vector<size_t> order;
  order.reserve(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] <= 1) {
      continue;
    }
    if (strides[i] <= 0) {
      return true;
    }
    order.push_back(i);
  }
  std::sort(order.begin(), order.end(), [&strides](size_t a, size_t b) {
    return strides[a] < strides[b];
  });
  int64_t max_index_in_slice = 0;
  for (size_t i : order) {
    if (strides[i] <= max_index_in_slice) {
      return true;
    }
    max_index_in_slice += strides[i] * (dims[i] - 1);
  }
  return false;
}

// Accumulates `out_grad` into the storage of `input_grad` through the view
// described by (dims, stride, offset), where `offset` is a byte offset into
// `input_grad`'s allocation, matching AsStridedKernel.
//
// Unlike StridedTensorCopy, positions that are hit more than once by the view
// receive the sum of all contributions. `input_grad` must already be
// allocated, contiguous and zero filled.
//
// Implementation note: the flat storage index of every output element is
// materialized by taking the very same view over arange(0, storage_numel),
// which turns the problem into a plain scatter-add handled by `index_put` with
// accumulate=true. `arange` and `index_put` are dispatched at runtime via
// PD_VISIT_KERNEL because they are not registered for every (dtype, backend)
// pair that the STRIDED grad kernels are registered for.
template <typename T>
inline void StridedTensorAccumulate(const DenseTensor& out_grad,
                                    const std::vector<int64_t>& dims,
                                    const std::vector<int64_t>& stride,
                                    int64_t offset,
                                    DenseTensor* input_grad) {
  const int64_t itemsize = static_cast<int64_t>(SizeOf(input_grad->dtype()));
  PADDLE_ENFORCE_EQ(offset % itemsize,
                    0,
                    common::errors::InvalidArgument(
                        "The byte offset(%d) of a strided view must be a "
                        "multiple of its element size(%d).",
                        offset,
                        itemsize));
  const int64_t elem_offset = offset / itemsize;
  const int64_t storage_numel = input_grad->numel();
  const int64_t value_numel = out_grad.numel();

  int64_t max_elem_index = elem_offset;
  int64_t min_elem_index = elem_offset;
  for (size_t i = 0; i < dims.size(); ++i) {
    const int64_t span = stride[i] * (dims[i] - 1);
    if (span > 0) {
      max_elem_index += span;
    } else {
      min_elem_index += span;
    }
  }
  PADDLE_ENFORCE_GE(min_elem_index,
                    0,
                    common::errors::InvalidArgument(
                        "The strided view reaches element %d of the gradient "
                        "buffer, which is out of range.",
                        min_elem_index));
  PADDLE_ENFORCE_LT(max_elem_index,
                    storage_numel,
                    common::errors::InvalidArgument(
                        "The strided view reaches element %d, but the gradient "
                        "buffer only holds %d elements.",
                        max_elem_index,
                        storage_numel));

  auto& pool = DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(input_grad->place());
  const KernelKey index_key(TransToPhiBackend(input_grad->place()),
                            DataLayout::ALL_LAYOUT,
                            DataType::INT64);

  // 1. arange(0, storage_numel) over the destination storage.
  DenseTensor storage_index(DataType::INT64);
  storage_index.Resize(common::make_ddim(std::vector<int64_t>{storage_numel}));
  using arange_signature = void (*)(const DeviceContext&,
                                    const Scalar&,
                                    const Scalar&,
                                    const Scalar&,
                                    DenseTensor*);
  PD_VISIT_KERNEL("arange",
                  index_key,
                  arange_signature,
                  false,
                  *dev_ctx,
                  Scalar(static_cast<int64_t>(0)),
                  Scalar(storage_numel),
                  Scalar(static_cast<int64_t>(1)),
                  &storage_index);

  // 2. Apply the view to it, so that element i of `index` is the storage slot
  // that element i of `out_grad` belongs to.
  DenseTensor index_view(storage_index);
  DenseTensorMeta index_view_meta = storage_index.meta();
  index_view_meta.dims = DDim(dims.data(), static_cast<int>(dims.size()));
  index_view_meta.strides =
      DDim(stride.data(), static_cast<int>(stride.size()));
  index_view_meta.offset =
      static_cast<size_t>(elem_offset * static_cast<int64_t>(sizeof(int64_t)));
  index_view.set_meta(index_view_meta);

  DenseTensor index;
  index.set_meta(index_view.meta());
  StridedTensorContiguous<int64_t>(index_view, &index);
  index.Resize(common::make_ddim(std::vector<int64_t>{value_numel}));

  // 3. Densify the incoming gradient in the matching order.
  DenseTensor value;
  value.set_meta(out_grad.meta());
  StridedTensorContiguous<T>(out_grad, &value);
  value.Resize(common::make_ddim(std::vector<int64_t>{value_numel}));

  // 4. Scatter-add into the flattened destination storage. Aliasing x and out
  // is intentional: `input_grad` is already initialized, so `index_put` skips
  // the x -> out copy and accumulates in place.
  DenseTensor flat(*input_grad);
  DenseTensorMeta flat_meta = input_grad->meta();
  flat_meta.dims = common::make_ddim(std::vector<int64_t>{storage_numel});
  flat_meta.strides = common::make_ddim(std::vector<int64_t>{1});
  flat_meta.offset = 0;
  flat.set_meta(flat_meta);

  std::vector<const DenseTensor*> indices = {&index};
  using index_put_signature = void (*)(const DeviceContext&,
                                       const DenseTensor&,
                                       const std::vector<const DenseTensor*>&,
                                       const DenseTensor&,
                                       bool,
                                       DenseTensor*);
  PD_VISIT_KERNEL("index_put",
                  KernelKey(TransToPhiBackend(input_grad->place()),
                            DataLayout::ALL_LAYOUT,
                            input_grad->dtype()),
                  index_put_signature,
                  false,
                  *dev_ctx,
                  flat,
                  indices,
                  value,
                  true,
                  &flat);
}

}  // namespace phi
