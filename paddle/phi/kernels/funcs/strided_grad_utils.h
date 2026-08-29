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
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"
#include "paddle/phi/kernels/funcs/strided_view_utils.h"

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
    int64_t span = 0;
    // A span that overflows int64 cannot describe a view that fits in any
    // allocation. Report it as overlapping and let the range check reject it.
    if (!SafeMulInt64(strides[i], dims[i] - 1, &span) ||
        !SafeAddInt64(max_index_in_slice, span, &max_index_in_slice)) {
      return true;
    }
  }
  return false;
}

// Serial scatter-add of `values` (dense, in view order) into `storage` through
// the view described by (dims, stride) starting at element `elem_base`.
//
// Both pointers must be host accessible. Correctness before speed: an
// overlapping view hits the same slot several times, so this loop cannot be
// parallelized without atomics.
template <typename T>
inline void HostAccumulateStridedView(const T* values,
                                      const std::vector<int64_t>& dims,
                                      const std::vector<int64_t>& stride,
                                      int64_t elem_base,
                                      T* storage) {
  const int rank = static_cast<int>(dims.size());
  int64_t numel = 1;
  for (int i = 0; i < rank; ++i) {
    numel *= dims[i];
  }
  if (numel == 0) {
    return;
  }
  std::vector<int64_t> counter(static_cast<size_t>(rank), 0);
  int64_t dest = elem_base;
  for (int64_t i = 0; i < numel; ++i) {
    storage[dest] = static_cast<T>(storage[dest] + values[i]);
    // Odometer over the logical indices, keeping `dest` in sync.
    for (int d = rank - 1; d >= 0; --d) {
      dest += stride[d];
      if (++counter[d] < dims[d]) {
        break;
      }
      counter[d] = 0;
      dest -= stride[d] * dims[d];
    }
  }
}

// Accumulates `out_grad` into the storage of `input_grad` through the view
// described by (dims, stride, offset), where `offset` is a byte offset into
// `input_grad`'s own allocation.
//
// Unlike StridedTensorCopy, positions that the view hits more than once
// receive the sum of all contributions. `input_grad` must already be
// allocated, contiguous, zero filled, and start at element 0 of its
// allocation.
template <typename T>
inline void StridedTensorAccumulate(const DenseTensor& out_grad,
                                    const std::vector<int64_t>& dims,
                                    const std::vector<int64_t>& stride,
                                    int64_t offset,
                                    DenseTensor* input_grad) {
  PADDLE_ENFORCE_EQ(input_grad->offset(),
                    0u,
                    common::errors::InvalidArgument(
                        "The gradient buffer must start at the beginning of "
                        "its own allocation, but its offset is %d.",
                        input_grad->offset()));
  const int64_t itemsize = static_cast<int64_t>(SizeOf(input_grad->dtype()));
  PADDLE_ENFORCE_EQ(offset % itemsize,
                    0,
                    common::errors::InvalidArgument(
                        "The byte offset(%d) of a strided view must be a "
                        "multiple of its element size(%d).",
                        offset,
                        itemsize));
  const int64_t elem_base = offset / itemsize;
  const int64_t storage_numel = input_grad->numel();
  const int64_t value_numel = out_grad.numel();
  // Overflow checked. The bound is numel() and not the size of the allocation
  // because the index buffer built below covers exactly that many elements.
  const StridedViewRange range =
      ComputeStridedViewRange(dims, stride, elem_base);
  if (range.empty || value_numel == 0) {
    return;
  }
  PADDLE_ENFORCE_GE(range.min_index,
                    0,
                    common::errors::InvalidArgument(
                        "The strided view reaches element %d of the gradient "
                        "buffer, which is out of range.",
                        range.min_index));
  PADDLE_ENFORCE_LT(range.max_index,
                    storage_numel,
                    common::errors::InvalidArgument(
                        "The strided view reaches element %d, but the gradient "
                        "buffer only holds %d elements.",
                        range.max_index,
                        storage_numel));

  auto& pool = DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(input_grad->place());
  const Backend backend = TransToPhiBackend(input_grad->place());
  auto& factory = KernelFactory::Instance();
  // Only the GPU index_put accumulates atomically (CudaAtomicAdd). The CPU
  // kernel uses a plain `+=` inside an OpenMP loop, which loses updates on
  // exactly the duplicated indices that an overlapping view produces by
  // construction; XPU registers no `arange` at all and covers only a handful of
  // dtypes in index_put. Every backend but GPU, and every dtype that GPU's
  // index_put does not implement, therefore takes the serial host path below.
  // HasKernel throws when the kernel *name* is unknown, which cannot happen
  // here: both names are registered unconditionally for CPU.
  const bool use_device_scatter_add =
      input_grad->place().GetType() == AllocationType::GPU &&
      factory.HasKernel(
          "arange",
          KernelKey(backend, DataLayout::ALL_LAYOUT, DataType::INT64)) &&
      factory.HasKernel(
          "index_put",
          KernelKey(backend, DataLayout::ALL_LAYOUT, input_grad->dtype()));

  if (use_device_scatter_add) {
    // 1. arange(0, storage_numel) over the destination storage.
    DenseTensor storage_index(DataType::INT64);
    storage_index.Resize(
        common::make_ddim(std::vector<int64_t>{storage_numel}));
    using arange_signature = void (*)(const DeviceContext&,
                                      const Scalar&,
                                      const Scalar&,
                                      const Scalar&,
                                      DenseTensor*);
    PD_VISIT_KERNEL("arange",
                    KernelKey(backend, DataLayout::ALL_LAYOUT, DataType::INT64),
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
        static_cast<size_t>(elem_base * static_cast<int64_t>(sizeof(int64_t)));
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
    PD_VISIT_KERNEL(
        "index_put",
        KernelKey(backend, DataLayout::ALL_LAYOUT, input_grad->dtype()),
        index_put_signature,
        false,
        *dev_ctx,
        flat,
        indices,
        value,
        true,
        &flat);
    return;
  }

  // Serial host fallback. Densify the incoming gradient first so that the
  // accumulate loop only has to walk the destination side.
  DenseTensor value;
  value.set_meta(out_grad.meta());
  StridedTensorContiguous<T>(out_grad, &value);
  value.Resize(common::make_ddim(std::vector<int64_t>{value_numel}));

  if (input_grad->place().GetType() == AllocationType::CPU) {
    HostAccumulateStridedView<T>(
        value.data<T>(), dims, stride, elem_base, input_grad->data<T>());
    return;
  }

  // Any other backend: stage both operands on the host, accumulate there and
  // copy the result back. Slow, but a correct scatter-add is worth more than a
  // fast wrong one, and the device kernels needed to do better do not exist.
  DenseTensor host_value;
  host_value.set_layout(value.layout());
  phi::Copy(*dev_ctx, value, CPUPlace(), /*blocking=*/true, &host_value);
  DenseTensor host_storage;
  host_storage.set_layout(input_grad->layout());
  phi::Copy(
      *dev_ctx, *input_grad, CPUPlace(), /*blocking=*/true, &host_storage);
  HostAccumulateStridedView<T>(
      host_value.data<T>(), dims, stride, elem_base, host_storage.data<T>());
  phi::Copy(*dev_ctx,
            host_storage,
            input_grad->place(),
            /*blocking=*/true,
            input_grad);
}

}  // namespace phi
