// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <cstring>
#include <limits>

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/fill_kernel.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"

namespace phi {
template <typename T>
inline void StridedTensorCopy(const DenseTensor& input,
                              const std::vector<int64_t>& dims,
                              const std::vector<int64_t>& out_stride,
                              int64_t offset,
                              DenseTensor* out) {
  auto& pool = DeviceContextPool::Instance();
  if (input.place().GetType() == AllocationType::CPU) {
    auto* dev_ctx = static_cast<CPUContext*>(pool.Get(input.place()));
    phi::StridedCopyKernel<T, CPUContext>(
        *dev_ctx, input, dims, out_stride, offset, out);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (input.place().GetType() == AllocationType::GPU) {
    auto* dev_ctx = static_cast<GPUContext*>(pool.Get(input.place()));
    phi::StridedCopyKernel<T, GPUContext>(
        *dev_ctx, input, dims, out_stride, offset, out);
#endif
#ifdef PADDLE_WITH_XPU
  } else if (input.place().GetType() == AllocationType::XPU) {
    auto* dev_ctx = static_cast<XPUContext*>(pool.Get(input.place()));
    phi::StridedCopyKernel<T, XPUContext>(
        *dev_ctx, input, dims, out_stride, offset, out);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (input.place().GetType() == AllocationType::CUSTOM) {
    auto* dev_ctx = static_cast<phi::CustomContext*>(pool.Get(input.place()));
    const phi::KernelKey& strided_copy_key = {
        phi::TransToPhiBackend(dev_ctx->GetPlace()),
        DataLayout::ALL_LAYOUT,
        input.dtype()};
    using strided_copy_signature = void (*)(const DeviceContext&,
                                            const DenseTensor&,
                                            const std::vector<int64_t>&,
                                            const std::vector<int64_t>&,
                                            int64_t,
                                            DenseTensor*);
    PD_VISIT_KERNEL("strided_copy",
                    strided_copy_key,
                    strided_copy_signature,
                    false,
                    *dev_ctx,
                    input,
                    dims,
                    out_stride,
                    offset,
                    out);
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Place type is not supported when `strided_copy` kernel is called."));
  }
}

template <typename T>
inline void StridedTensorFill(const DenseTensor& x,
                              const phi::Scalar& value,
                              DenseTensor* out) {
  auto& pool = DeviceContextPool::Instance();
  if (x.place().GetType() == AllocationType::CPU) {
    auto* dev_ctx = static_cast<CPUContext*>(pool.Get(x.place()));
    phi::FillKernel<T, CPUContext>(*dev_ctx, x, value, out);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (x.place().GetType() == AllocationType::GPU) {
    auto* dev_ctx = static_cast<GPUContext*>(pool.Get(x.place()));
    phi::FillKernel<T, GPUContext>(*dev_ctx, x, value, out);
#endif
#ifdef PADDLE_WITH_XPU
  } else if (x.place().GetType() == AllocationType::XPU) {
    auto* dev_ctx = static_cast<XPUContext*>(pool.Get(x.place()));
    phi::FillKernel<T, XPUContext>(*dev_ctx, x, value, out);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (x.place().GetType() == AllocationType::CUSTOM) {
    auto* dev_ctx = static_cast<phi::CustomContext*>(pool.Get(x.place()));
    const phi::KernelKey& fill_key = {
        phi::TransToPhiBackend(dev_ctx->GetPlace()),
        DataLayout::ALL_LAYOUT,
        x.dtype()};
    using fill_signature = void (*)(const DeviceContext&,
                                    const DenseTensor&,
                                    const phi::Scalar&,
                                    DenseTensor*);
    PD_VISIT_KERNEL(
        "fill", fill_key, fill_signature, false, *dev_ctx, x, value, out);
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Place type is not supported when `fill` kernel is called."));
  }
}

template <typename T>
inline void StridedTensorContiguous(const DenseTensor& input,
                                    DenseTensor* out) {
  auto& pool = DeviceContextPool::Instance();
  if (input.place().GetType() == AllocationType::CPU) {
    auto* dev_ctx = static_cast<CPUContext*>(pool.Get(input.place()));
    ContiguousKernel<T, CPUContext>(*dev_ctx, input, out);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (input.place().GetType() == AllocationType::GPU) {
    auto* dev_ctx = static_cast<GPUContext*>(pool.Get(input.place()));
    ContiguousKernel<T, GPUContext>(*dev_ctx, input, out);
#endif
#ifdef PADDLE_WITH_XPU
  } else if (input.place().GetType() == AllocationType::XPU) {
    auto* dev_ctx = static_cast<XPUContext*>(pool.Get(input.place()));
    ContiguousKernel<T, XPUContext>(*dev_ctx, input, out);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (input.place().GetType() == AllocationType::CUSTOM) {
    auto* dev_ctx = static_cast<phi::CustomContext*>(pool.Get(input.place()));
    const phi::KernelKey& contiguous_key = {
        phi::TransToPhiBackend(dev_ctx->GetPlace()),
        DataLayout::ALL_LAYOUT,
        input.dtype()};
    using contiguous_signature =
        void (*)(const DeviceContext&, const DenseTensor&, DenseTensor*);
    PD_VISIT_KERNEL("contiguous",
                    contiguous_key,
                    contiguous_signature,
                    false,
                    *dev_ctx,
                    input,
                    out);
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Place type is not supported when `contiguous` kernel is called."));
  }
}

// int64_t arithmetic that reports overflow instead of wrapping around.
//
// The element range of a strided view is computed from a caller supplied
// (shape, stride, offset) triple, so the products and sums below can exceed
// int64_t for adversarial input. Without these guards the range check wraps
// around and happily accepts a view that reaches far outside of the
// allocation, which is worse than having no check at all. __int128 and
// __builtin_*_overflow are avoided on purpose: this header is reachable from
// the public compat headers, which are also compiled by MSVC.
inline bool CheckedMulInt64(int64_t a, int64_t b, int64_t* out) {
  constexpr int64_t kMax = (std::numeric_limits<int64_t>::max)();
  constexpr int64_t kMin = (std::numeric_limits<int64_t>::lowest)();
  if (a == 0 || b == 0) {
    *out = 0;
    return true;
  }
  // Handled separately so that the divisions below never evaluate kMin / -1.
  if (a == -1) {
    if (b == kMin) return false;
    *out = -b;
    return true;
  }
  if (b == -1) {
    if (a == kMin) return false;
    *out = -a;
    return true;
  }
  if (a > 0) {
    if (b > 0 ? a > kMax / b : b < kMin / a) return false;
  } else {
    if (b > 0 ? a < kMin / b : a < kMax / b) return false;
  }
  *out = a * b;
  return true;
}

inline bool CheckedAddInt64(int64_t a, int64_t b, int64_t* out) {
  constexpr int64_t kMax = (std::numeric_limits<int64_t>::max)();
  constexpr int64_t kMin = (std::numeric_limits<int64_t>::lowest)();
  if (b > 0 && a > kMax - b) return false;
  if (b < 0 && a < kMin - b) return false;
  *out = a + b;
  return true;
}

// Turns a torch style storage offset, expressed in *elements* and relative to
// the offset of the tensor being viewed, into the absolute *byte* offset that
// DenseTensorMeta::offset and phi::AsStridedKernel expect. Do not call this on
// an offset that is already a byte offset.
//
// The conversion needs its own helper because phi::SizeOf returns size_t, so
// the obvious `static_cast<size_t>(offset) * SizeOf(dtype)` is unsigned
// arithmetic that wraps around silently: a storage offset of 1 << 62 on a four
// byte dtype becomes 0, and the range check below would then happily validate
// a completely different view and let the out of range request through.
inline int64_t ElementOffsetToByteOffset(int64_t base_byte_offset,
                                         int64_t element_offset,
                                         int64_t itemsize) {
  PADDLE_ENFORCE_GE(element_offset,
                    0,
                    common::errors::InvalidArgument(
                        "The storage offset of a view must be non-negative, "
                        "but got %d.",
                        element_offset));
  int64_t byte_offset = 0;
  bool no_overflow = CheckedMulInt64(element_offset, itemsize, &byte_offset);
  if (no_overflow) {
    no_overflow = CheckedAddInt64(base_byte_offset, byte_offset, &byte_offset);
  }
  PADDLE_ENFORCE_EQ(no_overflow,
                    true,
                    common::errors::InvalidArgument(
                        "The byte offset of the view overflows int64: element "
                        "offset %d of a %d byte dtype, relative to byte offset "
                        "%d.",
                        element_offset,
                        itemsize,
                        base_byte_offset));
  return byte_offset;
}

// Element indices, relative to the start of the allocation, that a strided
// view touches. `empty` marks a view with a zero sized dimension, which reads
// and writes nothing at all.
struct StridedViewRange {
  int64_t min_index{0};
  int64_t max_index{0};
  bool empty{false};
};

inline StridedViewRange ComputeStridedViewRange(
    const std::vector<int64_t>& dims,
    const std::vector<int64_t>& strides,
    int64_t base_index) {
  // Every loop below indexes strides[i] for i < dims.size(); reject a
  // mismatched shape/stride pair before entering the loop.
  PADDLE_ENFORCE_EQ(dims.size(),
                    strides.size(),
                    common::errors::InvalidArgument(
                        "The size of dims(%d) and strides(%d) of a strided "
                        "view should be equal.",
                        dims.size(),
                        strides.size()));
  StridedViewRange range;
  range.min_index = base_index;
  range.max_index = base_index;
  for (size_t i = 0; i < dims.size(); ++i) {
    PADDLE_ENFORCE_GE(dims[i],
                      0,
                      common::errors::InvalidArgument(
                          "The shape of a strided view must be non-negative, "
                          "but got %s.",
                          common::make_ddim(dims)));
    if (dims[i] == 0) {
      range.empty = true;
      return range;
    }
    int64_t dim_span = 0;
    bool no_overflow = CheckedMulInt64(strides[i], dims[i] - 1, &dim_span);
    if (no_overflow) {
      no_overflow =
          dim_span > 0
              ? CheckedAddInt64(range.max_index, dim_span, &range.max_index)
              : CheckedAddInt64(range.min_index, dim_span, &range.min_index);
    }
    PADDLE_ENFORCE_EQ(
        no_overflow,
        true,
        common::errors::InvalidArgument(
            "The element range of the view described by shape %s, stride %s "
            "and element offset %d overflows int64.",
            common::make_ddim(dims),
            common::make_ddim(strides),
            base_index));
  }
  return range;
}

inline void ValidateZeroSizeTensorShape(const std::vector<int64_t>& dims,
                                        const std::vector<int64_t>& strides,
                                        const DenseTensor& input) {
  if (input.numel() != 0) {
    return;
  }
  PADDLE_ENFORCE_EQ(dims.size(),
                    strides.size(),
                    common::errors::InvalidArgument(
                        "The size of dims and strides should be equal."));
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] == 0) {
      return;
    }
  }
  PADDLE_THROW(common::errors::InvalidArgument(
      "When input is zero-size tensor, the shape attribute must also be "
      "zero-size."));
}

// Rejects views that would reach outside of `input`'s allocation. Without this
// check a bad (shape, stride, offset) triple silently produces a tensor whose
// reads and writes corrupt neighbouring heap memory.
//
// `offset` is a byte offset into the allocation, matching AsStridedKernel and
// DenseTensorMeta::offset.
inline void ValidateStridedViewStorage(const std::vector<int64_t>& dims,
                                       const std::vector<int64_t>& strides,
                                       int64_t offset,
                                       const DenseTensor& input) {
  PADDLE_ENFORCE_EQ(
      dims.size(),
      strides.size(),
      common::errors::InvalidArgument(
          "The size of dims(%d) and strides(%d) of a strided view should be "
          "equal.",
          dims.size(),
          strides.size()));
  if (input.numel() == 0 || input.Holder() == nullptr) {
    return;
  }
  const int64_t itemsize = static_cast<int64_t>(SizeOf(input.dtype()));
  PADDLE_ENFORCE_EQ(offset % itemsize,
                    0,
                    common::errors::InvalidArgument(
                        "The offset(%d) is a byte offset and must be a "
                        "multiple of the element size(%d) of the input.",
                        offset,
                        itemsize));
  const StridedViewRange range =
      ComputeStridedViewRange(dims, strides, offset / itemsize);
  if (range.empty) {
    return;
  }
  const int64_t storage_numel =
      static_cast<int64_t>(input.Holder()->size()) / itemsize;
  PADDLE_ENFORCE_GE(range.min_index,
                    0,
                    common::errors::InvalidArgument(
                        "The view described by shape %s, stride %s and offset "
                        "%d reaches element %d, which is before the beginning "
                        "of the input storage.",
                        common::make_ddim(dims),
                        common::make_ddim(strides),
                        offset,
                        range.min_index));
  PADDLE_ENFORCE_LT(range.max_index,
                    storage_numel,
                    common::errors::InvalidArgument(
                        "The view described by shape %s, stride %s and offset "
                        "%d reaches element %d, but the input storage only "
                        "holds %d elements.",
                        common::make_ddim(dims),
                        common::make_ddim(strides),
                        offset,
                        range.max_index,
                        storage_numel));
}

// Returns true when two different logical indices of the view described by
// (dims, strides) may map to the same memory location. A view whose memory
// overlaps requires its backward to *accumulate* the incoming gradient instead
// of scattering it, otherwise the writes destroy each other.
//
// Sort dimensions by absolute stride and check that each dimension starts
// beyond the span covered by all smaller-stride dimensions. Dimensions of
// size <= 1 can only ever contribute index 0 and are skipped. A zero stride
// on a dimension of size > 1 is a broadcast and always overlaps; negative
// strides are valid reversed views and are checked by their absolute span.
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
    if (strides[i] == 0) {
      return true;
    }
    if (strides[i] == (std::numeric_limits<int64_t>::lowest)()) {
      return true;
    }
    order.push_back(i);
  }
  std::sort(order.begin(), order.end(), [&strides](size_t a, size_t b) {
    const int64_t stride_a = strides[a] < 0 ? -strides[a] : strides[a];
    const int64_t stride_b = strides[b] < 0 ? -strides[b] : strides[b];
    return stride_a < stride_b;
  });
  int64_t max_index_in_slice = 0;
  for (size_t i : order) {
    const int64_t abs_stride = strides[i] < 0 ? -strides[i] : strides[i];
    if (abs_stride <= max_index_in_slice) {
      return true;
    }
    int64_t dim_span = 0;
    // A span that overflows int64 cannot describe a view that fits in any
    // allocation. Report it as overlapping and let the range check reject it.
    if (!CheckedMulInt64(abs_stride, dims[i] - 1, &dim_span) ||
        !CheckedAddInt64(max_index_in_slice, dim_span, &max_index_in_slice)) {
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
// allocation. Zero filling is a hard requirement and not just a convention:
// the host path below relies on it to skip reading the destination back from
// the device, so a non-zero buffer would be accumulated into on GPU but
// overwritten everywhere else.
template <typename T>
inline void StridedTensorAccumulate(const DenseTensor& out_grad,
                                    const std::vector<int64_t>& dims,
                                    const std::vector<int64_t>& stride,
                                    int64_t offset,
                                    DenseTensor* input_grad) {
  PADDLE_ENFORCE_EQ(input_grad->meta().offset,
                    0u,
                    common::errors::InvalidArgument(
                        "The gradient buffer must start at the beginning of "
                        "its own allocation, but its offset is %d.",
                        input_grad->meta().offset));
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
    // DenseTensor(DataType) lives in dense_tensor.inl and is hidden from
    // PADDLE_WITH_CUSTOM_KERNEL TUs, including ATen_resize_custom_kernel_test
    // which includes this header through as_strided.h. set_meta(const&) is
    // always visible and overwrites the default-constructed float32 meta.
    DenseTensor storage_index;
    storage_index.set_meta(DenseTensorMeta(
        DataType::INT64,
        common::make_ddim(std::vector<int64_t>{storage_numel})));
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
    DenseTensor dense_grad;
    dense_grad.set_meta(out_grad.meta());
    StridedTensorContiguous<T>(out_grad, &dense_grad);
    dense_grad.Resize(common::make_ddim(std::vector<int64_t>{value_numel}));

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
        dense_grad,
        true,
        &flat);
    return;
  }

  // Serial host fallback. Densify the incoming gradient first so that the
  // accumulate loop only has to walk the destination side.
  DenseTensor dense_grad;
  dense_grad.set_meta(out_grad.meta());
  StridedTensorContiguous<T>(out_grad, &dense_grad);
  dense_grad.Resize(common::make_ddim(std::vector<int64_t>{value_numel}));

  if (input_grad->place().GetType() == AllocationType::CPU) {
    HostAccumulateStridedView<T>(
        dense_grad.data<T>(), dims, stride, elem_base, input_grad->data<T>());
    return;
  }

  // Any other backend: accumulate on the host and copy the result back. Slow,
  // but a correct scatter-add is worth more than a fast wrong one, and the
  // device kernels needed to do better do not exist.
  //
  // Only the incoming gradient is staged with a device to host copy. The
  // destination is required to be zero filled on entry, so reading it back
  // would only reproduce zeros; allocating the host side zeroed instead saves
  // a full copy of the gradient buffer.
  DenseTensor host_grad;
  const DenseTensorMeta host_grad_meta(dense_grad.dtype(),
                                       dense_grad.dims(),
                                       dense_grad.layout(),
                                       dense_grad.meta().offset);
  host_grad.set_meta(host_grad_meta);
  phi::Copy(*dev_ctx, dense_grad, CPUPlace(), /*blocking=*/true, &host_grad);
  DenseTensor host_storage;
  host_storage.set_meta(input_grad->meta());
  T* host_ptr =
      static_cast<T*>(dev_ctx->HostAlloc(&host_storage, host_storage.dtype()));
  std::memset(host_ptr, 0, static_cast<size_t>(storage_numel) * sizeof(T));
  HostAccumulateStridedView<T>(
      host_grad.data<T>(), dims, stride, elem_base, host_ptr);
  phi::Copy(*dev_ctx,
            host_storage,
            input_grad->place(),
            /*blocking=*/true,
            input_grad);
}

// Backward of a strided view whose `input` is itself a non-contiguous view, or
// whose window starts before `input` does.
//
// (dims, stride, offset) describe the view in the coordinate system of the
// allocation shared with `input`, while `input_grad` is a dense row-major
// buffer over `input`'s own logical indices. Those two coordinate systems only
// differ by a constant when `input` is contiguous; in general the storage index
// of an element is not its row-major index, so the gradient has to be routed
// through a temporary buffer laid out in storage coordinates: scatter-add
// `out_grad` into it through the out geometry, then gather it back through
// `input`'s own geometry. This mirrors at::as_strided_backward.
//
// `input_grad` must already be allocated with `input`'s dims; its previous
// contents are overwritten. Only the meta of `input` is read, never its data,
// because the forward declares `no_need_buffer : input`.
template <typename T>
inline void StridedTensorAccumulateThroughStorage(
    const DenseTensor& out_grad,
    const std::vector<int64_t>& dims,
    const std::vector<int64_t>& stride,
    int64_t offset,
    const DenseTensor& input,
    DenseTensor* input_grad) {
  const int64_t itemsize = static_cast<int64_t>(SizeOf(input_grad->dtype()));
  PADDLE_ENFORCE_EQ(offset % itemsize,
                    0,
                    common::errors::InvalidArgument(
                        "The byte offset(%d) of a strided view must be a "
                        "multiple of its element size(%d).",
                        offset,
                        itemsize));
  const std::vector<int64_t> input_dims =
      common::vectorize<int64_t>(input.dims());
  const std::vector<int64_t> input_stride =
      common::vectorize<int64_t>(input.strides());
  const int64_t out_base = offset / itemsize;
  const int64_t input_base =
      static_cast<int64_t>(input.meta().offset) / itemsize;
  const StridedViewRange out_range =
      ComputeStridedViewRange(dims, stride, out_base);
  const StridedViewRange input_range =
      ComputeStridedViewRange(input_dims, input_stride, input_base);
  if (out_range.empty || input_range.empty || out_grad.numel() == 0) {
    return;
  }
  const int64_t shared_base =
      std::min(out_range.min_index, input_range.min_index);
  PADDLE_ENFORCE_GE(shared_base,
                    0,
                    common::errors::InvalidArgument(
                        "The strided view reaches element %d of the shared "
                        "allocation, which is before its beginning.",
                        shared_base));
  int64_t storage_numel = 0;
  const int64_t shared_last =
      std::max(out_range.max_index, input_range.max_index);
  PADDLE_ENFORCE_EQ(
      CheckedAddInt64(shared_last - shared_base, 1, &storage_numel),
      true,
      common::errors::InvalidArgument(
          "The element range spanned by the view described by shape %s, "
          "stride %s and byte offset %d together with its input overflows "
          "int64.",
          common::make_ddim(dims),
          common::make_ddim(stride),
          offset));

  auto& pool = DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(input_grad->place());
  // The rvalue set_meta overload requires an invalid destination meta, and a
  // default constructed DenseTensor already reports a valid one (float32,
  // NCHW, rank -1 dims). The const-ref overload overwrites dtype/dims instead.
  // DenseTensor(DataType) is also unavailable under PADDLE_WITH_CUSTOM_KERNEL.
  DenseTensor storage;
  storage.set_meta(
      DenseTensorMeta(input_grad->dtype(),
                      common::make_ddim(std::vector<int64_t>{storage_numel})));
  dev_ctx->Alloc(&storage, storage.dtype());
  StridedTensorFill<T>(storage, 0, &storage);

  const int64_t out_byte_base = (out_base - shared_base) * itemsize;
  if (MaybeOverlappingStrides(dims, stride)) {
    StridedTensorAccumulate<T>(out_grad, dims, stride, out_byte_base, &storage);
  } else {
    // Distinct slots, so a plain scatter is enough and avoids the serial
    // accumulate. Mirrors the copy_ branch of at::as_strided_backward.
    StridedTensorCopy<T>(out_grad, dims, stride, out_byte_base, &storage);
  }

  // Read the accumulated storage back through `input`'s geometry. Contiguous
  // materialization is exactly the gather that turns storage indices into
  // row-major ones, and it reuses the buffer `input_grad` already owns.
  DenseTensor gathered(storage);
  DenseTensorMeta gathered_meta = storage.meta();
  gathered_meta.dims = input.dims();
  gathered_meta.strides = input.strides();
  gathered_meta.offset =
      static_cast<size_t>((input_base - shared_base) * itemsize);
  gathered.set_meta(gathered_meta);
  StridedTensorContiguous<T>(gathered, input_grad);
}
}  // namespace phi
