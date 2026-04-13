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

#include <cstddef>
#include <cstring>
#include <memory>

#include "paddle/common/ddim.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_meta.h"

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#elif defined(PADDLE_WITH_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace compat {

inline void _PD_FreeMappedPinnedAllocation(phi::Allocation* allocation) {
  if (allocation == nullptr || allocation->ptr() == nullptr) {
    return;
  }
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipHostFree(allocation->ptr()));
#elif defined(PADDLE_WITH_CUDA)
  PADDLE_ENFORCE_GPU_SUCCESS(cudaFreeHost(allocation->ptr()));
#endif
}

inline std::shared_ptr<phi::Allocation> _PD_CreateMappedPinnedAllocation(
    size_t bytes, const phi::Place& pinned_place) {
  if (bytes == 0) {
    return std::make_shared<phi::Allocation>(nullptr, 0, pinned_place);
  }

  void* ptr = nullptr;
#ifdef PADDLE_WITH_HIP
  constexpr unsigned int kMappedPinnedFlags =
      hipHostMallocPortable | hipHostMallocMapped;
  PADDLE_ENFORCE_GPU_SUCCESS(hipHostMalloc(&ptr, bytes, kMappedPinnedFlags));
#elif defined(PADDLE_WITH_CUDA)
  constexpr unsigned int kMappedPinnedFlags =
      cudaHostAllocPortable | cudaHostAllocMapped;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaHostAlloc(&ptr, bytes, kMappedPinnedFlags));
#else
  PD_THROW("Mapped GPU pinned memory requires CUDA or HIP support.");
#endif

  return std::make_shared<phi::Allocation>(
      ptr, bytes, &_PD_FreeMappedPinnedAllocation, pinned_place);
}

inline paddle::Tensor _PD_EmptyPinnedTensor(const paddle::IntArray& shape,
                                            phi::DataType dtype,
                                            const phi::Place& pinned_place) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::is_cuda_pinned_place(pinned_place)) {
    auto dims = common::make_ddim(shape.GetData());
    auto meta = phi::DenseTensorMeta(dtype, dims);
    auto bytes =
        static_cast<size_t>(common::product(dims)) * phi::SizeOf(dtype);
    auto holder = _PD_CreateMappedPinnedAllocation(bytes, pinned_place);
    return paddle::Tensor(std::make_shared<phi::DenseTensor>(holder, meta));
  }
#endif

  auto dense = paddle::experimental::empty(shape, dtype, phi::CPUPlace());
  return dense.copy_to(pinned_place, /*blocking=*/true);
}

inline paddle::Tensor _PD_CopyTensorToPinnedPlace(
    const paddle::Tensor& src, const phi::Place& pinned_place) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::is_cuda_pinned_place(pinned_place)) {
    auto src_dense = std::dynamic_pointer_cast<phi::DenseTensor>(src.impl());
    if (src_dense && src_dense->meta().is_contiguous() &&
        src_dense->meta().offset == 0) {
      auto bytes = src_dense->memory_size();
      auto holder = _PD_CreateMappedPinnedAllocation(bytes, pinned_place);
      if (bytes > 0) {
        std::memcpy(holder->ptr(), src_dense->data(), bytes);
      }
      return paddle::Tensor(
          std::make_shared<phi::DenseTensor>(holder, src_dense->meta()));
    }
  }
#endif

  return src.copy_to(pinned_place, /*blocking=*/true);
}

inline void* _PD_GetKernelVisibleDataPtr(const paddle::Tensor& tensor) {
  if (!tensor.defined()) {
    return nullptr;
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::is_cuda_pinned_place(tensor.place())) {
    auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
    if (!dense) {
      return const_cast<void*>(tensor.data());
    }

    auto holder = dense->Holder();
    if (!holder || holder->ptr() == nullptr) {
      return const_cast<void*>(tensor.data());
    }

    void* mapped_base = nullptr;
#ifdef PADDLE_WITH_HIP
    auto err = hipHostGetDevicePointer(&mapped_base, holder->ptr(), 0);
    if (err == hipSuccess && mapped_base != nullptr) {
      return static_cast<char*>(mapped_base) + dense->meta().offset;
    }
    (void)hipGetLastError();
#elif defined(PADDLE_WITH_CUDA)
    auto err = cudaHostGetDevicePointer(&mapped_base, holder->ptr(), 0);
    if (err == cudaSuccess && mapped_base != nullptr) {
      return static_cast<char*>(mapped_base) + dense->meta().offset;
    }
    (void)cudaGetLastError();
#endif
  }
#endif

  return const_cast<void*>(tensor.data());
}

}  // namespace compat
