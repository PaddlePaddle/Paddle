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
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_factory.h"
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
}  // namespace phi
