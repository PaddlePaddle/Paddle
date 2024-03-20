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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/fill_kernel.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"

namespace phi {
template <typename T, typename Context>
inline void StridedTensorCopy(const Context& dev_ctx,
                              const phi::DenseTensor& input,
                              const std::vector<int64_t>& dims,
                              const std::vector<int64_t>& out_stride,
                              int64_t offset,
                              phi::DenseTensor* out) {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
  phi::StridedCopyKernel<T, Context>(
      dev_ctx, input, dims, out_stride, offset, out);

#else
  const phi::KernelKey& strided_copy_key = {
      phi::TransToPhiBackend(dev_ctx.GetPlace()),
      phi::DataLayout::ALL_LAYOUT,
      input.dtype()};
  using strided_copy_signature = void (*)(const phi::DeviceContext&,
                                          const phi::DenseTensor&,
                                          const std::vector<int64_t>&,
                                          const std::vector<int64_t>&,
                                          int64_t,
                                          phi::DenseTensor*);
  PD_VISIT_KERNEL("strided_copy",
                  strided_copy_key,
                  strided_copy_signature,
                  false,
                  dev_ctx,
                  input,
                  dims,
                  out_stride,
                  offset,
                  out);
#endif
}

template <typename T, typename Context>
inline void StridedTensorFill(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const phi::Scalar& value,
                              phi::DenseTensor* out) {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
  phi::FillKernel<T, Context>(dev_ctx, x, value, out);
#else
  const phi::KernelKey& fill_key = {phi::TransToPhiBackend(dev_ctx.GetPlace()),
                                    phi::DataLayout::ALL_LAYOUT,
                                    x.dtype()};
  using fill_signature = void (*)(const phi::DeviceContext&,
                                  const phi::DenseTensor&,
                                  const phi::Scalar&,
                                  phi::DenseTensor*);

  PD_VISIT_KERNEL(
      "fill", fill_key, fill_signature, false, dev_ctx, x, value, out);
#endif
}

template <typename T, typename Context>
inline void StridedTensorContiguous(const Context& dev_ctx,
                                    const phi::DenseTensor& input,
                                    phi::DenseTensor* out) {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
  phi::ContiguousKernel<T, Context>(dev_ctx, input, out);
#else
  const phi::KernelKey& contiguous_key = {
      phi::TransToPhiBackend(dev_ctx.GetPlace()),
      phi::DataLayout::ALL_LAYOUT,
      input.dtype()};
  using contiguous_signature = void (*)(
      const phi::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);

  PD_VISIT_KERNEL("contiguous",
                  contiguous_key,
                  contiguous_signature,
                  false,
                  dev_ctx,
                  input,
                  out);
#endif
}
}  // namespace phi
