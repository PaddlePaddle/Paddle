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

#include "paddle/phi/kernels/memcpy_kernel.h"

#include <vector>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

static constexpr size_t WAIT_THRESHOLD = 64 * 1024;

template <>
void MemcpyH2DKernel(const GPUContext& dev_ctx,
                     const DenseTensor& x,
                     int dst_place_type,
                     DenseTensor* out) {
  PADDLE_ENFORCE_GE(
      dst_place_type,
      0,
      errors::OutOfRange("dst_place_type only support 0-3, but got: %d",
                         dst_place_type));
  PADDLE_ENFORCE_LE(
      dst_place_type,
      3,
      errors::OutOfRange("dst_place_type only support 0-3, but got: %d",
                         dst_place_type));

  auto stream = static_cast<const phi::GPUContext*>(&dev_ctx)->stream();
  out->mutable_data(dev_ctx.GetPlace(),
                    x.dtype(),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
}

template <typename Context>
void MemcpyH2DKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int dst_place_type,
                     DenseTensor* out) {
  PADDLE_ENFORCE_GE(
      dst_place_type,
      0,
      errors::OutOfRange("dst_place_type only support 0-3, but got: %d",
                         dst_place_type));
  PADDLE_ENFORCE_LE(
      dst_place_type,
      3,
      errors::OutOfRange("dst_place_type only support 0-3, but got: %d",
                         dst_place_type));

  Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
}

template <typename Context>
void MemcpyD2HKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int dst_place_type,
                     DenseTensor* out) {
  switch (dst_place_type) {
    case 0:
      out->mutable_data(CPUPlace());
      Copy(dev_ctx, x, CPUPlace(), false, out);
      // NOTE(copy from Aurelius84): host <-> device memory copies of a memory
      // block of 64 KB or less are asynchronous. See
      // https://forums.developer.nvidia.com/t/host-device-memory-copies-up-to-64-kb-are-asynchronous/17907
      if (x.memory_size() <= WAIT_THRESHOLD) {
        dev_ctx.Wait();
      }
      break;

    case 1:
      out->mutable_data(GPUPinnedPlace());
      Copy(dev_ctx, x, GPUPinnedPlace(), false, out);
      // paddle::memory::Copy use async copy for GPUPinnedPlace
      dev_ctx.Wait();
      break;

    default:
      PADDLE_THROW(errors::InvalidArgument(
          "Arugment 'dst_place_type' only support 0-1, but got: %d",
          dst_place_type));
      break;
  }
}

template <typename Context>
void MemcpyD2HMultiIOKernel(const Context& dev_ctx,
                            const std::vector<const DenseTensor*>& array,
                            int dst_place_type,
                            std::vector<DenseTensor*> out_array) {
  PADDLE_ENFORCE_EQ(
      array.size(),
      out_array.size(),
      errors::PreconditionNotMet(
          "input size %d != output size %d", array.size(), out_array.size()));

  for (size_t i = 0; i < array.size(); i++) {
    PADDLE_ENFORCE_NOT_NULL(
        array[i],
        errors::PreconditionNotMet("input tesnor %d should not be nullptr", i));
    PADDLE_ENFORCE_NOT_NULL(
        out_array[i],
        errors::PreconditionNotMet("input tesnor %d should not be nullptr", i));

    const auto& x = *(array[i]);
    MemcpyD2HKernel<Context>(dev_ctx, x, dst_place_type, out_array[i]);
  }
}

}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(memcpy_h2d,
                           CPU,
                           ALL_LAYOUT,
                           phi::MemcpyH2DKernel<phi::CPUContext>,
                           ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(memcpy_d2h,
                           CPU,
                           ALL_LAYOUT,
                           phi::MemcpyD2HKernel<phi::CPUContext>,
                           ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(memcpy_d2h_multi_io,
                           CPU,
                           ALL_LAYOUT,
                           phi::MemcpyD2HMultiIOKernel<phi::CPUContext>,
                           ALL_DTYPE) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(memcpy_h2d,
                           GPU,
                           ALL_LAYOUT,
                           phi::MemcpyH2DKernel<phi::GPUContext>,
                           ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(memcpy_d2h,
                           GPU,
                           ALL_LAYOUT,
                           phi::MemcpyD2HKernel<phi::GPUContext>,
                           ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(memcpy_d2h_multi_io,
                           GPU,
                           ALL_LAYOUT,
                           phi::MemcpyD2HMultiIOKernel<phi::GPUContext>,
                           ALL_DTYPE) {}

#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_GENERAL_KERNEL(memcpy_h2d,
                           XPU,
                           ALL_LAYOUT,
                           phi::MemcpyH2DKernel<phi::XPUContext>,
                           ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(memcpy_d2h,
                           XPU,
                           ALL_LAYOUT,
                           phi::MemcpyD2HKernel<phi::XPUContext>,
                           ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(memcpy_d2h_multi_io,
                           XPU,
                           ALL_LAYOUT,
                           phi::MemcpyD2HMultiIOKernel<phi::XPUContext>,
                           ALL_DTYPE) {}

#endif
