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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/stream.h"

namespace phi {

static constexpr size_t WAIT_THRESHOLD = 64 * 1024;

template <typename Context>
void MemcpyH2DKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int dst_place_type,
                     DenseTensor* out) {
  if (!x.initialized()) {
    out->set_meta(x.meta());
    return;
  }

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
      Copy(dev_ctx, x, CPUPlace(), false, out);
      // NOTE(copy from Aurelius84): host <-> device memory copies of a memory
      // block of 64 KB or less are asynchronous. See
      // https://forums.developer.nvidia.com/t/host-device-memory-copies-up-to-64-kb-are-asynchronous/17907
      if (x.memory_size() <= WAIT_THRESHOLD) {
        dev_ctx.Wait();
      }
      break;

    case 1:
      Copy(dev_ctx, x, GPUPinnedPlace(), false, out);
      // Copy use async copy for GPUPinnedPlace
      dev_ctx.Wait();
      break;

    default:
      PADDLE_THROW(errors::InvalidArgument(
          "Argument 'dst_place_type' only support 0-1, but got: %d",
          dst_place_type));
      break;
  }
}

template <typename Context>
void MemcpyD2HMultiIOKernel(const Context& dev_ctx,
                            const TensorArray& array,
                            int dst_place_type,
                            TensorArray* out_array) {
  PADDLE_ENFORCE_NOT_NULL(
      out_array,
      errors::PreconditionNotMet("output tensor_array should not be nullptr"));

  out_array->clear();
  out_array->resize(array.size());
  for (size_t i = 0; i < array.size(); i++) {
    const auto& x = array[i];
    MemcpyD2HKernel<Context>(dev_ctx, x, dst_place_type, &(out_array->at(i)));
  }
}

template <typename Context>
void MemcpyKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int dst_place_type,
                  DenseTensor* out) {
  if (!x.IsInitialized()) {
    return;
  }
  PADDLE_ENFORCE_GE(
      dst_place_type,
      0,
      errors::OutOfRange("dst_place_type only support 0-4, but got: %d",
                         dst_place_type));
  PADDLE_ENFORCE_LE(
      dst_place_type,
      4,
      errors::OutOfRange("dst_place_type only support 0-4, but got: %d",
                         dst_place_type));
  switch (dst_place_type) {
    case 0: /* CPUPlace */
      dev_ctx.HostAlloc(out, out->dtype());
      Copy(dev_ctx, x, CPUPlace(), true, out);
      break;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case 1: /* CUDAPlace */
      dev_ctx.Alloc(out, x.dtype());
      Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
      break;
    case 2: /* CUDAPinnedPlace */
      dev_ctx.Alloc(out, x.dtype(), 0, true);
      Copy(dev_ctx, x, GPUPinnedPlace(), false, out);
      break;
#elif defined(PADDLE_WITH_XPU)
    case 3:  // XPUPlace
      dev_ctx.Alloc(out, x.dtype());
      Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
      break;
#elif defined(PADDLE_WITH_CUSTOM_DEVICE)
    case 4:  // CustomPlace
      dev_ctx.Alloc(out, x.dtype());
      Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
      break;
#endif
    default:
      PADDLE_THROW(errors::Unimplemented(
          "memcpy dst_place_type: %d is not supported yet.", dst_place_type));
      break;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(memcpy_h2d,
                                         ALL_LAYOUT,
                                         phi::MemcpyH2DKernel) {}
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(memcpy_d2h,
                                         ALL_LAYOUT,
                                         phi::MemcpyD2HKernel) {
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
}
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(memcpy_d2h_multi_io,
                                         ALL_LAYOUT,
                                         phi::MemcpyD2HMultiIOKernel) {
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
}
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(memcpy,
                                         ALL_LAYOUT,
                                         phi::MemcpyKernel) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
