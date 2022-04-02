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

#include "paddle/phi/kernels/copy_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"

namespace phi {

template <typename Context>
void Copy(const Context& dev_ctx,
          const DenseTensor& src,
          Place dst_place,
          bool blocking,
          DenseTensor* dst) {
  auto* src_ptr = src.data();
  const auto& src_place = src.place();

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;

  dst->Resize(src.dims());

  void* dst_ptr = nullptr;
  if (paddle::platform::is_cpu_place(dst_place)) {
    dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
  } else if (paddle::platform::is_cuda_pinned_place(dst_place)) {
    // now we only can use mutable_data to Alloc pinned memory here,
    // dev_ctx can not alloc pinned memory now
    dst_ptr = dst->mutable_data(dst_place, src.dtype());
  } else {
    dst_ptr = dev_ctx.Alloc(
        dst, src.dtype(), 0, paddle::platform::is_cuda_pinned_place(dst_place));
  }

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << dst_place;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

  CHECK(dst->layout() == src.layout());

  auto size = src.numel() * paddle::experimental::SizeOf(src.dtype());

  if ((paddle::platform::is_cpu_place(src_place) ||
       paddle::platform::is_cuda_pinned_place(src_place)) &&  // NOLINT
      (paddle::platform::is_cpu_place(dst_place) ||
       paddle::platform::is_cuda_pinned_place(dst_place))) {
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  } else if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
             paddle::platform::is_cpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_cpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        phi::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(src_gpu_place,
                      ctx_gpu_place,
                      phi::errors::Unavailable(
                          "Source place and context place do not match, source "
                          "place is %s, context place is %s.",
                          src_gpu_place,
                          ctx_gpu_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
    paddle::memory::Copy(
        dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  } else if ((paddle::platform::is_cpu_place(src_place) ||
              paddle::platform::is_cuda_pinned_place(src_place)) &&  // NOLINT
             paddle::platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        phi::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(dst_gpu_place,
                      ctx_gpu_place,
                      phi::errors::Unavailable(
                          "Destination place and context place do not match, "
                          "destination place is %s, context place is %s.",
                          dst_gpu_place,
                          ctx_gpu_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
    paddle::memory::Copy(
        dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);
  } else if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
             paddle::platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        phi::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
    if (paddle::platform::is_same_place(src_place, dst_place)) {
      paddle::memory::Copy(
          dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
    } else {
      if (paddle::platform::is_same_place(ctx_place, src_place)) {
        paddle::memory::Copy(
            dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
        paddle::platform::DeviceContextPool::Instance()
            .Get(src.place())
            ->Wait();
      } else if (paddle::platform::is_same_place(ctx_place, dst_place)) {
        paddle::platform::DeviceContextPool::Instance()
            .Get(src.place())
            ->Wait();
        paddle::memory::Copy(
            dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
      } else {
        PADDLE_THROW(phi::errors::Unavailable(
            "Context place dose not match the source and destination place."));
      }
    }
  } else if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
             paddle::platform::is_cuda_pinned_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_cuda_pinned_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        phi::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(src_gpu_place,
                      ctx_gpu_place,
                      phi::errors::Unavailable(
                          "Source place and context place do not match, source "
                          "place is %s, context place is %s.",
                          src_gpu_place,
                          ctx_gpu_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
    paddle::memory::Copy(
        dst_cuda_pinned_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Place type error. Please check the place of src and dst Tensor."));
  }
}

}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(
    copy, GPU, ALL_LAYOUT, phi::Copy<phi::GPUContext>, ALL_DTYPE) {}
