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

#include "paddle/phi/core/tensor_utils.h"

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

  if (&src == dst) {
    if (paddle::platform::is_same_place(src_place, dst_place)) {
      VLOG(6) << "Skip copy the same data(" << src_ptr << ") from " << src_place
              << " to " << dst_place;
    } else {
      VLOG(6) << "Src and dst are the same Tensor, in-place copy data("
              << src_ptr << ") from " << src_place << " to " << dst_place;
      const DenseTensor src_copy = src;
      Copy(dev_ctx, src_copy, dst_place, blocking, dst);
    }
    return;
  }

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;

  dst->Resize(src.dims());

  void* dst_ptr = nullptr;
  if (paddle::platform::is_cpu_place(dst_place)) {
    dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (paddle::platform::is_gpu_place(dst_place) ||
             paddle::platform::is_cuda_pinned_place(dst_place)) {
    dst_ptr = dev_ctx.Alloc(
        dst, src.dtype(), 0, paddle::platform::is_cuda_pinned_place(dst_place));
#endif

#ifdef PADDLE_WITH_XPU
  } else if (paddle::platform::is_xpu_place(dst_place)) {
    dst_ptr = dev_ctx.Alloc(dst, src.dtype());
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (paddle::platform::is_custom_place(dst_place)) {
    dst_ptr = dev_ctx.Alloc(dst, src.dtype());
#endif
  }

  auto size = src.numel() * paddle::experimental::SizeOf(src.dtype());
  if (UNLIKELY(size) == 0) {
    return;
  }

  PADDLE_ENFORCE_EQ(
      dst->place(),
      dst_place,
      phi::errors::Unavailable(
          "The Dst Tensor's place and dst_place do not match, Tensor's place "
          "place is %s, dst_place is %s.",
          dst->place(),
          dst_place));

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << dst_place;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;
  CHECK(dst->layout() == src.layout());

  if (paddle::platform::is_cpu_place(src_place) &&
      paddle::platform::is_cpu_place(dst_place)) {
    paddle::memory::Copy(src_place, dst_ptr, src_place, src_ptr, size);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if ((paddle::platform::is_cpu_place(src_place) ||
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
#endif
#ifdef PADDLE_WITH_XPU
  } else if (paddle::platform::is_xpu_place(src_place) &&  // NOLINT
             paddle::platform::is_cpu_place(dst_place)) {
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else if (paddle::platform::is_cpu_place(src_place) &&
             paddle::platform::is_xpu_place(dst_place)) {
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else if (paddle::platform::is_xpu_place(src_place) &&
             paddle::platform::is_xpu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (paddle::platform::is_custom_place(src_place) &&  // NOLINT
             paddle::platform::is_cpu_place(dst_place)) {
    auto stream =
        blocking
            ? nullptr
            : reinterpret_cast<const paddle::platform::CustomDeviceContext&>(
                  dev_ctx)
                  .stream();
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  } else if (paddle::platform::is_cpu_place(src_place) &&  // NOLINT
             paddle::platform::is_custom_place(dst_place)) {
    auto stream =
        blocking
            ? nullptr
            : reinterpret_cast<const paddle::platform::CustomDeviceContext&>(
                  dev_ctx)
                  .stream();
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  } else if (paddle::platform::is_custom_place(src_place) &&  // NOLINT
             paddle::platform::is_custom_place(dst_place)) {
    auto stream =
        blocking
            ? nullptr
            : reinterpret_cast<const paddle::platform::CustomDeviceContext&>(
                  dev_ctx)
                  .stream();
    paddle::memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
#endif
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
}

template <typename Context>
void Copy(const Context& dev_ctx,
          const SelectedRows& src,
          Place dst_place,
          bool blocking,
          SelectedRows* dst) {
  if (src.value().Holder() != dst->value().Holder() ||
      src.value().data() != dst->value().data()) {
    dst->set_rows(src.rows());
    dst->set_height(src.height());
  }
  Copy<Context>(
      dev_ctx, src.value(), dst_place, blocking, dst->mutable_value());
}

template <typename Context>
void Copy(const Context& dev_ctx,
          const SparseCooTensor& src,
          Place dst_place,
          bool blocking,
          SparseCooTensor* dst) {
  phi::Copy<Context>(dev_ctx,
                     src.non_zero_indices(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_indices());

  phi::Copy<Context>(dev_ctx,
                     src.non_zero_elements(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_elements());
  dst->set_dims(src.dims());
  dst->SetCoalesced(src.coalesced());
}

template <typename Context>
void Copy(const Context& dev_ctx,
          const SparseCsrTensor& src,
          Place dst_place,
          bool blocking,
          SparseCsrTensor* dst) {
  phi::Copy<Context>(dev_ctx,
                     src.non_zero_crows(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_crows());

  phi::Copy<Context>(dev_ctx,
                     src.non_zero_cols(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_cols());

  phi::Copy<Context>(dev_ctx,
                     src.non_zero_elements(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_elements());
  dst->set_dims(src.dims());
}

template void Copy(const CPUContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);

template void Copy(const DeviceContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);

template void Copy(const CPUContext& dev_ctx,
                   const SelectedRows& src,
                   Place dst_place,
                   bool blocking,
                   SelectedRows* dst);
template void Copy(const DeviceContext& dev_ctx,
                   const SelectedRows& src,
                   Place dst_place,
                   bool blocking,
                   SelectedRows* dst);

template void Copy(const CPUContext& dev_ctx,
                   const SparseCooTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCooTensor* dst);

template void Copy(const DeviceContext& dev_ctx,
                   const SparseCooTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCooTensor* dst);

template void Copy(const CPUContext& dev_ctx,
                   const SparseCsrTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCsrTensor* dst);

template void Copy(const DeviceContext& dev_ctx,
                   const SparseCsrTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCsrTensor* dst);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template void Copy(const GPUContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);
template void Copy(const GPUContext& dev_ctx,
                   const SelectedRows& src,
                   Place dst_place,
                   bool blocking,
                   SelectedRows* dst);
template void Copy(const GPUContext& dev_ctx,
                   const SparseCooTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCooTensor* dst);
template void Copy(const GPUContext& dev_ctx,
                   const SparseCsrTensor& src,
                   Place dst_place,
                   bool blocking,
                   SparseCsrTensor* dst);
#endif

#ifdef PADDLE_WITH_XPU
template void Copy(const XPUContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template void Copy(const CustomContext& dev_ctx,
                   const DenseTensor& src,
                   Place dst_place,
                   bool blocking,
                   DenseTensor* dst);
#endif
}  // namespace phi
