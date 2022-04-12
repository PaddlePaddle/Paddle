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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/tensor_meta.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"

namespace phi {

class DenseTensorUtils {
 public:
  static DenseTensorMeta* GetMutableMeta(DenseTensor* tensor) {
    return &(tensor->meta_);
  }

  static const std::shared_ptr<phi::Allocation>& GetHolder(
      const DenseTensor& tensor) {
    return tensor.holder_;
  }

  static DenseTensor Slice(const DenseTensor& tensor,
                           int64_t begin_idx,
                           int64_t end_idx) {
    size_t bytes = tensor.numel() * SizeOf(tensor.dtype());
    PADDLE_ENFORCE_GE(tensor.capacity(),
                      bytes,
                      phi::errors::InvalidArgument(
                          "The memory size %d should be enough to meet the "
                          "volume required by metadata %d.",
                          tensor.capacity(),
                          bytes));
    PADDLE_ENFORCE_GE(
        begin_idx,
        0,
        phi::errors::OutOfRange("The start row index must be greater than 0."
                                "But received the start index is d%.",
                                begin_idx));
    PADDLE_ENFORCE_LE(
        end_idx,
        tensor.dims()[0],
        phi::errors::OutOfRange("The end row index is out of bound."));
    PADDLE_ENFORCE_LT(
        begin_idx,
        end_idx,
        phi::errors::InvalidArgument(
            "The start row index must be less than the end row index."
            "But received the start index = %d, the end index = %d.",
            begin_idx,
            end_idx));
    DenseTensor ret(tensor);
    if (tensor.dims()[0] != 1) {
      ret.meta_.dims[0] = end_idx - begin_idx;
      ret.meta_.offset = tensor.meta_.offset +
                         begin_idx * (tensor.numel() / tensor.dims()[0]) *
                             paddle::experimental::SizeOf(tensor.dtype());
    }
    return ret;
  }
};

template <typename Context>
void Copy(const Context& dev_ctx,
          const DenseTensor& src,
          Place dst_place,
          bool blocking,
          DenseTensor* dst) {
  auto* src_ptr = src.data();
  const auto& src_place = src.place();

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << src_place;

  dst->Resize(src.dims());

  void* dst_ptr = nullptr;
  if (paddle::platform::is_cpu_place(dst_place)) {
    dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
  } else {
    dst_ptr = dev_ctx.Alloc(
        dst, src.dtype(), 0, paddle::platform::is_cuda_pinned_place(dst_place));
  }

  if (src_ptr == dst_ptr) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << src_place;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;
  CHECK(dst->layout() == src.layout());

  auto size = src.numel() * paddle::experimental::SizeOf(src.dtype());

  if (paddle::platform::is_cpu_place(src_place) &&
      paddle::platform::is_cpu_place(dst_place)) {
    paddle::memory::Copy(src_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if ((paddle::platform::is_cpu_place(src_place) ||
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
  }
#endif

#ifdef PADDLE_WITH_XPU
  else if (paddle::platform::is_xpu_place(src_place) &&  // NOLINT
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
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
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

}  // namespace phi
