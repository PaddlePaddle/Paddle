/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/tensor_util.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/core/dense_tensor.h"

#ifdef PADDLE_WITH_MKLDNN
#include "dnnl_debug.h"  // NOLINT
#endif

namespace paddle {
namespace framework {

template <typename TENSOR>
void TensorCopyImpl(const TENSOR& src,
                    const platform::Place& dst_place,
                    const platform::DeviceContext& ctx,
                    TENSOR* dst) {
  if (&src == dst) {
    auto src_copy = src;
    TensorCopyImpl(src_copy, dst_place, ctx, dst);
    return;
  }

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;
  src.check_memory_size();
  dst->Resize(src.dims());
  dst->set_layout(src.layout());
  auto src_place = src.place();
  auto src_ptr = src.data();
#ifdef PADDLE_WITH_MKLDNN
  dst->set_mem_desc(src.mem_desc());
  // oneDNN tensors due to padding may be of bigger size
  // than numel()*size(type())
  auto dst_ptr =
      src.layout() == DataLayout::kMKLDNN
          ? dst->mutable_data(dst_place, src.dtype(), src.memory_size())
          : dst->mutable_data(dst_place, src.dtype());
#else
  auto dst_ptr = dst->mutable_data(dst_place, src.dtype());
#endif
  dst->set_layout(src.layout());
  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << dst_place;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

#ifdef PADDLE_WITH_MKLDNN
  auto size = src.layout() == DataLayout::kMKLDNN
                  ? src.memory_size()
                  : src.numel() * framework::DataTypeSize(src.dtype());
#else
  auto size = src.numel() * framework::DataTypeSize(src.dtype());
#endif

  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (platform::is_custom_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    auto stream =
        reinterpret_cast<const platform::CustomDeviceContext&>(ctx).stream();
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  } else if (platform::is_cpu_place(src_place) &&  // NOLINT
             platform::is_custom_place(dst_place)) {
    auto stream =
        reinterpret_cast<const platform::CustomDeviceContext&>(ctx).stream();
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  } else if (platform::is_custom_place(src_place) &&  // NOLINT
             platform::is_custom_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    auto stream =
        reinterpret_cast<const platform::CustomDeviceContext&>(ctx).stream();
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (platform::is_xpu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_xpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else if (platform::is_xpu_place(src_place) &&
             platform::is_xpu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  // TODO(zhiqiu): handle different condition like CUDA code below
  else if (platform::is_npu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    auto stream =
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream();
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  }
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_npu_place(dst_place)) {
    //  1. cpu tensor -> npu pinned tensor
    platform::NPUPinnedPlace npu_pinned_place;
    phi::DenseTensor npu_pinned_tensor;
    npu_pinned_tensor.Resize(src.dims());
    auto npu_pinned_ptr =
        npu_pinned_tensor.mutable_data(npu_pinned_place, src.dtype());
    memory::Copy(npu_pinned_place, npu_pinned_ptr, src_place, src_ptr, size);

    //  2. async copy npu pinned tensor -> npu tensor
    memory::Copy(
        dst_place,
        dst_ptr,
        npu_pinned_place,
        npu_pinned_ptr,
        size,
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream());

    //  3. record event
    auto npu_pinned_allocator =
        static_cast<paddle::memory::allocation::NPUPinnedAllocator*>(
            paddle::memory::allocation::AllocatorFacade::Instance()
                .GetAllocator(npu_pinned_place)
                .get());
    phi::Allocation* allocation = npu_pinned_tensor.Holder().get();
    npu_pinned_allocator->RecordEvent(
        allocation,
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream());
  }
  else if (platform::is_npu_place(src_place) &&  // NOLINT
           platform::is_npu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    auto stream =
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream();
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, stream);
  }
  else if (platform::is_npu_pinned_place(src_place) &&  // NOLINT
           platform::is_npu_place(dst_place)) {         /* npu_pinned->npu */
    auto src_npu_pinned_place = src_place;
    auto dst_npu_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        platform::is_npu_place(ctx_place),
        true,
        platform::errors::PreconditionNotMet(
            "Device context place mismatch. When copying phi::DenseTensor "
            "data from NPU Pinned memory to NPU memory, current "
            "device context place should be NPU."));
    auto ctx_npu_place = ctx_place;
    PADDLE_ENFORCE_EQ(dst_npu_place,
                      ctx_npu_place,
                      platform::errors::PreconditionNotMet(
                          "The target NPU device and current device context do "
                          "not match. The target NPU device number is %d, but "
                          "device context NPU number is %d.",
                          dst_npu_place.device,
                          ctx_npu_place.device));
    auto stream =
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream();
    memory::Copy(
        dst_npu_place, dst_ptr, src_npu_pinned_place, src_ptr, size, stream);
  }
  else if (platform::is_npu_place(src_place) &&        // NOLINT
           platform::is_npu_pinned_place(dst_place)) { /* npu->npu_pinned */
    auto src_npu_place = src_place;
    auto dst_npu_pinned_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        platform::is_npu_place(ctx_place),
        true,
        platform::errors::PreconditionNotMet(
            "Device context place mismatch. When copying phi::DenseTensor "
            "data from NPU memory to NPU Pinned memory, current "
            "device context place should be NPU."));
    auto ctx_npu_place = ctx_place;
    PADDLE_ENFORCE_EQ(src_place,
                      ctx_npu_place,
                      platform::errors::PreconditionNotMet(
                          "The source NPU device and current device context do "
                          "not match. The source NPU device number is %d, but "
                          "device context NPU number is %d.",
                          src_npu_place.device,
                          ctx_npu_place.device));
    auto stream =
        reinterpret_cast<const platform::NPUDeviceContext&>(ctx).stream();
    memory::Copy(
        dst_npu_pinned_place, dst_ptr, src_npu_place, src_ptr, size, stream);
  }
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (platform::is_cuda_pinned_place(src_place) &&  // NOLINT
           platform::is_cuda_pinned_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (platform::is_cuda_pinned_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_cuda_pinned_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_cpu_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx_place),
        true,
        platform::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(src_gpu_place,
                      ctx_gpu_place,
                      platform::errors::Unavailable(
                          "Source place and context place do not match, source "
                          "place is %s, context place is %s.",
                          src_gpu_place,
                          ctx_gpu_place));
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    memory::Copy(dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  }
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx_place),
        true,
        platform::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(dst_gpu_place,
                      ctx_gpu_place,
                      platform::errors::Unavailable(
                          "Destination place and context place do not match, "
                          "destination place is %s, context place is %s.",
                          dst_gpu_place,
                          ctx_gpu_place));
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    memory::Copy(dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);
  }
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_cuda_pinned_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_cuda_pinned_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx_place),
        true,
        platform::errors::PreconditionNotMet(
            "Device context place mismatch. When copying phi::DenseTensor "
            "data from GPU memory to CUDA Pinned memory, current "
            "device context place should be GPU."));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(src_gpu_place,
                      ctx_gpu_place,
                      platform::errors::PreconditionNotMet(
                          "The source GPU device and current device context do "
                          "not match. The source GPU device number is %d, but "
                          "device context GPU number is %d.",
                          src_gpu_place.device,
                          ctx_gpu_place.device));
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    memory::Copy(
        dst_cuda_pinned_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  }
  else if (platform::is_cuda_pinned_place(src_place) &&  // NOLINT
           platform::is_gpu_place(dst_place)) {
    auto src_cuda_pinned_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx_place),
        true,
        platform::errors::PreconditionNotMet(
            "Device context place mismatch. When copying phi::DenseTensor "
            "data from CUDA Pinned memory to GPU memory, current "
            "device context place should be GPU."));
    auto ctx_gpu_place = ctx_place;
    PADDLE_ENFORCE_EQ(dst_gpu_place,
                      ctx_gpu_place,
                      platform::errors::PreconditionNotMet(
                          "The target GPU device and current device context do "
                          "not match. The target GPU device number is %d, but "
                          "device context GPU number is %d.",
                          dst_gpu_place.device,
                          ctx_gpu_place.device));
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    memory::Copy(
        dst_gpu_place, dst_ptr, src_cuda_pinned_place, src_ptr, size, stream);
  }
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx_place),
        true,
        platform::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    if (platform::is_same_place(src_place, dst_place)) {
      memory::Copy(
          dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
    } else {
      if (platform::is_same_place(ctx_place, src_place)) {
        memory::Copy(
            dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
        platform::DeviceContextPool::Instance().Get(src.place())->Wait();
      } else if (platform::is_same_place(ctx_place, dst_place)) {
        platform::DeviceContextPool::Instance().Get(src.place())->Wait();
        memory::Copy(
            dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
      } else {
        PADDLE_THROW(platform::errors::Unavailable(
            "Context place dose not match the source and destination place."));
      }
    }
  }
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "Copying from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#ifdef PADDLE_WITH_MLU
  else if (platform::is_mlu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    auto src_mlu_place = src_place;
    auto dst_cpu_place = dst_place;
    auto stream =
        reinterpret_cast<const platform::MLUDeviceContext&>(ctx).stream();
    memory::Copy(dst_cpu_place, dst_ptr, src_mlu_place, src_ptr, size, stream);
  }
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_mlu_place(dst_place)) {
    auto src_cpu_place = src_place;
    auto dst_mlu_place = dst_place;
    auto stream =
        reinterpret_cast<const platform::MLUDeviceContext&>(ctx).stream();
    memory::Copy(dst_mlu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);
  }
  else if (platform::is_mlu_place(src_place) &&  // NOLINT
           platform::is_mlu_place(dst_place)) {
    auto src_mlu_place = src_place;
    auto dst_mlu_place = dst_place;
    auto stream =
        reinterpret_cast<const platform::MLUDeviceContext&>(ctx).stream();
    memory::Copy(dst_mlu_place, dst_ptr, src_mlu_place, src_ptr, size, stream);
  }
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "Copying from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#ifdef PADDLE_WITH_IPU
  else if (platform::is_ipu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_ipu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (platform::is_ipu_place(src_place) &&  // NOLINT
           platform::is_ipu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data sync from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "Copying from %s to %s is not supported.", src_place, dst_place));
  }
#endif
}

template <typename TENSOR>
void TensorCopyImpl(const TENSOR& src,
                    const platform::Place& dst_place,
                    TENSOR* dst) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  const platform::DeviceContext* dev_ctx;
  if (platform::is_gpu_place(dst_place) || platform::is_npu_place(dst_place) ||
      platform::is_mlu_place(dst_place) ||
      platform::is_custom_place(dst_place)) {
    dev_ctx = pool.Get(dst_place);
  } else {
    dev_ctx = pool.Get(src.place());
  }
  TensorCopyImpl(src, dst_place, *dev_ctx, dst);
}

void TensorCopy(const phi::DenseTensor& src,
                const platform::Place& dst_place,
                phi::DenseTensor* dst) {
  TensorCopyImpl<phi::DenseTensor>(src, dst_place, dst);
}
void TensorCopy(const phi::DenseTensor& src,
                const platform::Place& dst_place,
                const platform::DeviceContext& ctx,
                phi::DenseTensor* dst) {
  TensorCopyImpl<phi::DenseTensor>(src, dst_place, ctx, dst);
}

void TensorCopySync(const phi::DenseTensor& src,
                    const platform::Place& dst_place,
                    phi::DenseTensor* dst) {
  if (&src == dst) {
    auto src_copy = src;
    TensorCopySync(src_copy, dst_place, dst);
    return;
  }

  VLOG(3) << "TensorCopySync " << src.dims() << " from " << src.place()
          << " to " << dst_place;
  src.check_memory_size();
  dst->Resize(src.dims());
  dst->set_layout(src.layout());
#ifdef PADDLE_WITH_MKLDNN
  dst->set_format(src.format());
#endif
  auto src_place = src.place();
  auto src_ptr = src.data();
  auto dst_ptr = dst->mutable_data(dst_place, src.dtype());
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data from " << src_place << " to "
            << dst_place;
    return;
  }

  auto size = src.numel() * framework::DataTypeSize(src.dtype());
  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  else if (platform::is_custom_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {     /* custom_device -> cpu*/
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }                                                // NOLINT
  else if (platform::is_cpu_place(src_place) &&    // NOLINT
           platform::is_custom_place(dst_place)) { /* cpu -> custom_device*/
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }                                                 // NOLINT
  else if (platform::is_custom_place(src_place) &&  // NOLINT
           platform::is_custom_place(
               dst_place)) { /* custom_device -> custom_device*/
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data sync from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
#endif
#ifdef PADDLE_WITH_XPU
  else if (platform::is_xpu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }                                              // NOLINT
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_xpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }                                              // NOLINT
  else if (platform::is_xpu_place(src_place) &&  // NOLINT
           platform::is_xpu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
    platform::XPUPlace xpu_dst_place = dst_place;
    platform::XPUPlace xpu_src_place = src_place;
    if (xpu_dst_place.device == xpu_src_place.device) {
      auto xpu_ctx = platform::DeviceContextPool::Instance().Get(xpu_dst_place);
      xpu_ctx->Wait();
    }
  }       // NOLINT
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  else if (platform::is_npu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {  /* npu -> cpu*/
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_npu_place(dst_place)) {  /* cpu -> npu*/
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
  else if (platform::is_npu_place(src_place) &&  // NOLINT
           platform::is_npu_place(dst_place)) {  /* npu -> npu*/
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data sync from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  else if (platform::is_cuda_pinned_place(src_place) &&  // NOLINT
           platform::is_cuda_pinned_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (platform::is_cuda_pinned_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_cuda_pinned_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_cuda_pinned_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_cpu_place = dst_place;
    memory::Copy(dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, nullptr);
  }
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = src_place;
    auto dst_gpu_place = dst_place;
    memory::Copy(dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, nullptr);
  }
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = src_place;
    auto dst_gpu_place = dst_place;
    memory::Copy(dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size, nullptr);
  }
  else if (platform::is_cuda_pinned_place(src_place) &&  // NOLINT
           platform::is_gpu_place(dst_place)) {
    auto src_pinned_place = src_place;
    auto dst_gpu_place = dst_place;
    memory::Copy(
        dst_gpu_place, dst_ptr, src_pinned_place, src_ptr, size, nullptr);
  }
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#ifdef PADDLE_WITH_MLU
  else if (platform::is_mlu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_mlu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
  else if (platform::is_mlu_place(src_place) &&  // NOLINT
           platform::is_mlu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data async from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size, nullptr);
  }
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
#ifdef PADDLE_WITH_IPU
  else if (platform::is_ipu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (platform::is_cpu_place(src_place) &&  // NOLINT
           platform::is_ipu_place(dst_place)) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else if (platform::is_ipu_place(src_place) &&  // NOLINT
           platform::is_ipu_place(dst_place)) {
    if (src_ptr == dst_ptr) {
      VLOG(3) << "Skip copy the same data sync from " << src_place << " to "
              << dst_place;
      return;
    }
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
  else {  // NOLINT
    PADDLE_THROW(platform::errors::Unimplemented(
        "Copy from %s to %s is not supported.", src_place, dst_place));
  }
#endif
}

void TensorToStream(std::ostream& os,
                    const phi::DenseTensor& tensor,
                    const platform::DeviceContext& dev_ctx) {
  {  // the 1st field, uint32_t version
    constexpr uint32_t version = 0;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  }
  {  // the 2nd field, tensor description
     // int32_t  size
     // void*    protobuf message
    proto::VarType::TensorDesc desc;
    desc.set_data_type(framework::TransToProtoVarType(tensor.dtype()));
    auto dims = phi::vectorize(tensor.dims());
    auto* pb_dims = desc.mutable_dims();
    pb_dims->Resize(static_cast<int>(dims.size()), 0);
    std::copy(dims.begin(), dims.end(), pb_dims->begin());
    int32_t size = desc.ByteSize();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    auto out = desc.SerializeAsString();
    os.write(out.data(), size);
  }
  {  // the 3rd field, tensor data
    uint64_t size = tensor.numel() * framework::DataTypeSize(tensor.dtype());

    auto* data_ptr = tensor.data();
    PADDLE_ENFORCE_LT(size,
                      (std::numeric_limits<std::streamsize>::max)(),
                      platform::errors::ResourceExhausted(
                          "tensor size %d overflow when writing tensor", size));
    if (platform::is_gpu_place(tensor.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& gpu_dev_ctx = static_cast<const phi::GPUContext&>(dev_ctx);
      platform::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        memory::Copy(cpu,
                     buf.get(),
                     tensor.place(),
                     reinterpret_cast<const void*>(data),
                     size_to_write,
                     gpu_dev_ctx.stream());
        gpu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "CUDAPlace is not supported when not compiled with CUDA"));
#endif
    } else if (platform::is_xpu_place(tensor.place())) {
#ifdef PADDLE_WITH_XPU
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& xpu_dev_ctx =
          static_cast<const platform::XPUDeviceContext&>(dev_ctx);
      platform::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        memory::Copy(cpu,
                     buf.get(),
                     tensor.place(),
                     reinterpret_cast<const void*>(data),
                     size_to_write);
        xpu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "XPUPlace is not supported when not compiled with XPU"));
#endif
    } else if (platform::is_mlu_place(tensor.place())) {
#ifdef PADDLE_WITH_MLU
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& mlu_dev_ctx =
          static_cast<const platform::MLUDeviceContext&>(dev_ctx);
      platform::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        memory::Copy(cpu,
                     buf.get(),
                     tensor.place(),
                     reinterpret_cast<const void*>(data),
                     size_to_write,
                     mlu_dev_ctx.stream());
        mlu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "MLUPlace is not supported when not compiled with MLU"));
#endif
    } else if (platform::is_npu_place(tensor.place())) {
#ifdef PADDLE_WITH_ASCEND_CL
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& npu_dev_ctx =
          static_cast<const platform::NPUDeviceContext&>(dev_ctx);
      platform::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        memory::Copy(cpu,
                     buf.get(),
                     tensor.place(),
                     reinterpret_cast<const void*>(data),
                     size_to_write,
                     npu_dev_ctx.stream());
        npu_dev_ctx.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "NPUPlace is not supported when not compiled with NPU"));
#endif
    } else if (platform::is_custom_place(tensor.place())) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
      std::unique_ptr<char[]> buf(new char[kBufSize]);
      auto& custom_device_context =
          static_cast<const platform::CustomDeviceContext&>(dev_ctx);
      platform::CPUPlace cpu;
      uintptr_t data = reinterpret_cast<uintptr_t>(data_ptr);
      while (size != 0) {
        size_t size_to_write = std::min(kBufSize, static_cast<size_t>(size));
        memory::Copy(cpu,
                     buf.get(),
                     tensor.place(),
                     reinterpret_cast<const void*>(data),
                     size_to_write,
                     custom_device_context.stream());
        custom_device_context.Wait();
        os.write(buf.get(), size_to_write);
        data += size_to_write;
        size -= size_to_write;
      }
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "CustomPlace is not supported when not compiled with "
          "CustomDevice"));
#endif
    } else {
      os.write(static_cast<const char*>(data_ptr),
               static_cast<std::streamsize>(size));
    }
  }
}

struct DeserializedDataFunctor {
  DeserializedDataFunctor(void** buf,
                          phi::DenseTensor* tensor,
                          const platform::Place& place)
      : buf_(buf), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() {
    *buf_ = tensor_->mutable_data<T>(place_);
  }

  void** buf_;
  phi::DenseTensor* tensor_;
  platform::Place place_;
};

void TensorFromStream(std::istream& is,
                      phi::DenseTensor* tensor,
                      const platform::DeviceContext& dev_ctx,
                      const size_t& seek,
                      const std::vector<int64_t>& shape) {
  uint32_t version;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));

  PADDLE_ENFORCE_EQ(
      version,
      0U,
      platform::errors::InvalidArgument(
          "tensor version %u is not supported, Only version 0 is supported",
          version));

  proto::VarType::TensorDesc desc;
  {  // int32_t size
    // proto buffer
    int32_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char*>(buf.get()), size);
    PADDLE_ENFORCE_EQ(
        desc.ParseFromArray(buf.get(), size),
        true,
        platform::errors::InvalidArgument("Cannot parse tensor desc"));
  }
  {  // read tensor
    tensor->Resize(phi::make_ddim(shape));
    size_t seekg = seek * framework::SizeOfType(desc.data_type());
    is.seekg(seekg, is.cur);

    void* buf;
    phi::CPUContext ctx;
    size_t size = tensor->numel() * framework::SizeOfType(desc.data_type());
    if (platform::is_gpu_place(dev_ctx.GetPlace()) ||
        platform::is_xpu_place(dev_ctx.GetPlace()) ||
        platform::is_mlu_place(dev_ctx.GetPlace()) ||
        platform::is_npu_place(dev_ctx.GetPlace()) ||
        platform::is_custom_place(dev_ctx.GetPlace())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU) || defined(PADDLE_WITH_MLU) ||  \
    defined(PADDLE_WITH_ASCEND_CL) || defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DenseTensor cpu_tensor;
      cpu_tensor.Resize(phi::make_ddim(shape));
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, &cpu_tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);
      auto dst_place = dev_ctx.GetPlace();
      framework::TensorCopy(cpu_tensor, dst_place, dev_ctx, tensor);
      if (platform::is_npu_place(dev_ctx.GetPlace()) ||
          platform::is_custom_place(dev_ctx.GetPlace())) {
        dev_ctx.Wait();
      }
#else
      if (platform::is_gpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "CUDAPlace is not supported when not compiled with CUDA"));
      } else if (platform::is_xpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "XPUPlace is not supported when not compiled with XPU"));
      } else if (platform::is_mlu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "MLUPlace is not supported when not compiled with MLU"));
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "NPUPlace is not supported when not compiled with NPU"));
      }
#endif
    } else {
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);
    }
  }
}

void TensorFromStream(std::istream& is,
                      phi::DenseTensor* tensor,
                      const platform::DeviceContext& dev_ctx) {
  uint32_t version;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));
  PADDLE_ENFORCE_EQ(
      version,
      0U,
      platform::errors::InvalidArgument(
          "tensor version %u is not supported, Only version 0 is supported",
          version));
  proto::VarType::TensorDesc desc;
  {  // int32_t size
     // proto buffer
    int32_t size = -1;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    PADDLE_ENFORCE_EQ(
        is.good(),
        true,
        platform::errors::Unavailable("Cannot read tensor desc size"));
    PADDLE_ENFORCE_GE(size,
                      0,
                      platform::errors::InvalidArgument(
                          "phi::DenseTensor desc size should >= 0"));
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char*>(buf.get()), size);
    PADDLE_ENFORCE_EQ(
        desc.ParseFromArray(buf.get(), size),
        true,
        platform::errors::InvalidArgument("Cannot parse tensor desc"));
  }
  {  // read tensor
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(desc.dims().size()));
    std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims));
    tensor->Resize(phi::make_ddim(dims));
    void* buf;
    phi::CPUContext ctx;
    size_t size = tensor->numel() * framework::SizeOfType(desc.data_type());
    if (platform::is_gpu_place(dev_ctx.GetPlace()) ||
        platform::is_xpu_place(dev_ctx.GetPlace()) ||
        platform::is_mlu_place(dev_ctx.GetPlace()) ||
        platform::is_npu_place(dev_ctx.GetPlace()) ||
        platform::is_custom_place(dev_ctx.GetPlace())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU) || defined(PADDLE_WITH_MLU) ||  \
    defined(PADDLE_WITH_ASCEND_CL) || defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DenseTensor cpu_tensor;
      cpu_tensor.Resize(phi::make_ddim(dims));
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, &cpu_tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);
      auto dst_place = dev_ctx.GetPlace();
      framework::TensorCopy(cpu_tensor, dst_place, dev_ctx, tensor);
      if (platform::is_npu_place(dev_ctx.GetPlace()) ||
          platform::is_custom_place(dev_ctx.GetPlace())) {
        dev_ctx.Wait();
      }
#else
      if (platform::is_gpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "CUDAPlace is not supported when not compiled with CUDA"));
      } else if (platform::is_xpu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "XPUPlace is not supported when not compiled with XPU"));
      } else if (platform::is_mlu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "MLUPlace is not supported when not compiled with MLU"));
      } else if (platform::is_npu_place(dev_ctx.GetPlace())) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "NPUPlace is not supported when not compiled with NPU"));
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "CutomPlace is not supported when not compiled with CustomDevice"));
      }
#endif
    } else {
      framework::VisitDataType(
          desc.data_type(),
          DeserializedDataFunctor(&buf, tensor, ctx.GetPlace()));
      is.read(static_cast<char*>(buf), size);
    }
  }
}

// get tensor data point by DLDataType
void* GetDstPtrByDLDataType(DLDataType type,
                            phi::DenseTensor* dst,
                            const platform::Place& dst_place) {
  // vector types not currently supported
  PADDLE_ENFORCE_LE(type.lanes,
                    1,
                    platform::errors::Unimplemented(
                        "Vector type is not supported currently."));

  switch (type.bits) {
    case 8:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int8_t>(dst_place));
      if (type.code == kDLUInt)
        return static_cast<void*>(dst->mutable_data<uint8_t>(dst_place));
      PADDLE_THROW(platform::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 16:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int16_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void*>(
            dst->mutable_data<paddle::platform::float16>(dst_place));
      if (type.code == kDLBfloat)
        return static_cast<void*>(
            dst->mutable_data<paddle::platform::bfloat16>(dst_place));
      PADDLE_THROW(platform::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 32:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int32_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void*>(dst->mutable_data<float>(dst_place));
      PADDLE_THROW(platform::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 64:
      if (type.code == kDLInt)
        return static_cast<void*>(dst->mutable_data<int64_t>(dst_place));
      if (type.code == kDLFloat)
        return static_cast<void*>(dst->mutable_data<double>(dst_place));
      if (type.code == kDLComplex)
        return static_cast<void*>(
            dst->mutable_data<paddle::platform::complex<float>>(dst_place));
      PADDLE_THROW(platform::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    case 128:
      if (type.code == kDLComplex)
        return static_cast<void*>(
            dst->mutable_data<paddle::platform::complex<double>>(dst_place));
      PADDLE_THROW(platform::errors::Unimplemented(
          "DLDataType code <%d> is illegal when DLDataType.bits is <%d>.",
          type.code,
          type.bits));
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported DLDataType.bits %d.", type.bits));
  }
}

void TensorFromDLPack(const ::DLTensor& dl_tensor, phi::DenseTensor* dst) {
  platform::CPUPlace dst_place = platform::CPUPlace();
  platform::CPUPlace src_place = platform::CPUPlace();

  std::vector<int64_t> vec;
  std::copy(dl_tensor.shape,
            dl_tensor.shape + dl_tensor.ndim,
            std::back_inserter(vec));

  framework::DDim vddim = phi::make_ddim(vec);

  dst->Resize(vddim);
  ::DLDataType type = dl_tensor.dtype;
  void* dst_ptr = GetDstPtrByDLDataType(type, dst, dst_place);

  auto src_ptr = static_cast<const void*>(dl_tensor.data);
  auto size = phi::product(vddim) * type.bits / 8;

  if (dl_tensor.device.device_type == kDLCPU) {
    memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (dl_tensor.device.device_type == kDLGPU) {
    platform::CUDAPlace dst_place =
        platform::CUDAPlace(dl_tensor.device.device_id);
    platform::CUDAPlace src_place =
        platform::CUDAPlace(dl_tensor.device.device_id);
    dst_ptr = GetDstPtrByDLDataType(type, dst, dst_place);
    auto* ctx = platform::DeviceContextPool::Instance().GetByPlace(dst_place);
    memory::Copy(dst_place,
                 dst_ptr,
                 src_place,
                 src_ptr,
                 size,
                 reinterpret_cast<const phi::GPUContext&>(*ctx).stream());
  }
#endif
#ifdef PADDLE_WITH_XPU
  PADDLE_THROW(platform::errors::Unimplemented("XPUPlace is not supported"));
#endif
}

template <typename T>
std::string format_tensor(const phi::DenseTensor& tensor) {
  // TODO(zhiqiu): use the print option to format tensor.
  return "NOT IMPLEMENTED";
}

template <typename T>
std::ostream& print_tensor(std::ostream& os, const phi::DenseTensor& tensor) {
  auto inspect = tensor.data<T>();
  auto element_num = tensor.numel();

  os << "  - data: [";
  // Note: int8_t && uint8_t is typedf of char, ostream unable to print properly
  if (typeid(int8_t) == typeid(T) || typeid(uint8_t) == typeid(T)) {
    if (element_num > 0) {
      os << signed(inspect[0]);
      for (int j = 1; j < element_num; ++j) {
        os << " " << signed(inspect[j]);
      }
    }
  } else {
    if (element_num > 0) {
      os << inspect[0];
      for (int j = 1; j < element_num; ++j) {
        os << " " << inspect[j];
      }
    }
  }
  os << "]";
  return os;
}

template <>
std::ostream& print_tensor<paddle::platform::complex<float>>(
    std::ostream& os, const phi::DenseTensor& tensor) {
  auto inspect = tensor.data<paddle::platform::complex<float>>();
  auto element_num = tensor.numel();

  os << "  - data: [";
  if (element_num > 0) {
    os << signed(inspect[0].real) << "+" << signed(inspect[0].imag) << "j";
    for (int j = 1; j < element_num; ++j) {
      os << " " << signed(inspect[j].real) << "+" << signed(inspect[j].imag)
         << "j";
    }
  }
  os << "]";
  return os;
}

template <>
std::ostream& print_tensor<paddle::platform::complex<double>>(
    std::ostream& os, const phi::DenseTensor& tensor) {
  auto inspect = tensor.data<paddle::platform::complex<double>>();
  auto element_num = tensor.numel();

  os << "  - data: [";
  if (element_num > 0) {
    os << signed(inspect[0].real) << "+" << signed(inspect[0].imag) << "j";
    for (int j = 1; j < element_num; ++j) {
      os << " " << signed(inspect[j].real) << "+" << signed(inspect[j].imag)
         << "j";
    }
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const LoD& lod) {
  // NOTE(xiongkun):
  // https://stackoverflow.com/questions/5195512/namespaces-and-operator-resolution
  // if we don't redefine, the operator << of phi / framework LoD is not found.
  paddle::string::operator<<(os, lod);
  return os;
}

}  // namespace framework
}  // namespace paddle

namespace phi {

std::ostream& operator<<(std::ostream& os, const LoD& lod) {
  paddle::string::operator<<(os, lod);
  return os;
}

std::ostream& operator<<(std::ostream& os, const phi::DenseTensor& t) {
  if (t.lod().size() > 0) {
    os << "  - lod: " << t.lod() << "\n";
  }

  os << "  - place: " << t.place() << "\n";
  os << "  - shape: [" << t.dims() << "]\n";
  os << "  - layout: " << paddle::framework::DataLayoutToString(t.layout())
     << "\n";

#ifdef PADDLE_WITH_MKLDNN
  os << "  - format: "
     << dnnl_fmt_tag2str(static_cast<dnnl_format_tag_t>(t.format())) << "\n";
#endif

  DenseTensor tensor;
  tensor.Resize(t.dims());
  if (paddle::platform::is_cpu_place(t.place())) {
    tensor.ShareDataWith(t);
  } else {
    paddle::platform::CPUPlace place;
    paddle::framework::TensorCopy(t, place, &tensor);
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(t.place());
    dev_ctx.Wait();
  }

#define PrintTensorCallback(cpp_type, proto_type)                 \
  do {                                                            \
    if (paddle::framework::TransToProtoVarType(tensor.dtype()) == \
        proto_type) {                                             \
      os << "  - dtype: " << proto_type << "\n";                  \
      paddle::framework::print_tensor<cpp_type>(os, tensor);      \
      return os;                                                  \
    }                                                             \
  } while (0)

  _ForEachDataType_(PrintTensorCallback);
  VLOG(1) << "PrintVar: unrecognized data type:" << t.type();
  return os;
}
}  // namespace phi
