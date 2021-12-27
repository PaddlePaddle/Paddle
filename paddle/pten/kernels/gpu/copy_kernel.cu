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

#include "paddle/pten/kernels/copy_kernel.h"

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/memcpy.h"

namespace pten {

template <typename ContextT>
void Copy(const ContextT& dev_ctx,
          const DenseTensor& src,
          bool blocking,
          DenseTensor* dst) {
  auto* src_ptr = src.data();
  const auto& src_place = src.place();
  const auto& dst_place = dst->place();

  if (src_place == dst_place && paddle::platform::is_cpu_place(src_place)) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "The src and dst tensor are all CPU tensor, you should call copy "
        "function in CPU mode."));
  }

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;

  dst->Resize(src.dims());
  auto* dst_ptr = dst->mutable_data();

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << dst_place;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;
  CHECK(dst->layout() == src.layout());

  auto size = src.numel() *
              paddle::framework::SizeOfType(TransToProtoVarType(src.dtype()));

  if (paddle::platform::is_cuda_pinned_place(src_place) &&  // NOLINT
      paddle::platform::is_cuda_pinned_place(dst_place)) {
    paddle::memory::Copy(
        BOOST_GET_CONST(paddle::platform::CUDAPinnedPlace, dst_place),
        dst_ptr,
        BOOST_GET_CONST(paddle::platform::CUDAPinnedPlace, src_place),
        src_ptr,
        size);
  } else if (paddle::platform::is_cuda_pinned_place(src_place) &&  // NOLINT
             paddle::platform::is_cpu_place(dst_place)) {
    paddle::memory::Copy(
        BOOST_GET_CONST(paddle::platform::CPUPlace, dst_place),
        dst_ptr,
        BOOST_GET_CONST(paddle::platform::CUDAPinnedPlace, src_place),
        src_ptr,
        size);
  } else if (paddle::platform::is_cpu_place(src_place) &&  // NOLINT
             paddle::platform::is_cuda_pinned_place(dst_place)) {
    paddle::memory::Copy(
        BOOST_GET_CONST(paddle::platform::CUDAPinnedPlace, dst_place),
        dst_ptr,
        BOOST_GET_CONST(paddle::platform::CPUPlace, src_place),
        src_ptr,
        size);
  } else if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
             paddle::platform::is_cpu_place(dst_place)) {
    auto src_gpu_place =
        BOOST_GET_CONST(paddle::platform::CUDAPlace, src_place);
    auto dst_cpu_place = BOOST_GET_CONST(paddle::platform::CPUPlace, dst_place);
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        paddle::platform::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place =
        BOOST_GET_CONST(paddle::platform::CUDAPlace, ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place,
                      ctx_gpu_place,
                      paddle::platform::errors::Unavailable(
                          "Source place and context place do not match, source "
                          "place is %s, context place is %s.",
                          src_gpu_place,
                          ctx_gpu_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const paddle::platform::CUDADeviceContext&>(
                       dev_ctx)
                       .stream();
    paddle::memory::Copy(
        dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  } else if (paddle::platform::is_cpu_place(src_place) &&  // NOLINT
             paddle::platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = BOOST_GET_CONST(paddle::platform::CPUPlace, src_place);
    auto dst_gpu_place =
        BOOST_GET_CONST(paddle::platform::CUDAPlace, dst_place);
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        paddle::platform::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto ctx_gpu_place =
        BOOST_GET_CONST(paddle::platform::CUDAPlace, ctx_place);
    PADDLE_ENFORCE_EQ(dst_gpu_place,
                      ctx_gpu_place,
                      paddle::platform::errors::Unavailable(
                          "Destination place and context place do not match, "
                          "destination place is %s, context place is %s.",
                          dst_gpu_place,
                          ctx_gpu_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const paddle::platform::CUDADeviceContext&>(
                       dev_ctx)
                       .stream();
    paddle::memory::Copy(
        dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size, stream);
  } else if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
             paddle::platform::is_cuda_pinned_place(dst_place)) {
    auto src_gpu_place =
        BOOST_GET_CONST(paddle::platform::CUDAPlace, src_place);
    auto dst_cuda_pinned_place =
        BOOST_GET_CONST(paddle::platform::CUDAPinnedPlace, dst_place);
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(paddle::platform::is_gpu_place(ctx_place),
                      true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Device context place mismatch. When copying Tensor "
                          "data from GPU memory to CUDA Pinned memory, current "
                          "device context place should be GPU."));
    auto ctx_gpu_place =
        BOOST_GET_CONST(paddle::platform::CUDAPlace, ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place,
                      ctx_gpu_place,
                      paddle::platform::errors::PreconditionNotMet(
                          "The source GPU device and current device context do "
                          "not match. The source GPU device number is %d, but "
                          "device context GPU number is %d.",
                          src_gpu_place.device,
                          ctx_gpu_place.device));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const paddle::platform::CUDADeviceContext&>(
                       dev_ctx)
                       .stream();
    paddle::memory::Copy(
        dst_cuda_pinned_place, dst_ptr, src_gpu_place, src_ptr, size, stream);
  } else if (paddle::platform::is_cuda_pinned_place(src_place) &&  // NOLINT
             paddle::platform::is_gpu_place(dst_place)) {
    auto src_cuda_pinned_place =
        BOOST_GET_CONST(paddle::platform::CUDAPinnedPlace, src_place);
    auto dst_gpu_place =
        BOOST_GET_CONST(paddle::platform::CUDAPlace, dst_place);
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(paddle::platform::is_gpu_place(ctx_place),
                      true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Device context place mismatch. When copying Tensor "
                          "data from CUDA Pinned memory to GPU memory, current "
                          "device context place should be GPU."));
    auto ctx_gpu_place =
        BOOST_GET_CONST(paddle::platform::CUDAPlace, ctx_place);
    PADDLE_ENFORCE_EQ(dst_gpu_place,
                      ctx_gpu_place,
                      paddle::platform::errors::PreconditionNotMet(
                          "The target GPU device and current device context do "
                          "not match. The target GPU device number is %d, but "
                          "device context GPU number is %d.",
                          dst_gpu_place.device,
                          ctx_gpu_place.device));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const paddle::platform::CUDADeviceContext&>(
                       dev_ctx)
                       .stream();
    paddle::memory::Copy(
        dst_gpu_place, dst_ptr, src_cuda_pinned_place, src_ptr, size, stream);
  } else if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
             paddle::platform::is_gpu_place(dst_place)) {
    auto src_gpu_place =
        BOOST_GET_CONST(paddle::platform::CUDAPlace, src_place);
    auto dst_gpu_place =
        BOOST_GET_CONST(paddle::platform::CUDAPlace, dst_place);
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        paddle::platform::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    auto stream =
        blocking ? nullptr
                 : reinterpret_cast<const paddle::platform::CUDADeviceContext&>(
                       dev_ctx)
                       .stream();
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
        PADDLE_THROW(paddle::platform::errors::Unavailable(
            "Context place dose not match the source and destination place."));
      }
    }
  }
}

}  // namespace pten

PT_REGISTER_GENERAL_KERNEL(
    copy, GPU, ALL_LAYOUT, pten::Copy<pten::GPUContext>, ALL_DTYPE) {}
