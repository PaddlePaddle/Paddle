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

#include "paddle/pten/kernels/strings/strings_deserialize_kernel.h"

#include "paddle/pten/backends/gpu/gpu_helper.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/copy_kernel.h"
#include "paddle/pten/kernels/empty_kernel.h"
#include "paddle/pten/kernels/strings/strings_copy_kernel.h"
#include "paddle/pten/kernels/strings/strings_serialize_kernel.h"

#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"

using pstring = ::pten::dtype::pstring;

namespace pten {
namespace strings {

__global__ void CopyFromStringTensor(pstring* dst,
                                     const pstring* src,
                                     int64_t num) {
  CUDA_KERNEL_LOOP(i, num) { dst[i] = src[i]; }
}

template <typename Context>
void Copy(const Context& dev_ctx,
          const StringTensor& src,
          bool blocking,
          StringTensor* dst) {
  auto* src_ptr = src.data();
  const auto& src_place = src.place();
  auto dst_place = dst->place();

  if (src_place == dst_place && paddle::platform::is_cpu_place(src_place)) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "The src and dst tensor are all CPU tensor, you should call copy "
        "function in CPU mode."));
  }
  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;

  dst->ResizeAndAllocate(src.dims());
  auto* dst_ptr = dst->mutable_data(dst_place);

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << dst_place;
    return;
  }

  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;
  if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
      paddle::platform::is_cpu_place(dst_place)) {
    // Situation 1: gpu_place->cpu_place
    pten::DeviceContextPool& pool = pten::DeviceContextPool::Instance();
    CPUContext* dst_ctx = reinterpret_cast<CPUContext*>(pool.Get(dst_place));

    DenseTensor gpu_serialized =
        pten::Empty<uint8_t, GPUContext>(dev_ctx, {1}, DataType::UINT8);
    pten::strings::Serialize(dev_ctx, src, &gpu_serialized);

    DenseTensor cpu_serialized =
        pten::EmptyLike<uint8_t>(*dst_ctx, gpu_serialized, DataType::UINT8);
    pten::Copy(dev_ctx, gpu_serialized, false, &cpu_serialized);

    pten::strings::Deserialize(*dst_ctx, cpu_serialized, dst);
  } else if (paddle::platform::is_cpu_place(src_place) &&  // NOLINT
             paddle::platform::is_gpu_place(dst_place)) {
    // Situation 2: cpu_place->gpu_place
    pten::DeviceContextPool& pool = pten::DeviceContextPool::Instance();
    CPUContext* src_ctx = reinterpret_cast<CPUContext*>(pool.Get(src_place));

    DenseTensor cpu_serialized =
        pten::Empty<uint8_t, CPUContext>(*src_ctx, {1}, DataType::UINT8);
    pten::strings::Serialize(*src_ctx, src, &cpu_serialized);

    DenseTensor gpu_serialized =
        pten::EmptyLike<uint8_t>(dev_ctx, cpu_serialized, DataType::UINT8);
    pten::Copy(dev_ctx, cpu_serialized, false, &gpu_serialized);

    pten::strings::Deserialize(dev_ctx, gpu_serialized, dst);
  } else if (paddle::platform::is_gpu_place(src_place) &&  // NOLINT
             paddle::platform::is_gpu_place(dst_place)) {
    // Situation 3: gpu_place->gpu_place
    auto src_gpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_gpu_place(ctx_place),
        true,
        paddle::platform::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    int64_t numel = src.numel();
    dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
    dim3 grid_size =
        dim3((numel + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
    if (paddle::platform::is_same_place(src_place, dst_place)) {
      // Copy
      CopyFromStringTensor<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
          dst_ptr, src_ptr, numel);
    } else {
      if (paddle::platform::is_same_place(ctx_place, src_place)) {
        // Copy
        CopyFromStringTensor<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
            dst_ptr, src_ptr, numel);
        paddle::platform::DeviceContextPool::Instance()
            .Get(src.place())
            ->Wait();
      } else if (paddle::platform::is_same_place(ctx_place, dst_place)) {
        paddle::platform::DeviceContextPool::Instance()
            .Get(src.place())
            ->Wait();
        CopyFromStringTensor<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
            dst_ptr, src_ptr, numel);
      } else {
        PADDLE_THROW(paddle::platform::errors::Unavailable(
            "Context place dose not match the source and destination place."));
      }
    }
  }
  // TODO(zhoushunjie): Add pinned memory copy
  // Situation 4: cuda_pinned_place->cuda_pinned_place
  // Situation 5: cuda_pinned_place->cpu_place
  // Situation 6: cpu_place->cuda_pinned_place
  // Situation 7: gpu_place->cuda_pinned_place
  // Situation 8: cuda_pinned_place->gpu_place
}

}  // namespace strings
}  // namespace pten

PT_REGISTER_GENERAL_KERNEL(strings_copy,
                           GPU,
                           ALL_LAYOUT,
                           pten::strings::Copy<pten::GPUContext>,
                           pstring) {}
