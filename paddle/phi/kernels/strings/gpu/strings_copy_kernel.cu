/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/strings/strings_copy_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/pstring.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/strings/gpu/copy_utils.h"

using pstring = ::phi::dtype::pstring;

namespace phi {
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

  if (src_place == dst_place &&
      src_place.GetType() == phi::AllocationType::CPU) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The src and dst string tensor are all "
        "CPU string tensor, you should call copy "
        "function in CPU mode."));
  }
  VLOG(3) << "StringTensorCopy " << src.dims() << " from " << src.place()
          << " to " << dst_place;

  dst->Resize(src.dims());
  auto* dst_ptr = dev_ctx.template Alloc<dtype::pstring>(dst);

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same string data async from " << src_place
            << " to " << dst_place;
    return;
  }

  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

  if (src_place.GetType() == phi::AllocationType::GPU &&
      dst_place.GetType() == phi::AllocationType::CPU) {
    // Situation 1: gpu_place->cpu_place
    DenseTensor gpu_serialized = phi::Empty<uint8_t, GPUContext>(dev_ctx, {1});
    phi::strings::SerializeOnGPU(dev_ctx, src, &gpu_serialized);

    DenseTensor cpu_serialized;
    cpu_serialized.Resize(gpu_serialized.dims());
    dev_ctx.template HostAlloc<uint8_t>(&cpu_serialized);

    phi::Copy(dev_ctx, gpu_serialized, dst_place, false, &cpu_serialized);

    phi::strings::DeserializeOnCPU(dev_ctx, cpu_serialized, dst);

  } else if (src_place.GetType() == phi::AllocationType::CPU &&
             dst_place.GetType() == phi::AllocationType::GPU) {
    // Situation 2: cpu_place->gpu_place
    DenseTensor cpu_serialized;
    cpu_serialized.Resize({1});
    dev_ctx.template HostAlloc<uint8_t>(&cpu_serialized);

    phi::strings::SerializeOnCPU(dev_ctx, src, &cpu_serialized);

    DenseTensor gpu_serialized =
        phi::EmptyLike<uint8_t>(dev_ctx, cpu_serialized);
    phi::Copy(
        dev_ctx, cpu_serialized, dev_ctx.GetPlace(), false, &gpu_serialized);

    phi::strings::DeserializeOnGPU(dev_ctx, gpu_serialized, dst);
  } else if (src_place.GetType() == phi::AllocationType::GPU &&
             dst_place.GetType() == phi::AllocationType::GPU) {
    // Situation 3: gpu_place->gpu_place
    auto src_gpu_place = src_place;
    auto dst_gpu_place = dst_place;
    auto ctx_place = dev_ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        ctx_place.GetType(),
        phi::AllocationType::GPU,
        common::errors::PreconditionNotMet(
            "Context place error, excepted GPUPlace, but actually %s.",
            ctx_place));
    int64_t numel = src.numel();
    dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
    dim3 grid_size =
        dim3((numel + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
    // Copy
    CopyFromStringTensor<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
        dst_ptr, src_ptr, numel);
  }
}

}  // namespace strings
}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_DTYPE(strings_copy,
                                 GPU,
                                 ALL_LAYOUT,
                                 phi::strings::Copy<phi::GPUContext>) {}
