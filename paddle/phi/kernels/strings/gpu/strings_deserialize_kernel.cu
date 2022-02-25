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

#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/pstring.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/strings/strings_deserialize_kernel.h"

using pstring = ::phi::dtype::pstring;

namespace phi {
namespace strings {

__global__ void DeserializeCUDAKernel(const char* strings_data,
                                      const int* strings_offset,
                                      phi::dtype::pstring* dst_str,
                                      int numel) {
  CUDA_KERNEL_LOOP(i, numel) {
    // -1 not include '\0'
    auto len = strings_offset[i + 1] - strings_offset[i] - 1;
    dst_str[i] = phi::dtype::pstring(strings_data + strings_offset[i], len);
  }
}

template <typename Context>
void Deserialize(const Context& dev_ctx,
                 const DenseTensor& src,
                 StringTensor* dst) {
  auto* strings_data = reinterpret_cast<const char*>(src.data<uint8_t>());
  auto* strings_offset = reinterpret_cast<const int*>(strings_data);
  int numel = 0;
#ifdef PADDLE_WITH_HIP
  phi::backends::gpu::GpuMemcpySync(
      &numel, strings_data, sizeof(numel), hipMemcpyDeviceToHost);
#else
  phi::backends::gpu::GpuMemcpySync(
      &numel, strings_data, sizeof(numel), cudaMemcpyDeviceToHost);
#endif
  numel = numel / sizeof(int) - 1;
  dst->Resize({numel});
  dtype::pstring* dst_str = dev_ctx.template Alloc<dtype::pstring>(dst);

  dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
  dim3 grid_size =
      dim3((numel + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
  DeserializeCUDAKernel<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      strings_data, strings_offset, dst_str, numel);
}

}  // namespace strings
}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(strings_deserialize,
                           GPU,
                           ALL_LAYOUT,
                           phi::strings::Deserialize<phi::GPUContext>,
                           pstring) {}
