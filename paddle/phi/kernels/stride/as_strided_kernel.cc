// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/as_strided_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {
bool checkInBoundsForMemory(const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& strides,
                            int64_t offset,
                            size_t memory_size,
                            phi::DataType dtype) {
  size_t size = 1;
  for (size_t i = 0; i < dims.size(); i++) {
    if (dims[i] == 0) {
      return true;
    }
    size += strides[i] * (dims[i] - 1);
  }
  size_t size_bytes = (size + offset) * phi::SizeOf(dtype);
  return size_bytes <= memory_size;
}

template <typename Context>
void AsStridedKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const std::vector<int64_t>& dims,
                     const std::vector<int64_t>& stride,
                     int64_t offset,
                     DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  auto meta = out->meta();
  meta.dims = DDim(dims.data(), static_cast<int>(dims.size()));
  meta.strides = DDim(stride.data(), static_cast<int>(stride.size()));
  meta.offset = offset;
  if (!checkInBoundsForMemory(
          dims, stride, offset, input.memory_size(), input.dtype())) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "sizes: [%s], strides: [%s], offset: %d, dtype: %s is out "
        "of bounds for input memory_size: %d.",
        meta.dims,
        meta.strides,
        offset,
        input.dtype(),
        input.memory_size()));
  }
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
  out->ShareInplaceVersionCounterWith(input);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(as_strided,
                                         STRIDED,
                                         phi::AsStridedKernel) {}
