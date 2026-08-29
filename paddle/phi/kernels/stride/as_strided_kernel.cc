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
#include "paddle/common/ddim.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {
void ValidateZeroSizeTensorShape(const std::vector<int64_t>& dims,
                                 const std::vector<int64_t>& strides,
                                 const DenseTensor& input) {
  if (input.numel() != 0) {
    return;
  }
  PADDLE_ENFORCE_EQ(dims.size(),
                    strides.size(),
                    common::errors::InvalidArgument(
                        "The size of dims and strides should be equal."));
  for (size_t i = 0; i < dims.size(); i++) {
    if (dims[i] == 0) {
      return;
    }
  }

  PADDLE_THROW(common::errors::InvalidArgument(
      "When input is zero-size tensor, the shape attribute must also be "
      "zero-size."));
}

// Rejects views that would reach outside of the input's allocation. Without
// this check a bad (shape, stride, offset) triple silently produces a tensor
// whose reads and writes corrupt neighbouring heap memory.
void ValidateStorageRange(const std::vector<int64_t>& dims,
                          const std::vector<int64_t>& strides,
                          int64_t offset,
                          const DenseTensor& input) {
  if (input.numel() == 0 || input.Holder() == nullptr) {
    return;
  }
  PADDLE_ENFORCE_EQ(dims.size(),
                    strides.size(),
                    common::errors::InvalidArgument(
                        "The size of dims and strides should be equal."));
  const int64_t itemsize = static_cast<int64_t>(SizeOf(input.dtype()));
  PADDLE_ENFORCE_EQ(offset % itemsize,
                    0,
                    common::errors::InvalidArgument(
                        "The offset(%d) is a byte offset and must be a "
                        "multiple of the element size(%d) of the input.",
                        offset,
                        itemsize));
  // Element index range covered by the view, relative to `offset`.
  int64_t min_index = 0;
  int64_t max_index = 0;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] == 0) {
      return;  // An empty view never touches the storage.
    }
    const int64_t span = strides[i] * (dims[i] - 1);
    if (span > 0) {
      max_index += span;
    } else {
      min_index += span;
    }
  }
  const int64_t base = offset / itemsize;
  const int64_t storage_numel =
      static_cast<int64_t>(input.Holder()->size()) / itemsize;
  PADDLE_ENFORCE_GE(base + min_index,
                    0,
                    common::errors::InvalidArgument(
                        "The view described by shape %s, stride %s and offset "
                        "%d reaches element %d, which is before the beginning "
                        "of the input storage.",
                        common::make_ddim(dims),
                        common::make_ddim(strides),
                        offset,
                        base + min_index));
  PADDLE_ENFORCE_LT(base + max_index,
                    storage_numel,
                    common::errors::InvalidArgument(
                        "The view described by shape %s, stride %s and offset "
                        "%d reaches element %d, but the input storage only "
                        "holds %d elements.",
                        common::make_ddim(dims),
                        common::make_ddim(strides),
                        offset,
                        base + max_index,
                        storage_numel));
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
  ValidateZeroSizeTensorShape(dims, stride, input);
  PADDLE_ENFORCE_GE(
      offset,
      0,
      common::errors::InvalidArgument(
          "The offset must be non-negative, but got %d.", offset));
  ValidateStorageRange(dims, stride, offset, input);
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
  out->ShareInplaceVersionCounterWith(input);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(as_strided,
                                         STRIDED,
                                         phi::AsStridedKernel) {}
