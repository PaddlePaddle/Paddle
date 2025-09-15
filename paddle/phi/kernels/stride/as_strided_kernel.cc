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
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
  out->ShareInplaceVersionCounterWith(input);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(as_strided,
                                         STRIDED,
                                         phi::AsStridedKernel) {}
