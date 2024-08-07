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

#include "paddle/phi/kernels/diagonal_kernel.h"

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void DiagonalStridedKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           int offset,
                           int axis1,
                           int axis2,
                           DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  size_t x_rank = x.dims().size();
  if (axis1 < 0) {
    axis1 += static_cast<int>(x_rank);
  }
  if (axis2 < 0) {
    axis2 += static_cast<int>(x_rank);
  }

  int64_t diag_size = 0;
  int64_t x_offset = static_cast<int64_t>(x.offset());
  if (offset >= 0) {
    diag_size = std::max<int64_t>(
        std::min(x.dims()[axis1], x.dims()[axis2] - offset), 0);
    if (diag_size != 0) {
      x_offset +=
          static_cast<int64_t>(offset * x.strides()[axis2] * SizeOf(x.dtype()));
    }
  } else {
    diag_size = std::max<int64_t>(
        std::min(x.dims()[axis1] + offset, x.dims()[axis2]), 0);
    if (diag_size != 0) {
      x_offset -=
          static_cast<int64_t>(offset * x.strides()[axis1] * SizeOf(x.dtype()));
    }
  }

  std::vector<int64_t> shape = common::vectorize<int64_t>(x.dims());
  std::vector<int64_t> stride = common::vectorize<int64_t>(x.strides());
  shape.erase(shape.begin() + std::max(axis1, axis2));
  stride.erase(stride.begin() + std::max(axis1, axis2));
  shape.erase(shape.begin() + std::min(axis1, axis2));
  stride.erase(stride.begin() + std::min(axis1, axis2));
  shape.push_back(diag_size);
  stride.push_back(x.strides()[axis1] + x.strides()[axis2]);

  auto meta = out->meta();
  auto tmp_dim = DDim(shape.data(), static_cast<int>(shape.size()));
  // if (product(meta.dims) > 0 && meta.dims != tmp_dim) {
  //   PADDLE_THROW(
  //       common::errors::Fatal("Diagonal kernel stride compute diff, infer
  //       shape
  //       "
  //                          "is %s, but compute is %s.",
  //                          meta.dims,
  //                          tmp_dim));
  // }
  meta.dims = tmp_dim;
  meta.strides = DDim(stride.data(), static_cast<int>(stride.size()));
  meta.offset = x_offset;
  out->set_meta(meta);
  out->ResetHolder(x.Holder());
  out->ShareInplaceVersionCounterWith(x);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(diagonal,
                                         STRIDED,
                                         phi::DiagonalStridedKernel) {}
