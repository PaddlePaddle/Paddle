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

#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context>
void TransposeStridedKernel(const Context& ctx,
                            const DenseTensor& x,
                            const std::vector<int>& axis,
                            DenseTensor* out) {
  size_t x_rank = x.dims().size();
  std::vector<int> formated_axis = axis;
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      formated_axis[i] = axis[i] + x_rank;
    }
  }

  auto meta = out->meta();
  auto in_stride = x.strides();
  auto in_dims = x.dims();
  meta.strides = in_stride;
  for (size_t i = 0; i < formated_axis.size(); i++) {
    meta.strides[i] = in_stride[formated_axis[i]];
    meta.dims[i] = in_dims[formated_axis[i]];
  }
  meta.offset = x.offset();

  out->set_meta(meta);
  out->ResetHolder(x.Holder());
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    transpose, STRIDED, phi::TransposeStridedKernel) {}
