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

#include "paddle/phi/kernels/transpose_grad_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/transpose_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void TransposeGradStridedKernel(const Context& dev_ctx,
                                const DenseTensor& out_grad,
                                const std::vector<int>& axis,
                                DenseTensor* x_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  size_t axis_size = axis.size();
  std::vector<int> formatted_axis = axis;
  for (size_t i = 0; i < axis_size; i++) {
    if (axis[i] < 0) {
      formatted_axis[i] = static_cast<int>(axis[i] + axis_size);
    }
  }

  std::vector<int> reversed_axis(axis);
  for (int i = 0; i < static_cast<int>(axis_size); i++) {
    reversed_axis[formatted_axis[i]] = i;
  }

  TransposeStridedKernel<Context>(dev_ctx, out_grad, reversed_axis, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(transpose_grad,
                                         STRIDED,
                                         phi::TransposeGradStridedKernel) {}
