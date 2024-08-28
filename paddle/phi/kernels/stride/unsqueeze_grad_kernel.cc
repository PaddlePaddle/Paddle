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
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/flatten_grad_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void UnsqueezeGradStridedKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& dout,
                                DenseTensor* dx) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  // NOTE: [Why not to use x.dims() ?]
  // Because inplace strategy is different between old IR and PIR,
  // we need fix it into x.dims() after cleaning old IR system.
  const auto& x_dims = dx->dims();
  ReshapeStridedKernel<Context>(
      dev_ctx, dout, IntArray(common::vectorize<int64_t>(x_dims)), dx);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(unsqueeze_grad,
                                         STRIDED,
                                         phi::UnsqueezeGradStridedKernel) {}
