// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/log_softmax_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace phi {

template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
  const int rank = x.dims().size();
  axis = funcs::CanonicalAxis(axis, rank);

  if (x.numel() != 0) {
    auto x_shape = phi::vectorize<int>(x.dims());
    out->mutable_data<float>(dev_ctx.GetPlace());
    if (axis < 0) axis += rank;
    int r = xpu::softmax<T>(
        dev_ctx.x_context(), x.data<T>(), out->data<T>(), x_shape, axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax");
    r = xpu::log<T>(
        dev_ctx.x_context(), out->data<T>(), out->data<T>(), out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "log");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(log_softmax, XPU, ALL_LAYOUT, phi::LogSoftmaxKernel, float) {
}
