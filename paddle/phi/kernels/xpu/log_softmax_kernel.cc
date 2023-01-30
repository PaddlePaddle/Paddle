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
<<<<<<< HEAD
#include "paddle/phi/kernels/funcs/math_function.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace phi {

template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const int rank = x.dims().size();
  axis = funcs::CanonicalAxis(axis, rank);

<<<<<<< HEAD
  // For 0D Tensor
  if (rank == 0) {
    phi::funcs::set_constant(dev_ctx, out, 0.0);
    return;
  }
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  if (x.numel() != 0) {
    auto x_shape = phi::vectorize<int>(x.dims());
    dev_ctx.template Alloc<T>(out);
    if (axis < 0) axis += rank;
<<<<<<< HEAD
    int r =
        xpu::log_softmax<XPUType>(dev_ctx.x_context(),
=======
    int r = xpu::softmax<XPUType>(dev_ctx.x_context(),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                  reinterpret_cast<const XPUType*>(x.data<T>()),
                                  reinterpret_cast<XPUType*>(out->data<T>()),
                                  x_shape,
                                  axis);
<<<<<<< HEAD
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "log_softmax");
=======
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax");
    r = xpu::log<XPUType>(dev_ctx.x_context(),
                          reinterpret_cast<const XPUType*>(out->data<T>()),
                          reinterpret_cast<XPUType*>(out->data<T>()),
                          out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "log");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(log_softmax, XPU, ALL_LAYOUT, phi::LogSoftmaxKernel, float) {
}
