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

#include "paddle/phi/kernels/cum_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 1) {
    int r = xpu::copy<XPUType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x.data<T>()),
                               reinterpret_cast<XPUType*>(out->data<T>()),
                               x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    return;
  }

  // prepare for call xdnn api
  std::vector<int> x_shape = common::vectorize<int>(x.dims());
  int axis_as_int = axis.to<int>();

  if (flatten) {
    // flatten to 1-dim vector
    x_shape = {static_cast<int>(x.numel())};
    axis_as_int = 0;
  } else {
    // not flatten
    // check axis_as_int
    auto out_dims = out->dims();

    PADDLE_ENFORCE_EQ(
        axis_as_int < out_dims.size() && axis_as_int >= (0 - out_dims.size()),
        true,
        common::errors::OutOfRange(
            "Attr(axis) is out of range, It's expected "
            "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
            out_dims.size(),
            out_dims.size() - 1,
            axis_as_int));
    if (axis_as_int < 0) {
      axis_as_int += out_dims.size();
    }
  }

  // template<typename T> DLL_EXPORT int cumsum(Context* ctx, const T* x, T*
  // y, const std::vector<int>& xshape, bool reverse, bool exclusive, int
  // axis);
  int r = xpu::cumsum<XPUType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x.data<T>()),
                               reinterpret_cast<XPUType*>(out->data<T>()),
                               x_shape,
                               reverse,
                               exclusive,
                               axis_as_int);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cumsum");
}

}  // namespace phi

PD_REGISTER_KERNEL(cumsum,
                   XPU,
                   ALL_LAYOUT,
                   phi::CumsumKernel,
                   float,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
