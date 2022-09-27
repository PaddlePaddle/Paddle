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

#include "paddle/phi/kernels/arg_min_max_kernel.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      (dtype < 0 || dtype == 2 || dtype == 3),
      true,
      errors::InvalidArgument(
          "The attribute of dtype in xpu argmin/argmax must be [%s] or [%s], "
          "but "
          "received [%s]",
          DataType::INT64,
          DataType::INT32,
          dtype));
  dev_ctx.template Alloc<int64_t>(out);

  DDim x_dims;
  int axis_val = axis.to<int>();
  if (flatten) {
    x_dims = phi::make_ddim({x.numel()});
    // if flatten, the axis just as 0
    axis_val = 0;
  } else {
    x_dims = x.dims();
    if (axis_val < 0) axis_val += x_dims.size();
  }
  auto xdims_vec = phi::vectorize<int>(x_dims);
  int r = xpu::argmax(dev_ctx.x_context(),
                      x.data<T>(),
                      out->data<int64_t>(),
                      xdims_vec,
                      axis_val);
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      errors::External("XPU argmax kernel return wrong value[%d %s].",
                       r,
                       XPUAPIErrorMsg[r]));
}
}  // namespace phi
PD_REGISTER_KERNEL(arg_max, XPU, ALL_LAYOUT, phi::ArgMaxKernel, float) {}
