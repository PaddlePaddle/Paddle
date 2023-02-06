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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
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
  using XPUType = typename XPUTypeTrait<T>::Type;

  if (out->dtype() == phi::DataType::INT32) {
    dev_ctx.template Alloc<int>(out);
  } else if (out->dtype() == phi::DataType::INT64) {
    dev_ctx.template Alloc<int64_t>(out);
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("argmax out only support int32 and int64"));
  }

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
  auto* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  int64_t* out_tmp;
  if (out->dtype() == phi::DataType::INT32) {
    out_tmp = RAII_GUARD.alloc_l3_or_gm<int64_t>(out->numel());
  } else {
    out_tmp = out->data<int64_t>();
  }

  int r = xpu::argmax(xpu_ctx,
                      reinterpret_cast<const XPUType*>(x.data<T>()),
                      out_tmp,
                      xdims_vec,
                      axis_val);
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      errors::External("XPU argmax kernel return wrong value[%d %s].",
                       r,
                       XPUAPIErrorMsg[r]));
  if (out->dtype() == phi::DataType::INT32) {
    int r = xpu::cast<int64_t, int>(
        xpu_ctx, out_tmp, out->data<int>(), out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(
    arg_max, XPU, ALL_LAYOUT, phi::ArgMaxKernel, float, phi::dtype::float16) {}
