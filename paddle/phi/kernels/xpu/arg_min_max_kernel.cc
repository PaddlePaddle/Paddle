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
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

namespace {
const int ARG_MAX_OUTPUT_DATATYPE_INT32 = 2;
const int ARG_MAX_OUTPUT_DATATYPE_INT64 = 3;
}  // Anonymous namespace

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  PADDLE_ENFORCE_EQ(
      (dtype < 0 || dtype == ARG_MAX_OUTPUT_DATATYPE_INT32 ||
       dtype == ARG_MAX_OUTPUT_DATATYPE_INT64),
      true,
      errors::InvalidArgument(
          "The attribute of dtype in xpu argmin/argmax must be [%s] or [%s], "
          "but "
          "received [%s]",
          DataType::INT64,
          DataType::INT32,
          dtype));
  // TODO(ZHUI): fix dtype of out
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
  int r = 0;
  if (dtype != ARG_MAX_OUTPUT_DATATYPE_INT32) {
    dev_ctx.template Alloc<int64_t>(out);
    if (x.dims().size() == 0) {
      xpu::constant(dev_ctx.x_context(),
                    out->data<int64_t>(),
                    x.numel(),
                    static_cast<int64_t>(0));
      return;
    }
    r = xpu::argmax(dev_ctx.x_context(),
                    reinterpret_cast<const XPUType*>(x.data<T>()),
                    out->data<int64_t>(),
                    xdims_vec,
                    axis_val);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        errors::External("XPU argmax kernel return wrong value[%d %s].",
                         r,
                         XPUAPIErrorMsg[r]));
  } else {
    DenseTensor out_int64;
    out_int64.Resize(out->dims());
    dev_ctx.template Alloc<int64_t>(&out_int64);
    if (x.dims().size() == 0) {
      xpu::constant(dev_ctx.x_context(),
                    out_int64.data<int64_t>(),
                    x.numel(),
                    static_cast<int64_t>(0));
    } else {
      r = xpu::argmax(dev_ctx.x_context(),
                      reinterpret_cast<const XPUType*>(x.data<T>()),
                      out_int64.data<int64_t>(),
                      xdims_vec,
                      axis_val);
    }

    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        errors::External("XPU argmax kernel return wrong value[%d %s].",
                         r,
                         XPUAPIErrorMsg[r]));
    dev_ctx.template Alloc<int>(out);
    r = xpu::cast_v2<int64_t, int>(dev_ctx.x_context(),
                                   out_int64.data<int64_t>(),
                                   out->data<int>(),
                                   out_int64.numel());
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        errors::External("XPU cast kernel return wrong value[%d %s].",
                         r,
                         XPUAPIErrorMsg[r]));
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(argmax,
                   XPU,
                   ALL_LAYOUT,
                   phi::ArgMaxKernel,
                   float,
                   int,
                   phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
