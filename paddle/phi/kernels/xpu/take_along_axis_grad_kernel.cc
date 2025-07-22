// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/take_along_axis_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TakeAlongAxisGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& index,
                             const DenseTensor& out_grad,
                             int axis,
                             DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(x_grad);

  if (x_grad->numel() == 0) {
    return;
  }

  const auto& index_dtype = index.dtype();
  bool index_dtype_match =
      index_dtype == DataType::INT32 || index_dtype == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_dtype_match,
                    true,
                    errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(index_dtype),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  int r = xpu::constant(dev_ctx.x_context(),
                        reinterpret_cast<XPUType*>(x_grad->data<T>()),
                        x_grad->numel(),
                        XPUType(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  auto x_shape = common::vectorize<int64_t>(x.dims());
  auto out_grad_shape = common::vectorize<int64_t>(out_grad.dims());
  auto index_shape = common::vectorize<int64_t>(index.dims());

  if (index_dtype == DataType::INT32) {
    r = xpu::paddle_put_along_axis<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x_grad->data<T>()),
        reinterpret_cast<const XPUType*>(out_grad.data<T>()),
        reinterpret_cast<const int*>(index.data<int>()),
        reinterpret_cast<XPUType*>(x_grad->data<T>()),
        x_shape,
        out_grad_shape,
        index_shape,
        axis,
        1,
        false);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_put_along_axis");
  } else {
    r = xpu::paddle_put_along_axis<XPUType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x_grad->data<T>()),
        reinterpret_cast<const XPUType*>(out_grad.data<T>()),
        reinterpret_cast<const int64_t*>(index.data<int64_t>()),
        reinterpret_cast<XPUType*>(x_grad->data<T>()),
        x_shape,
        out_grad_shape,
        index_shape,
        axis,
        1,
        false);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_put_along_axis");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(take_along_axis_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::TakeAlongAxisGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
