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

template <typename T>
void cumsum_impl(const phi::XPUContext& dev_ctx,
                 const phi::DenseTensor& x_,
                 int axis_val,
                 bool flatten_val,
                 bool reverse_val,
                 bool exclusive_val,
                 phi::DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  if (x_.numel() == 1) {
    int r = xpu::copy<XPUType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x_.data<T>()),
                               reinterpret_cast<XPUType*>(out->data<T>()),
                               x_.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    return;
  }

  std::vector<int64_t> x_shape = common::vectorize<int64_t>(x_.dims());

  if (flatten_val) {
    x_shape = {x_.numel()};
    axis_val = 0;
  } else {
    int x_rank = x_.dims().size();
    PADDLE_ENFORCE_EQ(
        axis_val < x_rank && axis_val >= (0 - x_rank),
        true,
        common::errors::OutOfRange(
            "Attr(axis) is out of range, It's expected "
            "to be in range of [-%d, %d). But received Attr(axis) = %d.",
            x_rank,
            x_rank,
            axis_val));
    if (axis_val < 0) {
      axis_val += x_rank;
    }
  }

  int r = xpu::cumsum<XPUType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x_.data<T>()),
                               reinterpret_cast<XPUType*>(out->data<T>()),
                               x_shape,
                               reverse_val,
                               exclusive_val,
                               axis_val);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cumsum");
}

template <typename InT, typename Context>
struct CumsumKernelVisitor {
  const Context& dev_ctx_;
  const DenseTensor& x_;
  int axis_val_;
  bool flatten_val_;
  bool exclusive_val_;
  bool reverse_val_;
  DenseTensor* out_;

  CumsumKernelVisitor(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      bool flatten,
                      bool exclusive,
                      bool reverse,
                      DenseTensor* out)
      : dev_ctx_(dev_ctx),
        x_(x),
        axis_val_(axis),
        flatten_val_(flatten),
        exclusive_val_(exclusive),
        reverse_val_(reverse),
        out_(out) {}

  template <typename OutT>
  void apply() const {
    DenseTensor x_casted;
    x_casted.Resize(x_.dims());
    dev_ctx_.template Alloc<OutT>(&x_casted);

    using XPUInT = typename XPUTypeTrait<InT>::Type;
    using XPUOutT = typename XPUTypeTrait<OutT>::Type;

    int r_cast = xpu::cast<XPUInT, XPUOutT>(
        dev_ctx_.x_context(),
        reinterpret_cast<const XPUInT*>(x_.data<InT>()),
        reinterpret_cast<XPUOutT*>(x_casted.data<OutT>()),
        x_.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r_cast, "xpu::cast_in_visitor_failed");

    cumsum_impl<OutT>(dev_ctx_,
                      x_casted,
                      axis_val_,
                      flatten_val_,
                      reverse_val_,
                      exclusive_val_,
                      out_);
  }
};

template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  DataType dtype,
                  DenseTensor* out) {
  if (out->dtype() == x.dtype()) {
    cumsum_impl<T>(
        dev_ctx, x, axis.to<int>(), flatten, reverse, exclusive, out);
  } else {
    CumsumKernelVisitor<T, Context> visitor(
        dev_ctx, x, axis.to<int>(), flatten, exclusive, reverse, out);
    phi::VisitDataType(out->dtype(), visitor);
  }
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
