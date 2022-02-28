/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/transfer_layout_kernel.h"

#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

std::vector<int> GetAxis(const DataLayout& from, const DataLayout& to) {
  PADDLE_ENFORCE_NE(
      from,
      to,
      phi::errors::InvalidArgument(
          "Layout transform should transform between different layout."));
  if (from == DataLayout::NCHW && to == DataLayout::NHWC) {
    return {0, 2, 3, 1};
  } else if (from == DataLayout::NHWC && to == DataLayout::NCHW) {
    return {0, 3, 1, 2};
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("Unsupported layout transform."));
  }
}

template <typename T, typename Context>
void CastDataLayout(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<int>& axis,
                    DenseTensor* out) {
  funcs::Transpose<Context, T, 4> trans4;
  trans4(dev_ctx, x, out, axis);
}

template <typename Context>
void TransferLayoutKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          DataLayout dst_layout,
                          DenseTensor* out) {
  auto src_dim = x.dims();

  auto axis = GetAxis(x.layout(), dst_layout);

  std::vector<int64_t> dst_dim;
  dst_dim.resize(axis.size());
  for (size_t i = 0; i < axis.size(); i++) {
    dst_dim[i] = src_dim[axis[i]];
  }

  out->ResizeAndAllocate(phi::make_ddim(dst_dim));

  PD_VISIT_ALL_TYPES(x.dtype(), "CastDataLayout", ([&] {
                       CastDataLayout<data_t, Context>(dev_ctx, x, axis, out);
                     }));
}

}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(phi_transfer_layout,
                           CPU,
                           ALL_LAYOUT,
                           phi::TransferLayoutKernel<phi::CPUContext>,
                           ALL_DTYPE) {}
