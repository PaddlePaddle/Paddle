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

#include "paddle/phi/kernels/concat_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace phi {

template <typename T, typename Context>
void ConcatGradKernel(const Context& dev_ctx,
                      const std::vector<const DenseTensor*>& x,
                      const DenseTensor& out_grad,
                      const Scalar& axis_scalar,
                      std::vector<DenseTensor*> x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto outs = x_grad;
  {
    auto dx = outs;
    for (size_t i = 0; i < dx.size(); ++i) {
      if (dx[i] != nullptr) {
        dx[i]->set_lod(x[i]->lod());
      }
    }
  }
  PADDLE_ENFORCE_NE(
      x[0],
      nullptr,
      common::errors::InvalidArgument("The input should not be null."));
  auto axis = axis_scalar.to<int>();
  axis = phi::funcs::ComputeAxis(static_cast<int64_t>(axis),
                                 static_cast<int64_t>(x[0]->dims().size()));
  // get output tensor that the name is not kEmptyVarName
  std::vector<XPUType*> ptrs(outs.size());
  for (size_t j = 0; j < outs.size(); ++j) {
    if (outs[j] && outs[j]->numel() != 0UL) {
      dev_ctx.template Alloc<T>(outs[j]);
      ptrs[j] = reinterpret_cast<XPUType*>(outs[j]->data<T>());
    } else {
      ptrs[j] = nullptr;
    }
  }
  PADDLE_ENFORCE_GE(axis,
                    0,
                    common::errors::InvalidArgument(
                        "concat_grad: axis should be larger than or "
                        "equal to 0, but received axis is %d.",
                        axis));
  PADDLE_ENFORCE_LT(axis,
                    out_grad.dims().size(),
                    common::errors::InvalidArgument(
                        "concat_grad: axis should be less than x[0]->dims()!"
                        "But received axis is %d, while x[0]->dims()"
                        "size is %d.",
                        axis,
                        out_grad.dims().size()));

  auto input_dims = x[0]->dims();
  std::vector<int> split_list(x.size());
  std::vector<int> xdims_list(input_dims.size());
  int total_length = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    split_list[i] = x[i]->dims()[axis];
    total_length += x[i]->dims()[axis];
  }
  for (int i = 0; i < input_dims.size(); ++i) {
    if (i == axis) {
      continue;
    }
    xdims_list[i] = input_dims[i];
  }
  xdims_list[axis] = total_length;

  int r =
      xpu::split<XPUType>(dev_ctx.x_context(),
                          reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                          ptrs,
                          xdims_list,
                          split_list,
                          axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "concat_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(concat_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ConcatGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
