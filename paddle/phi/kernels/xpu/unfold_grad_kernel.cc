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

#include "paddle/phi/kernels/unfold_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/unfold_functor.h"

#include "paddle/phi/kernels/xpu/xpu_mem_util.h"
namespace phi {

template <typename T, typename Context>
void UnfoldGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const std::vector<int>& kernel_sizes,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  FLAGS_limited_idle_chunk = true;
  ctx.template Alloc<T>(x_grad);
  const std::string data_format = phi::DataLayoutToString(x.layout());
  bool is_nchw = data_format == "NCHW";
  PADDLE_ENFORCE_EQ(is_nchw,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Unfold grad op only supports datalayout == NCHW"));

  std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
  int n = x_dims[0];
  int c = x_dims[1];
  int h = x_dims[2];
  int w = x_dims[3];

  std::vector<int64_t> kernel_sizes_l(kernel_sizes.begin(), kernel_sizes.end());
  std::vector<int64_t> strides_l(strides.begin(), strides.end());
  std::vector<int64_t> paddings_l(paddings.begin(), paddings.end());
  std::vector<int64_t> dilations_l(dilations.begin(), dilations.end());

  int r = xpu::col2im_v2(ctx.x_context(),
                         reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                         reinterpret_cast<XPUType*>(x_grad->data<T>()),
                         n,
                         c,
                         h,
                         w,
                         kernel_sizes_l,
                         strides_l,
                         paddings_l,
                         dilations_l,
                         is_nchw);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "col2im_v2");
}

}  // namespace phi

PD_REGISTER_KERNEL(unfold_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::UnfoldGradKernel,
                   float,
                   phi::dtype::float16) {}
