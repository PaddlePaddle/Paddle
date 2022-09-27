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
#include "paddle/phi/kernels/label_smooth_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
namespace phi {

template <typename T, typename Context>
void LabelSmoothKernel(const Context& ctx,
                       const DenseTensor& label,
                       const paddle::optional<DenseTensor>& prior_dist,
                       float epsilon,
                       DenseTensor* out) {
  auto label_dim = label.dims()[label.dims().size() - 1];
  auto ptr = ctx.template Alloc<T>(out);
  if (prior_dist.is_initialized()) {
    PADDLE_THROW(
        phi::errors::External("XPU doesn't support dist label smooth"));
  } else {
    int r = xpu::label_smooth<T>(ctx.x_context(),
                                 label.data<T>(),
                                 ptr,
                                 label.numel(),
                                 epsilon,
                                 label_dim);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        phi::errors::External("XPU API(label_smooth) return wrong "
                              "value[%d %s]",
                              r,
                              XPUAPIErrorMsg[r]));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    label_smooth, XPU, ALL_LAYOUT, phi::LabelSmoothKernel, float) {}
