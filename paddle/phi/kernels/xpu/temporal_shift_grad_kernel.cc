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

#include "paddle/phi/kernels/temporal_shift_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace phi {

template <typename T, typename Context>
void TemporalShiftGradKernel(const Context& dev_ctx,
                             const DenseTensor& out_grad,
                             int seg_num,
                             float shift_ratio,
                             const std::string& data_format_str,
                             DenseTensor* x_grad) {
  auto* input_grad = x_grad;
  auto* output_grad = &out_grad;
  int t = seg_num;
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_format_str);

  const int nt = output_grad->dims()[0];
  const int n = nt / t;
  const int c = (data_layout == DataLayout::kNCHW ? output_grad->dims()[1]
                                                  : output_grad->dims()[3]);
  const int h = (data_layout == DataLayout::kNCHW ? output_grad->dims()[2]
                                                  : output_grad->dims()[1]);
  const int w = (data_layout == DataLayout::kNCHW ? output_grad->dims()[3]
                                                  : output_grad->dims()[2]);

  DDim in_grad_dims =
      (data_layout == DataLayout::kNCHW ? phi::make_ddim({nt, c, h, w})
                                        : phi::make_ddim({nt, h, w, c}));
  const T* output_grad_data = output_grad->data<T>();
  input_grad->Resize(in_grad_dims);
  T* input_grad_data = dev_ctx.template Alloc<T>(input_grad);

  if (data_layout == DataLayout::kNCHW) {
    int r = xpu::temporal_shift_grad(dev_ctx.x_context(),
                                     output_grad_data,
                                     input_grad_data,
                                     n,
                                     c,
                                     h,
                                     w,
                                     t,
                                     shift_ratio,
                                     false);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "temporal_shift_grad");
  } else {
    int r = xpu::temporal_shift_grad(dev_ctx.x_context(),
                                     output_grad_data,
                                     input_grad_data,
                                     n,
                                     c,
                                     h,
                                     w,
                                     t,
                                     shift_ratio,
                                     true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "temporal_shift_grad");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    temporal_shift_grad, XPU, ALL_LAYOUT, phi::TemporalShiftGradKernel, float) {
}
