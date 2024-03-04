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

#include "paddle/phi/kernels/temporal_shift_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace phi {

template <typename T, typename Context>
void TemporalShiftKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         int seg_num,
                         float shift_ratio,
                         const std::string& data_format_str,
                         DenseTensor* out) {
  auto* input = &x;
  auto* output = out;
  int t = seg_num;
  const DataLayout data_layout = common::StringToDataLayout(data_format_str);

  const int nt = input->dims()[0];
  const int n = nt / t;
  const int c =
      (data_layout == DataLayout::kNCHW ? input->dims()[1] : input->dims()[3]);
  const int h =
      (data_layout == DataLayout::kNCHW ? input->dims()[2] : input->dims()[1]);
  const int w =
      (data_layout == DataLayout::kNCHW ? input->dims()[3] : input->dims()[2]);

  DDim out_dims =
      (data_layout == DataLayout::kNCHW ? common::make_ddim({nt, c, h, w})
                                        : common::make_ddim({nt, h, w, c}));
  const T* input_data = input->data<T>();
  output->Resize(out_dims);
  T* output_data = dev_ctx.template Alloc<T>(output);

  if (data_layout == DataLayout::kNCHW) {
    int r = xpu::temporal_shift(dev_ctx.x_context(),
                                input_data,
                                output_data,
                                n,
                                c,
                                h,
                                w,
                                t,
                                shift_ratio,
                                false);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "temporal_shift");
  } else {
    int r = xpu::temporal_shift(dev_ctx.x_context(),
                                input_data,
                                output_data,
                                n,
                                c,
                                h,
                                w,
                                t,
                                shift_ratio,
                                true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "temporal_shift");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    temporal_shift, XPU, ALL_LAYOUT, phi::TemporalShiftKernel, float) {}
