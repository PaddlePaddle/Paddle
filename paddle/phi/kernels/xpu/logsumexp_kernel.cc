// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void XPULogsumexpKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<int>& axis,
                        bool keepdim,
                        bool reduce_all,
                        DenseTensor* out) {
  auto* input = &x;
  auto* output = out;

  const auto& input_dim_size = input->dims().size();
  // The dims has full dim, set the reduce_all is True
  reduce_all |= (static_cast<int>(axis.size()) == input_dim_size);

  const T* input_data = input->data<T>();
  T* output_data = dev_ctx.template Alloc<T>(output);

  std::vector<int> axis_shape;
  std::vector<int> xdims(input_dim_size);
  for (int i = 0; i < input_dim_size; ++i) {
    xdims[i] = input->dims()[i];
  }
  if (reduce_all) {
    for (int i = 0; i < input_dim_size; ++i) {
      axis_shape.push_back(i);
    }
  } else {
    for (size_t i = 0; i < axis.size(); ++i) {
      int rdim = axis[i] < 0 ? axis[i] + input_dim_size : axis[i];
      axis_shape.push_back(rdim);
    }
  }

  int r = xpu::logsumexp<T>(
      dev_ctx.x_context(), input_data, output_data, xdims, axis_shape);
  PADDLE_ENFORCE_EQ(
      r,
      xpu::Error_t::SUCCESS,
      phi::errors::External("XPU logsumexp kernel error! error value[%d %]",
                            r,
                            XPUAPIErrorMsg[r]));
}
}  // namespace phi

PD_REGISTER_KERNEL(logsumexp, XPU, ALL_LAYOUT, phi::XPULogsumexpKernel, float) {
}
