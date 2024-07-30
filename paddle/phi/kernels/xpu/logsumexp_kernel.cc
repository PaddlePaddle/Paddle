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
#include "paddle/phi/kernels/logsumexp_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void LogsumexpKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     bool keepdim,
                     bool reduce_all,
                     DenseTensor* out) {
  auto* output = out;
  using XPUType = typename XPUTypeTrait<T>::Type;
  int input_dim_size = x.dims().size();
  // The dims has full dim, set the reduce_all is True
  reduce_all |= (static_cast<int>(axis.size()) == input_dim_size);

  auto input_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto output_data =
      reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(output));

  std::vector<int> axis_shape;
  std::vector<int> xdims = common::vectorize<int>(x.dims());
  if (input_dim_size == 0 && x.numel() != 0) {
    // 0-d Tensor.
    xdims = {1};
    input_dim_size = 1;
    reduce_all = true;
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
  for (size_t i = 0; i < xdims.size(); ++i) {
    PADDLE_ENFORCE_LT(0,
                      xdims[i],
                      errors::InvalidArgument(
                          "The dims of Input(X) should be greater than 0."));
  }

  int r = xpu::logsumexp<XPUType>(
      dev_ctx.x_context(), input_data, output_data, xdims, axis_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "logsumexp");
}
}  // namespace phi

PD_REGISTER_KERNEL(logsumexp,
                   XPU,
                   ALL_LAYOUT,
                   phi::LogsumexpKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
