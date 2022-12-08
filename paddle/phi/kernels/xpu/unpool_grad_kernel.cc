// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/unpool_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void UnpoolGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& indices,
                      const DenseTensor& out,
                      const DenseTensor& out_grad,
                      const std::vector<int>& ksize,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const IntArray& output_size,
                      const std::string& data_format,
                      DenseTensor* x_grad) {
  T* input_grad_data = dev_ctx.template Alloc<T>(x_grad);
  const T* output_grad_data = out_grad.data<T>();

  const int64_t n = x.dims()[0];
  const int64_t c = x.dims()[1];
  const int64_t xh = x.dims()[2];
  const int64_t xw = x.dims()[3];

  const int* indices_data = indices.data<int>();

  int r = xpu::max_unpool2d_grad<T>(dev_ctx.x_context(),
                                    output_grad_data,
                                    indices_data,
                                    input_grad_data,
                                    n,
                                    c,
                                    xh,
                                    xw,
                                    ksize,
                                    strides,
                                    paddings,
                                    true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "max_unpool2d_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(unpool_grad, XPU, ALL_LAYOUT, phi::UnpoolGradKernel, float) {
}
