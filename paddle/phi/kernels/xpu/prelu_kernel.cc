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

#include "paddle/phi/kernels/prelu_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void PReluKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& alpha,
                 const std::string& data_format,
                 const std::string& mode,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const T* x_ptr = x.data<T>();
  const T* alpha_ptr = alpha.data<T>();

  T* y_ptr = dev_ctx.template Alloc<T>(out);

  auto x_dim = x.dims();
  auto x_rank = x_dim.size();
  std::vector<int64_t> x_shape(x_rank);

  if (x_rank == 0) {
    x_shape = std::vector<int64_t>({1});
  } else {
    for (int i = 0; i < x_rank; i++) {
      x_shape[i] = x_dim[i];
    }
  }

  auto alpha_dim = alpha.dims();
  auto alpha_rank = alpha_dim.size();
  std::vector<int64_t> alpha_shape(x_rank, 1);  // same size with x_shape

  if (x_rank == 0) {
    alpha_shape = std::vector<int64_t>({1});
  } else {
    for (int i = 0; i < alpha_rank; i++) {
      alpha_shape[i] = alpha_dim[i];
    }
  }

  int r = xpu::prelu(dev_ctx.x_context(),
                     reinterpret_cast<const XPUType*>(x_ptr),
                     reinterpret_cast<const XPUType*>(alpha_ptr),
                     reinterpret_cast<XPUType*>(y_ptr),
                     x_shape,
                     alpha_shape);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "prelu");
}
}  // namespace phi

PD_REGISTER_KERNEL(
    prelu, XPU, ALL_LAYOUT, phi::PReluKernel, float, phi::dtype::float16) {}
