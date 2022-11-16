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

#include "paddle/phi/kernels/reduce_mean_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/xpu/reduce.h"

namespace phi {

template <typename T, typename Context>
void ReduceMeanGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& out_grad,
                          const IntArray& dims_arr,
                          bool keep_dim,
                          bool reduce_all,
                          DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(x_grad);
  const XPUType* dy_data = reinterpret_cast<const XPUType*>(out_grad.data<T>());

  XPUType* x_data = reinterpret_cast<XPUType*>(x_grad->data<T>());

  auto reduce_dims = dims_arr.GetData();

  std::vector<int> xdims = vectorize<int>(x.dims());
  std::vector<int> ydims = vectorize<int>(out_grad.dims());

  int reduce_numel = 1;
  if (reduce_all) {
    reduce_dims.clear();
    for (size_t d = 0; d < xdims.size(); ++d) {
      reduce_dims.push_back(static_cast<int>(d));
    }
  }
  for (auto& d : reduce_dims) {
    if (d < 0) {
      d = d + xdims.size();
    }
    reduce_numel *= xdims[d];
  }

  if (keep_dim != true) {
    sort(reduce_dims.begin(), reduce_dims.end());
    for (auto& d : reduce_dims) {
      ydims.insert(ydims.begin() + d, 1);
    }
  }

  float val = 1.0f / static_cast<float>(reduce_numel);

  int r = xpu::constant(
      dev_ctx.x_context(), x_data, x.numel(), static_cast<XPUType>(val));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  // use [1] to replace [], because xpu not support []
  if (xdims.size() == 0) {
    xdims = std::vector<int>({1});
  }
  if (ydims.size() == 0) {
    ydims = std::vector<int>({1});
  }

  r = xpu::broadcast_mul(
      dev_ctx.x_context(), x_data, dy_data, x_data, xdims, ydims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    mean_grad, XPU, ALL_LAYOUT, phi::ReduceMeanGradKernel, float) {}
