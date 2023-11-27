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

#include "paddle/phi/kernels/reduce_min_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/xpu/reduce.h"

namespace phi {

template <typename T, typename Context>
void ReduceMinGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out,
                         const DenseTensor& out_grad,
                         const IntArray& dims_arr,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  reduce_all = recompute_reduce_all(x, dims_arr, reduce_all);
  auto dims = dims_arr.GetData();

  dev_ctx.template Alloc<T>(x_grad);
  const T* x_data = x.data<T>();
  const T* out_data = out.data<T>();
  const T* out_grad_data = out_grad.data<T>();
  auto* x_grad_data = x_grad->data<T>();
  const auto& input_dim_size = x.dims().size();
  std::vector<int> true_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      true_dims.push_back(dims[i] + input_dim_size);
    } else {
      true_dims.push_back(dims[i]);
    }
  }
  std::vector<int> ydims(input_dim_size);
  std::vector<int> xdims((input_dim_size));
  std::set<int> dims_set(true_dims.begin(), true_dims.end());
  for (auto i = 0; i < input_dim_size; i++) {
    xdims[i] = x.dims()[i];
    if (dims_set.find(i) != dims_set.end() || reduce_all) {
      ydims[i] = 1;
    } else {
      ydims[i] = x.dims()[i];
    }
  }

  T* brocast1 = nullptr;
  T* brocast2 = nullptr;
  bool* equal = nullptr;

  xpu::Context* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);

  brocast1 = RAII_GUARD.alloc_l3_or_gm<T>(x.numel());
  PADDLE_ENFORCE_XDNN_NOT_NULL(brocast1);

  brocast2 = RAII_GUARD.alloc_l3_or_gm<T>(x.numel());
  PADDLE_ENFORCE_XDNN_NOT_NULL(brocast2);

  equal = RAII_GUARD.alloc_l3_or_gm<bool>(x.numel());
  PADDLE_ENFORCE_XDNN_NOT_NULL(equal);

  // use [1] to replace [], because xpu not support []
  if (xdims.size() == 0) {
    xdims = std::vector<int>({1});
  }
  if (ydims.size() == 0) {
    ydims = std::vector<int>({1});
  }

  // step 1. brocast out and out_grad
  int r =
      xpu::broadcast<T>(dev_ctx.x_context(), out_data, brocast1, ydims, xdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");

  r = xpu::broadcast<T>(
      dev_ctx.x_context(), out_grad_data, brocast2, ydims, xdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");

  // step 2. comparse out_brocast and x
  r = xpu::equal<T>(dev_ctx.x_context(), x_data, brocast1, equal, x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "equal");
  // step 3. get x_grad
  r = xpu::constant<T>(dev_ctx.x_context(), brocast1, x.numel(), 0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  r = xpu::select<T>(dev_ctx.x_context(),
                     equal,
                     brocast2,
                     brocast1,
                     x_grad_data,
                     xdims,
                     xdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "select");

  if (dev_ctx.x_context()->xpu_stream) {
    dev_ctx.Wait();
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(min_grad, XPU, ALL_LAYOUT, phi::ReduceMinGradKernel, float) {
}
