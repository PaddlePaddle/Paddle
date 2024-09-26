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

#include "paddle/phi/kernels/reduce_max_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/xpu/reduce.h"

namespace phi {

template <typename T, typename Context>
void ReduceMaxGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out,
                         const DenseTensor& out_grad,
                         const IntArray& dims_arr,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  reduce_all = recompute_reduce_all(x, dims_arr, reduce_all);
  auto dims = dims_arr.GetData();

  dev_ctx.template Alloc<T>(x_grad);
  const XPUDataType* x_data = reinterpret_cast<const XPUDataType*>(x.data<T>());
  const XPUDataType* out_data =
      reinterpret_cast<const XPUDataType*>(out.data<T>());
  const XPUDataType* out_grad_data =
      reinterpret_cast<const XPUDataType*>(out_grad.data<T>());
  XPUDataType* x_grad_data = reinterpret_cast<XPUDataType*>(x_grad->data<T>());
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

  XPUDataType* broadcast1 = nullptr;
  XPUDataType* broadcast2 = nullptr;
  bool* equal = nullptr;

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  broadcast1 = RAII_GUARD.alloc_l3_or_gm<XPUDataType>(x.numel());
  PADDLE_ENFORCE_NOT_NULL(
      broadcast1, errors::ResourceExhausted("XPU has no enough memory"));

  equal = RAII_GUARD.alloc_l3_or_gm<bool>(x.numel());
  PADDLE_ENFORCE_NOT_NULL(
      equal, errors::ResourceExhausted("XPU has no enough memory"));

  broadcast2 = RAII_GUARD.alloc_l3_or_gm<XPUDataType>(x.numel());
  PADDLE_ENFORCE_NOT_NULL(
      broadcast2, errors::ResourceExhausted("XPU has no enough memory"));

  // use [1] to replace [], because xpu not support []
  if (xdims.size() == 0) {
    xdims = std::vector<int>({1});
  }
  if (ydims.size() == 0) {
    ydims = std::vector<int>({1});
  }

  // step 1. broadcast out and out_grad
  int r =
      xpu::broadcast(dev_ctx.x_context(), out_data, broadcast1, ydims, xdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");

  r = xpu::broadcast(
      dev_ctx.x_context(), out_grad_data, broadcast2, ydims, xdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");

  // step 2. compare out_broadcast and x
  r = xpu::equal(dev_ctx.x_context(), x_data, broadcast1, equal, x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "equal");
  // step 3. get x_grad
  r = xpu::constant(
      dev_ctx.x_context(), broadcast1, x.numel(), static_cast<XPUDataType>(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  r = xpu::select(dev_ctx.x_context(),
                  equal,
                  broadcast2,
                  broadcast1,
                  x_grad_data,
                  xdims,
                  xdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "select");
}

}  // namespace phi

PD_REGISTER_KERNEL(max_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ReduceMaxGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
