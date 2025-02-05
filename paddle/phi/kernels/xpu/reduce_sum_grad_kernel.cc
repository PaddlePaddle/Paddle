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
#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"

#include <set>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void ReduceSumGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out_grad,
                         const IntArray& dims_arr,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  reduce_all = recompute_reduce_all(x, dims_arr, reduce_all);
  auto dims = dims_arr.GetData();
  dev_ctx.template Alloc(x_grad, x.dtype());
  auto* out_data = reinterpret_cast<const XPUType*>(out_grad.data());
  auto* x_grad_data = reinterpret_cast<XPUType*>(x_grad->data());
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

  // use [1] to replace [], because xpu not support []
  if (xdims.size() == 0) {
    xdims = std::vector<int>({1});
  }
  if (ydims.size() == 0) {
    ydims = std::vector<int>({1});
  }

  if (x.dtype() != out_grad.dtype()) {
    DenseTensorMeta x_grad_meta(
        out_grad.dtype(), x_grad->dims(), x_grad->layout());
    DenseTensor x_grad_tmp =
        phi::Empty<Context>(dev_ctx, std::move(x_grad_meta));
    auto* x_grad_tmp_data = reinterpret_cast<XPUType*>(x_grad_tmp.data());

    int r = xpu::broadcast<XPUType>(
        dev_ctx.x_context(), out_data, x_grad_tmp_data, ydims, xdims);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");

    phi::CastKernel<T>(dev_ctx, x_grad_tmp, x.dtype(), x_grad);
  } else {
    int r = xpu::broadcast<XPUType>(
        dev_ctx.x_context(), out_data, x_grad_data, ydims, xdims);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sum_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ReduceSumGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int,
                   bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
