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

namespace phi {

template <typename TX, typename TO, typename Context>
void CastSumRawGradKernelImpl(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& out_grad,
                              const IntArray& dims_arr,
                              bool keep_dim,
                              bool reduce_all,
                              DenseTensor* x_grad) {
  dev_ctx.template Alloc<TX>(x_grad);
  auto* x_grad_data = x_grad->data<TX>();

  DenseTensor x_grad_tmp;
  DenseTensorMeta x_grad_meta(
      out_grad.dtype(), x_grad->dims(), x_grad->layout());
  x_grad_tmp.set_meta(x_grad_meta);
  TO* x_grad_tmp_data = dev_ctx.template Alloc<TO>(&x_grad_tmp);

  reduce_all = recompute_reduce_all(x, dims_arr, reduce_all);
  auto dims = dims_arr.GetData();
  const auto* out_data = out_grad.data<TO>();

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

  int r1 = xpu::broadcast<TO>(
      dev_ctx.x_context(), out_data, x_grad_tmp_data, ydims, xdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r1, "broadcast");

  int r2 = xpu::cast<TO, TX>(
      dev_ctx.x_context(), x_grad_tmp_data, x_grad_data, x_grad->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r2, "cast");
}

template <typename T, typename Context>
void ReduceSumGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out_grad,
                         const IntArray& dims_arr,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  if (out_grad.dtype() != x.dtype() &&
      out_grad.dtype() == phi::DataType::INT64) {
    if (x.dtype() == phi::DataType::INT32 &&
        out_grad.dtype() == phi::DataType::INT64) {
      CastSumRawGradKernelImpl<int32_t, int64_t, Context>(
          dev_ctx, x, out_grad, dims_arr, keep_dim, reduce_all, x_grad);
    } else if (x.dtype() == phi::DataType::BOOL &&
               out_grad.dtype() == phi::DataType::INT64) {
      CastSumRawGradKernelImpl<bool, int64_t, Context>(
          dev_ctx, x, out_grad, dims_arr, keep_dim, reduce_all, x_grad);
    }
  } else {
    using XPUType = typename XPUTypeTrait<T>::Type;
    reduce_all = recompute_reduce_all(x, dims_arr, reduce_all);
    auto dims = dims_arr.GetData();
    dev_ctx.template Alloc<XPUType>(x_grad);
    const auto* out_data = out_grad.data<XPUType>();
    auto* x_grad_data = x_grad->data<XPUType>();
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

    int r = xpu::broadcast<XPUType>(
        dev_ctx.x_context(), out_data, x_grad_data, ydims, xdims);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sum_grad, XPU, ALL_LAYOUT, phi::ReduceSumGradKernel, float) {
}
