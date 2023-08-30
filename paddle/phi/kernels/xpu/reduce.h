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

#pragma once
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/xpu/reduce_util.h"

namespace phi {

static void GetReduceDims(const DDim& xdims,
                          const std::vector<int64_t>& dims,
                          bool reduce_all,
                          std::vector<int>* reduce_dims) {
  const auto& input_dim_size = xdims.size();
  std::vector<int> true_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      true_dims.push_back(dims[i] + input_dim_size);
    } else {
      true_dims.push_back(dims[i]);
    }
  }

  if (reduce_all) {
    for (int i = 0; i < input_dim_size; ++i) {
      reduce_dims->push_back(i);
    }
  } else {
    std::set<int> dims_set(true_dims.begin(), true_dims.end());
    for (auto i = 0; i < input_dim_size; i++) {
      if (dims_set.find(i) != dims_set.end()) {
        if (xdims[i] != 1) {
          reduce_dims->push_back(i);
        }
      }
    }
  }
}

template <typename Context, typename T>
int XPUReduce(const Context& dev_ctx,
              const DenseTensor& x,
              const std::vector<int64_t>& dims,
              bool keep_dim,
              bool reduce_all,
              DenseTensor* out,
              std::function<int(xpu::Context*,
                                const T*,
                                T*,
                                const std::vector<int>&,
                                const std::vector<int>&)> func) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  dev_ctx.template Alloc<T>(out);

  const auto* x_data = x.data<T>();
  auto* y_data = out->data<T>();

  const auto& input_dim_size = x.dims().size();
  std::vector<int> xdims(input_dim_size);
  for (int i = 0; i < input_dim_size; ++i) {
    xdims[i] = x.dims()[i];
  }

  std::vector<int> reduce_dims;
  GetReduceDims(x.dims(), dims, reduce_all, &reduce_dims);

  int r = xpu::SUCCESS;
  if (reduce_dims.size() == 0) {
    r = xpu::copy<XPUType>(dev_ctx.x_context(),
                           reinterpret_cast<const XPUType*>(x_data),
                           reinterpret_cast<XPUType*>(y_data),
                           x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else {
    r = func(dev_ctx.x_context(), x_data, y_data, xdims, reduce_dims);
  }
  return r;
}

template <typename DeviceContext, typename T, typename OutT, typename Functor>
void ReduceKernelImpl(const DeviceContext& dev_ctx,
                      const phi::DenseTensor& input,
                      phi::DenseTensor* output,
                      const std::vector<int>& xdims,
                      const std::vector<int>& reduce_dims) {
  using XPUType = typename XPUTypeTrait<OutT>::Type;
  dev_ctx.template Alloc<OutT>(output);
  const auto* x_data = input.data<OutT>();
  auto* y_data = output->data<OutT>();
  if (reduce_dims.size() == 0) {
    int r = xpu::copy<XPUType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x_data),
                               reinterpret_cast<XPUType*>(y_data),
                               input.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else {
    Functor func;
    func(dev_ctx.x_context(), x_data, y_data, xdims, reduce_dims);
  }
}

template <typename DeviceContext, typename T, typename Functor>
void XPUReduce(const DeviceContext& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               bool reduce_all,
               DataType out_dtype,
               DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);

  const auto& input_dim_size = x.dims().size();
  std::vector<int> xdims(input_dim_size);
  for (int i = 0; i < input_dim_size; ++i) {
    xdims[i] = x.dims()[i];
  }

  std::vector<int> reduce_dims;
  GetReduceDims(x.dims(), dims, reduce_all, &reduce_dims);

  // no need to cast dtype
  if (out_dtype == phi::DataType::UNDEFINED || out_dtype == x.dtype()) {
    // do reduce sum
    PD_VISIT_XPU_REDUCE_TYPES(
        x.dtype(), "ReduceKernelImpl", ([&] {
          phi::ReduceKernelImpl<DeviceContext, T, data_t, Functor>(
              dev_ctx, x, out, xdims, reduce_dims);
        }));
  } else {
    // cast x tensor to out_dtype
    auto tmp_tensor = phi::Cast<T, DeviceContext>(dev_ctx, x, out_dtype);

    // do reduce sum
    PD_VISIT_XPU_REDUCE_TYPES(
        out_dtype, "ReduceKernelImpl", ([&] {
          phi::ReduceKernelImpl<DeviceContext, T, data_t, Functor>(
              dev_ctx, tmp_tensor, out, xdims, reduce_dims);
        }));

    if (dev_ctx.x_context()->xpu_stream) {
      dev_ctx.Wait();
    }
  }
}

}  // namespace phi
