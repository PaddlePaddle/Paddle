// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <set>

#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"

namespace phi {

template <typename DeviceContext, typename T, typename Functor>
void Reduce(const DeviceContext& dev_ctx,
            const DenseTensor& x,
            bool reduce_all,
            const std::vector<int64_t>& dims,
            bool keep_dim,
            DataType out_dtype,
            DenseTensor* out) {
  // If the dims has full dim, set the reduce_all is True
  const int& input_dim_size = x.dims().size();
  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (int i = 0; i < input_dim_size; ++i) {
    if (dims_set.find(i) == dims_set.end() &&
        dims_set.find(i - input_dim_size) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);

  // no need to cast dtype
  if (out_dtype == phi::DataType::UNDEFINED || out_dtype == x.dtype()) {
    // do reduce sum
    PD_VISIT_ALL_TYPES(
        x.dtype(), "ReduceKernelImpl", ([&] {
          phi::funcs::ReduceKernelImpl<DeviceContext, T, data_t, Functor>(
              dev_ctx, x, out, dims, keep_dim, reduce_all);
        }));
  } else {
    // cast x tensor to out_dtype
    auto tmp_tensor = phi::Cast<T, DeviceContext>(dev_ctx, x, out_dtype);

    // do reduce sum
    PD_VISIT_ALL_TYPES(
        out_dtype, "ReduceKernelImpl", ([&] {
          phi::funcs::ReduceKernelImpl<DeviceContext, T, data_t, Functor>(
              dev_ctx, tmp_tensor, out, dims, keep_dim, reduce_all);
        }));
  }
}

template <typename DeviceContext, typename OutT, typename Functor>
void BoolReduceKernel(const DeviceContext& dev_ctx,
                      const phi::DenseTensor& input,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      bool reduce_all,
                      phi::DenseTensor* output) {
  dev_ctx.template Alloc<OutT>(output);

  // The dims has full dim, set the reduce_all is True
  const auto& input_dim_size = input.dims().size();
  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (auto i = 0; i < input_dim_size; i++) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);

  funcs::ReduceKernelImpl<DeviceContext, bool, OutT, Functor>(
      dev_ctx, input, output, dims, keep_dim, reduce_all);
}

}  // namespace phi
