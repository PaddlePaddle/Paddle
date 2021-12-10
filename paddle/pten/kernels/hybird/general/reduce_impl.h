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
#include "paddle/fluid/platform/transform.h"
#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/hybird/eigen/reduce.h"
#include "paddle/pten/kernels/hybird/math/cast_func.h"
namespace pten {
namespace general {

template <typename DeviceContext, typename T, typename Functor>
void Reduce(const DeviceContext& dev_ctx,
            const DenseTensor& x,
            bool reduce_all,
            const std::vector<int64_t>& dims,
            bool keep_dim,
            DataType out_dtype,
            DenseTensor* out) {
  // If the dims has full dim, set the reduce_all is True
  const auto& input_dim_size = x.dims().size();
  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (auto i = 0; i < input_dim_size; ++i) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);

  // no need to cast dtype
  if (out_dtype == pten::DataType::UNDEFINED || out_dtype == x.dtype()) {
    if (out_dtype == pten::DataType::UNDEFINED) {
      out_dtype = x.dtype();
    }
    // do reduce sum
    PD_VISIT_ALL_TYPES(
        out_dtype, "ReduceKernelImpl", ([&] {
          pten::eigen::ReduceKernelImpl<DeviceContext, T, data_t, Functor>(
              dev_ctx, x, out, dims, keep_dim, reduce_all);
        }));
  } else {
    const auto alloc =
        std::make_shared<paddle::experimental::DefaultAllocator>(x.place());
    pten::DenseTensor tmp_tensor = pten::DenseTensor(
        alloc, pten::DenseTensorMeta(out_dtype, x.dims(), x.layout()));

    // cast x tensor to out_dtype first
    PD_VISIT_ALL_TYPES(out_dtype, "CastKernelImpl", ([&] {
                         math::CastKernelImpl<DeviceContext, T, data_t>(
                             dev_ctx, x, &tmp_tensor);
                       }));

    // do reduce sum
    PD_VISIT_ALL_TYPES(
        out_dtype, "ReduceKernelImpl", ([&] {
          pten::eigen::ReduceKernelImpl<DeviceContext, T, data_t, Functor>(
              dev_ctx, tmp_tensor, out, dims, keep_dim, reduce_all);
        }));
  }
}

}  // namespace general

}  // namespace pten
