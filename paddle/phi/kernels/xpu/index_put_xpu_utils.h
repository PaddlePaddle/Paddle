// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"

#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/stack_kernel.h"

namespace phi {
template <typename Context>
void XPUDealWithIndices(const Context& dev_ctx,
                        const std::vector<const DenseTensor*>& int_indices_v,
                        DDim bd_dim,
                        DenseTensor* out) {
  std::vector<DenseTensor> tmp_indices_v;
  for (size_t i = 0; i < int_indices_v.size(); ++i) {
    // Use int64 for all indices Because XPU needs to merge all indices into a
    // single tensor. Same with CPU and GPU.
    DenseTensor casted_index;
    if (int_indices_v[i]->dtype() == DataType::INT32) {
      casted_index =
          phi::Cast<int, Context>(dev_ctx, *int_indices_v[i], DataType::INT64);
    } else {
      casted_index = *int_indices_v[i];
    }

    DenseTensor expanded_index(DataType::INT64);
    if (casted_index.dims() == bd_dim) {
      expanded_index = casted_index;
    } else {
      expanded_index.Resize(bd_dim);
      ExpandKernel<int64_t, Context>(
          dev_ctx,
          casted_index,
          IntArray(common::vectorize<int64_t>(bd_dim)),
          &expanded_index);
    }

    tmp_indices_v.emplace_back(expanded_index);
  }

  auto bd_dim_vec = common::vectorize<int64_t>(bd_dim);
  std::vector<int64_t> stacked_dim_vec(bd_dim.size() + 1);
  std::copy(bd_dim_vec.begin(), bd_dim_vec.end(), stacked_dim_vec.begin());
  stacked_dim_vec.back() = int_indices_v.size();
  out->Resize(common::make_ddim(stacked_dim_vec));

  std::vector<const DenseTensor*> tmp_indices_ptr(tmp_indices_v.size(),
                                                  nullptr);
  for (size_t i = 0; i < tmp_indices_ptr.size(); ++i) {
    tmp_indices_ptr[i] = &tmp_indices_v[i];
  }

  StackKernel<int64_t, Context>(dev_ctx, tmp_indices_ptr, -1, out);
}
}  // namespace phi
