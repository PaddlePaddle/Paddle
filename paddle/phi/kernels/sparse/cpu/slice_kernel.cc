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

#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SliceCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const phi::IntArray& axes_arr,
                    const phi::IntArray& starts_arr,
                    const phi::IntArray& ends_arr,
                    SparseCooTensor* out) {
  const phi::DDim& x_dims = x.dims();

  std::vector<int64_t> axes = axes_arr.GetData();
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();

  int64_t rank = int64_t(x_dims.size());
  // Ensure that each axis in axes is between [0, rank-1).
  for (auto& axis : axes) {
    if (axis < 0) {
      axis = std::max(int64_t(0), axis + rank);
    }
    axis = std::min(axis, rank - 1);
  }

  // Step1: Check
  PADDLE_ENFORCE_EQ(
      axes.size(),
      starts.size(),
      phi::errors::InvalidArgument(
          "The length of axes (%d) and length of starts (%d) should be same.",
          axes.size(),
          starts.size()));
  PADDLE_ENFORCE_EQ(
      axes.size(),
      ends.size(),
      phi::errors::InvalidArgument(
          "The length of axes (%d) and length of ends (%d) should be same.",
          axes.size(),
          ends.size()));

  // update starts and ends
  funcs::CheckAndUpdateSliceAttrs<int64_t>(x_dims, axes, &starts, &ends);

  // Step2: Infer output dims
  auto out_dims = funcs::GetSliceDims<int64_t>(
      x_dims, axes, starts, ends, nullptr, nullptr);

  // Step3: Get out_nnz (the number of non-zero elements in output)
  const int64_t x_nnz = x.nnz();
  int64_t out_nnz = 0;
  const auto* x_indices_data = x.indices().data<int64_t>();
  for (int64_t j = 0; j < x_nnz; ++j) {
    bool hit = true;
    for (size_t ii = 0; ii < axes.size(); ++ii) {
      auto item = x_indices_data[ii * x_nnz + j];
      if (!(starts[ii] <= item && item < ends[ii])) {
        hit = false;
        break;
      }
    }
    if (!hit) continue;
    out_nnz++;
  }

  // Step4: Get the values and indices of output
  auto sparse_dim = static_cast<int64_t>(x.sparse_dim());
  DenseTensor out_indices =
      phi::Empty<int64_t, Context>(dev_ctx, {sparse_dim, out_nnz});
  DenseTensor out_values = phi::Empty<T, Context>(dev_ctx, {out_nnz});

  auto* out_indices_data = out_indices.data<int64_t>();
  auto* out_values_data = out_values.data<T>();
  const auto* x_values_data = x.values().data<T>();
  int64_t index = 0;
  for (int64_t j = 0; j < x_nnz && index < out_nnz; ++j) {
    bool hit = true;
    for (size_t ii = 0; ii < axes.size(); ++ii) {
      auto item = x_indices_data[ii * x_nnz + j];
      if (!(starts[ii] <= item && item < ends[ii])) {
        hit = false;
        break;
      }
    }
    if (!hit) continue;
    // set value
    out_values_data[index] = x_values_data[j];
    // set coordinate
    for (int64_t i = 0; i < sparse_dim; ++i) {
      out_indices_data[i * out_nnz + index] = x_indices_data[i * x_nnz + j];
    }
    for (size_t ii = 0; ii < axes.size(); ++ii) {
      auto i = axes[ii];
      out_indices_data[i * out_nnz + index] -= starts[ii];
    }
    index++;
  }
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(slice_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SliceCooKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
