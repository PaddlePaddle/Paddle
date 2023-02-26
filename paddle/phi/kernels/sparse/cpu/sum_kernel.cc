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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SumCooKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCooTensor* out) {
  unsigned int n_dim = axis.size();
  // create out sparse tensor
  const DDim& x_dims = x.dims();
  const DenseTensor& x_indices = x.indices();
  const DenseTensor& x_values = x.values();
  DDim out_dims;
  DenseTensor out_indices;
  DenseTensor out_values;
  if (n_dim == 0) {
    std::vector<int64_t> out_indices_shape;
    if (keep_dim) {
      out_dims = make_ddim(std::vector<int64_t>(x_dims.size(), 1));
      out_indices_shape = {x_dims.size(), 1};
      out_indices = Empty<int64_t, Context>(dev_ctx, out_indices_shape);
      auto* out_indices_data = out_indices.data<int64_t>();
      for (auto i = 0; i < x_dims.size(); ++i) {
        out_indices_data[i] = 0;
      }
    } else {
      out_dims = make_ddim({1});
      out_indices_shape = {1, 1};
      out_indices = Empty<int64_t, Context>(dev_ctx, out_indices_shape);
      auto* out_indices_data = out_indices.data<int64_t>();
      out_indices_data[0] = 0;
    }
    out_values = phi::Sum<T>(dev_ctx, x.values(), {}, dtype, true);
  } else {
    const auto* x_indices_data = x_indices.data<int64_t>();
    const auto* x_values_data = x_values.data<T>();
    std::map<std::vector<int>, std::vector<int64_t>> map_indices;
    for (auto j = 0; j < x_indices.dims()[1]; ++j) {
      std::vector<int> pos;
      for (int i = 0; i < x_indices.dims()[0]; ++i) {
        pos.push_back(x_indices_data[j + i * x_indices.dims()[1]]);
      }
      if (map_indices.find(pos) == map_indices.end()) {
        map_indices[pos] = {j};
      } else {
        map_indices[pos].push_back(j);
      }
    }

    if (keep_dim) {
      out_indices = Empty<int64_t, Context>(
          dev_ctx, {x_dims.size(), static_cast<int>(map_indices.size())});
    } else {
      out_indices = Empty<int64_t, Context>(
          dev_ctx, {x_dims.size() - 1, static_cast<int>(map_indices.size())});
    }
    out_values =
        Empty<T, Context>(dev_ctx, {static_cast<int>(map_indices.size())});
    auto* out_indices_data = out_indices.data<int64_t>();
    auto* out_values_data = out_values.data<T>();

    auto iter_map_indices = map_indices.begin();
    for (size_t j = 0; j < map_indices.size(); ++j) {
      std::vector<int> pos = iter_map_indices->first;
      std::vector<int64_t> values_index = iter_map_indices->second;
      iter_map_indices++;
      T out_value = 0;
      for (auto i = 0; i < x_dims[0]; ++i) {
        for (auto index : values_index) {
          out_value += x_values_data[index];
        }
        if (keep_dim) {
          if (i == axis[0]) {
            out_indices_data[j + i * map_indices.size()] = 0;
          } else {
            out_indices_data[j + i * map_indices.size()] = pos[i];
          }
        } else {
          if (i < axis[0]) {
            out_indices_data[j + i * map_indices.size()] = pos[i];
          } else if (i > axis[0]) {
            out_indices_data[j + (i - 1) * map_indices.size()] = pos[i];
          }
        }
      }
      out_values_data[j] = out_value;
    }
  }

  out->SetMember(out_indices, out_values, out_dims);
}

template <typename T, typename Context>
void SumCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCsrTensor* out) {
  unsigned int n_dim = axis.size();
  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_values = x.values();
  const auto* x_crows_data = x_crows.data<int64_t>();
  const T* x_values_data = x_values.data<T>();

  DenseTensor out_crows, out_cols, out_values;
  DDim out_dims;
  if (n_dim == 0) {
    out_dims = make_ddim({1, 1});
    out_crows = Empty<int64_t, Context>(dev_ctx, {2});  // crows = [0, 1]
    auto* out_crows_data = out_crows.data<int64_t>();
    out_crows_data[0] = 0;
    out_crows_data[0] = 1;

    out_cols = Empty<int64_t, Context>(dev_ctx, {1});  // crows = [0]
    auto* out_cols_data = out_cols.data<int64_t>();
    out_cols_data[0] = 0;

    out_values = phi::Sum<T>(dev_ctx, x.values(), {}, dtype, true);
  } else {
    PADDLE_ENFORCE_EQ(axis[0],
                      -1,
                      phi::errors::Unimplemented(
                          "`axis` of SumCsrKernel only support None or -1 now."
                          "More number will be supported in the future."));
    out_dims = make_ddim({x.dims()[0], 1});
    out_crows = EmptyLike<int64_t, Context>(dev_ctx, x.crows());
    auto* out_crows_data = out_crows.data<int64_t>();
    out_crows_data[0] = 0;

    std::vector<T> out_data;
    for (int i = 0; i < x.dims()[0]; ++i) {
      if (x_crows_data[i] != x_crows_data[i + 1]) {
        int sum_value = 0;
        for (auto j = x_crows_data[i]; j < x_crows_data[i + 1]; ++j) {
          sum_value += x_values_data[j];
        }
        out_crows_data[i + 1] = out_crows_data[i] + 1;
        out_data.push_back(sum_value);
      } else {
        out_crows_data[i + 1] = out_crows_data[i];
      }
    }

    out_cols =
        Empty<int64_t, Context>(dev_ctx, {static_cast<int>(out_data.size())});
    out_values =
        Empty<T, Context>(dev_ctx, {static_cast<int>(out_data.size())});
    auto* out_cols_data = out_cols.data<int64_t>();
    T* out_values_data = out_values.data<T>();
    for (size_t i = 0; i < out_data.size(); ++i) {
      out_cols_data[i] = 0;
      out_values_data[i] = out_data[i];
    }
  }
  out->SetMember(out_crows, out_cols, out_values, out_dims);
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(transpose_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCooKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(transpose_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCsrKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
