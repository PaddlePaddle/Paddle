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

#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi::sparse {

template <typename T, typename IntT, typename Context>
void SumCooCPUKernel(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     const IntArray& axis,
                     DataType dtype,
                     bool keep_dim,
                     SparseCooTensor* out) {
  size_t n_dim = axis.size();
  auto sparse_dim = x.sparse_dim();
  // create out sparse tensor
  const auto& x_dims = x.dims();
  const auto& x_indices = x.indices();
  const auto& x_values = x.values();
  DDim out_dims;
  DenseTensor out_indices;
  DenseTensor out_values;
  if (n_dim == 0) {
    std::vector<int64_t> out_indices_shape;
    if (keep_dim) {
      out_dims = common::make_ddim(std::vector<int64_t>(x_dims.size(), 1));
      out_indices_shape = {sparse_dim, 1};
    } else {
      out_dims = common::make_ddim({1});
      out_indices_shape = {1};
    }
    out_indices = Empty<IntT, Context>(dev_ctx, out_indices_shape);
    auto* out_indices_data = out_indices.data<IntT>();
    std::fill(out_indices_data, out_indices_data + out_indices.numel(), 0);
    out_values = phi::Sum<T>(dev_ctx, x.values(), {}, dtype, keep_dim);
    out->SetMember(out_indices, out_values, out_dims, x.coalesced());
    return;
  }

  auto dim = axis[0] < 0 ? x_dims.size() + axis[0] : axis[0];
  const auto* x_indices_data = x_indices.data<IntT>();
  const auto* x_values_data = x_values.data<T>();

  std::vector<int64_t> dims;
  for (int i = 0; i < x.dims().size(); ++i) {
    if (i != dim) {
      dims.emplace_back(x.dims()[i]);
    } else if (keep_dim || (dim < sparse_dim && sparse_dim == 1)) {
      dims.emplace_back(1);
    }
  }
  out_dims = common::make_ddim(dims);

  if (dim >= sparse_dim) {
    out_indices = x_indices;
    dim = dim - sparse_dim + 1;
    out_values = phi::Sum<T>(dev_ctx, x.values(), {dim}, dtype, keep_dim);
    out->SetMember(out_indices, out_values, out_dims, x.coalesced());
    return;
  }

  // Ensure the sparse_dim is not less than 1.
  if (sparse_dim == 1) {
    keep_dim = true;
  }
  // if axis in sparse_dim and keep_dim, sparse_dim will be reduced.
  if (!keep_dim) {
    sparse_dim -= 1;
  }

  // indices_map is a mapping from output's position to values to be summed.
  std::map<std::vector<IntT>, std::vector<int64_t>> indices_map;
  for (int64_t j = 0; j < x_indices.dims()[1]; ++j) {
    std::vector<IntT> pos;
    for (int64_t i = 0; i < x_indices.dims()[0]; ++i) {
      if (dim != i) {
        pos.emplace_back(x_indices_data[j + i * x_indices.dims()[1]]);
      } else if (keep_dim) {
        pos.emplace_back(0);
      }
    }
    indices_map[pos].emplace_back(j);
  }

  std::vector<int> out_values_dims;
  out_values_dims.push_back(static_cast<int>(indices_map.size()));
  for (auto i = 1; i < x.values().dims().size(); ++i) {
    out_values_dims.push_back(static_cast<int>(x.values().dims()[i]));
  }
  int64_t dense_dim = std::accumulate(out_values_dims.begin() + 1,
                                      out_values_dims.end(),
                                      1,
                                      std::multiplies<int64_t>());

  out_indices = Empty<IntT, Context>(
      dev_ctx, {sparse_dim, static_cast<int>(indices_map.size())});
  out_values = Empty<T, Context>(dev_ctx, out_values_dims);

  auto* out_indices_data = out_indices.data<IntT>();
  auto* out_values_data = out_values.data<T>();

  auto iter_indices_map = indices_map.begin();
  for (size_t j = 0; j < indices_map.size(); ++j) {
    std::vector<IntT> pos = iter_indices_map->first;
    std::vector<int64_t> values_index = iter_indices_map->second;
    iter_indices_map++;
    for (auto i = 0; i < sparse_dim; ++i) {
      out_indices_data[j + i * indices_map.size()] = pos[i];
    }
    for (auto i = 0; i < dense_dim; ++i) {
      T out_value = 0;
      for (auto index : values_index) {
        out_value += x_values_data[i + index * dense_dim];
      }
      out_values_data[i + j * dense_dim] = out_value;
    }
  }

  if (dtype != phi::DataType::UNDEFINED && dtype != x.dtype()) {
    out_values = phi::Cast<T, Context>(dev_ctx, out_values, dtype);
  }
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());
}

template <typename T, typename Context>
void SumCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCsrTensor* out) {
  size_t n_dim = axis.size();
  const auto& x_crows = x.crows();
  const auto& x_values = x.values();
  const auto* x_crows_data = x_crows.data<int64_t>();
  const auto* x_values_data = x_values.data<T>();

  DenseTensor out_crows, out_cols, out_values;
  DDim out_dims;
  if (n_dim == 0) {
    if (keep_dim && x.dims().size() == 3) {
      out_dims = common::make_ddim({1, 1, 1});
    } else {
      out_dims = common::make_ddim({1, 1});
    }
    out_crows = Empty<int64_t, Context>(dev_ctx, {2});  // crows = [0, 1]
    auto* out_crows_data = out_crows.data<int64_t>();
    out_crows_data[0] = 0;
    out_crows_data[1] = 1;

    out_cols = Empty<int64_t, Context>(dev_ctx, {1});  // crows = [0]
    auto* out_cols_data = out_cols.data<int64_t>();
    out_cols_data[0] = 0;
    out_values = phi::Sum<T>(dev_ctx, x.values(), {}, dtype, true);
  } else {
    PADDLE_ENFORCE_EQ(axis[0],
                      -1,
                      common::errors::Unimplemented(
                          "`axis` of SumCsrKernel only support None or -1 now."
                          "More number will be supported in the future."));
    out_crows = EmptyLike<int64_t, Context>(dev_ctx, x.crows());
    auto* out_crows_data = out_crows.data<int64_t>();
    std::vector<T> out_data;
    if (x.dims().size() == 2) {
      out_crows_data[0] = 0;
      out_dims = common::make_ddim({x.dims()[0], 1});
      for (int i = 0; i < x.dims()[0]; ++i) {
        if (x_crows_data[i] != x_crows_data[i + 1]) {
          T sum_value = 0;
          for (auto j = x_crows_data[i]; j < x_crows_data[i + 1]; ++j) {
            sum_value += x_values_data[j];
          }
          out_crows_data[i + 1] = out_crows_data[i] + 1;
          out_data.emplace_back(sum_value);
        } else {
          out_crows_data[i + 1] = out_crows_data[i];
        }
      }
    } else {
      if (keep_dim) {
        out_dims = common::make_ddim({x.dims()[0], x.dims()[1], 1});
      } else {
        out_dims = common::make_ddim({x.dims()[0], x.dims()[1]});
      }
      int j = 0;
      for (int batch = 0; batch < x.dims()[0]; ++batch) {
        auto* cur_x_crows_data = x_crows_data + batch * x.dims()[2];
        auto* cur_out_crows_data = out_crows_data + batch * x.dims()[2];
        for (int i = 0; i < x.dims()[1]; ++i) {
          cur_out_crows_data[0] = 0;
          if (cur_x_crows_data[i] != cur_x_crows_data[i + 1]) {
            T sum_value = 0;
            for (auto k = cur_x_crows_data[i]; k < cur_x_crows_data[i + 1];
                 ++k) {
              sum_value += x_values_data[j++];
            }
            out_data.emplace_back(sum_value);
            cur_out_crows_data[i + 1] = cur_out_crows_data[i] + 1;
          } else {
            cur_out_crows_data[i + 1] = cur_out_crows_data[i];
          }
        }
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
    if (dtype != phi::DataType::UNDEFINED && dtype != x.dtype()) {
      out_values = phi::Cast<T, Context>(dev_ctx, out_values, dtype);
    }
  }
  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename Context>
void SumCooKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "SumCooCPUKernel", ([&] {
                                 SumCooCPUKernel<T, data_t, Context>(
                                     dev_ctx, x, axis, dtype, keep_dim, out);
                               }));
}

}  // namespace phi::sparse

PD_REGISTER_KERNEL(sum_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCooKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(sum_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCsrKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
}
