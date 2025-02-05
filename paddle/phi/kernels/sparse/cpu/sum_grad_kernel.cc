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

#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_grad_kernel_impl.h"

namespace phi::sparse {

template <typename T, typename IntT, typename Context>
void SumCooGradCPUKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const SparseCooTensor& dout,
                         const IntArray& axis,
                         bool keep_dim,
                         SparseCooTensor* dx) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, dx);
  unsigned int n_dim = axis.size();

  const DenseTensor& x_indices = x.indices();
  const DenseTensor& dout_indices = dout.indices();
  const DenseTensor& dout_values = dout.values();
  const auto* dout_indices_data = dout_indices.data<int64_t>();
  const auto* dout_values_data = dout_values.data<T>();

  DenseTensor* dx_indices = dx->mutable_indices();
  DenseTensor* dx_values = dx->mutable_values();
  *dx_indices = x_indices;

  const auto* dx_indices_data = dx_indices->data<int64_t>();
  auto* dx_values_data = dx_values->data<T>();

  phi::funcs::SetConstant<Context, T> set_constant;
  if (n_dim == 0) {
    T value = dout_values.data<T>()[0];
    set_constant(dev_ctx, dx_values, value);
    if (dx_values->dtype() != dx->dtype()) {
      *dx_values = phi::Cast<T, Context>(dev_ctx, *dx_values, dx->dtype());
    }
    return;
  }

  auto dim = axis[0] < 0 ? x.dims().size() + axis[0] : axis[0];
  auto sparse_dim = x.sparse_dim();
  if (dim >= sparse_dim) {
    dim = dim - sparse_dim + 1;
    phi::ReduceSumGradKernel<T, Context>(
        dev_ctx, x.values(), dout.values(), {dim}, keep_dim, false, dx_values);
    if (dx_values->dtype() != dx->dtype()) {
      *dx_values = phi::Cast<T, Context>(dev_ctx, *dx_values, dx->dtype());
    }
    return;
  }
  // Ensure the sparse_dim is not less than 1.
  if (sparse_dim == 1) {
    keep_dim = true;
  }

  int64_t dense_dim = 1;
  for (auto i = 1; i < x.values().dims().size(); ++i) {
    dense_dim *= x.values().dims()[i];
  }

  std::map<std::vector<IntT>, int64_t> indices_map;
  for (auto j = 0; j < dout_indices.dims()[1]; ++j) {
    std::vector<IntT> pos;
    pos.reserve(dout_indices.dims()[0]);
    for (int i = 0; i < dout_indices.dims()[0]; ++i) {
      pos.push_back(dout_indices_data[j + i * dout_indices.dims()[1]]);
    }
    indices_map[pos] = j;
  }

  for (auto j = 0; j < dx_indices->dims()[1]; ++j) {
    std::vector<IntT> pos;
    for (int i = 0; i < dx_indices->dims()[0]; ++i) {
      if (i != dim) {
        pos.push_back(dx_indices_data[j + i * dx_indices->dims()[1]]);
      } else if (keep_dim) {
        pos.push_back(0);
      }
    }
    for (int i = 0; i < dense_dim; ++i) {
      dx_values_data[i + j * dense_dim] =
          dout_values_data[i + indices_map[pos] * dense_dim];
    }
  }
  if (dx_values->dtype() != dx->dtype()) {
    *dx_values = phi::Cast<T, Context>(dev_ctx, *dx_values, dx->dtype());
  }
}

template <typename T, typename Context>
void SumCsrGradKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const SparseCsrTensor& dout,
                      const IntArray& axis,
                      bool keep_dim UNUSED,
                      SparseCsrTensor* dx) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, dx);
  unsigned int n_dim = axis.size();

  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_cols = x.cols();
  const DenseTensor& dout_values = dout.values();
  const auto* x_crows_data = x_crows.data<int64_t>();

  DenseTensor* dx_crows = dx->mutable_crows();
  DenseTensor* dx_cols = dx->mutable_cols();
  DenseTensor* dx_values = dx->mutable_values();

  *dx_crows = x_crows;
  *dx_cols = x_cols;

  phi::funcs::SetConstant<Context, T> set_constant;
  if (n_dim == 0) {
    T value = dout_values.data<T>()[0];
    set_constant(dev_ctx, dx_values, value);
    if (dx_values->dtype() != dx->dtype()) {
      *dx_values = phi::Cast<T, Context>(dev_ctx, *dx_values, dx->dtype());
    }
    return;
  }
  PADDLE_ENFORCE_EQ(axis[0],
                    -1,
                    common::errors::Unimplemented(
                        "`axis` of SumCsrKernel only support None or -1 now."
                        "More number will be supported in the future."));

  if (x.dims().size() == 2) {
    int value_index = 0;
    for (int k = 0; k < x.dims()[0]; ++k) {
      if (x_crows_data[k] == x_crows_data[k + 1]) {
        continue;
      }
      T value = dout_values.data<T>()[value_index];
      set_constant(dev_ctx, dx_values, value);
      value_index += 1;
    }
  } else {
    int dout_value_index = 0;
    int dx_value_index = 0;
    for (auto batch = 0; batch < x.dims()[0]; ++batch) {
      for (auto k = batch * (x.dims()[1] + 1);
           k < batch * (x.dims()[1] + 1) + x.dims()[1];
           ++k) {
        if (x_crows_data[k] == x_crows_data[k + 1]) {
          continue;
        }
        T value = dout_values.data<T>()[dout_value_index];
        for (auto i = x_crows_data[k]; i < x_crows_data[k + 1]; ++i) {
          dx_values->data<T>()[dx_value_index] = value;
          dx_value_index++;
        }
        dout_value_index++;
      }
    }
  }

  if (dx_values->dtype() != dx->dtype()) {
    *dx_values = phi::Cast<T, Context>(dev_ctx, *dx_values, dx->dtype());
  }
}

template <typename T, typename Context>
void SumCooGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const SparseCooTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCooTensor* dx) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "SumCooGradCPUKernel", ([&] {
        SumCooGradCPUKernel<T, data_t, Context>(
            dev_ctx, x, dout, axis, keep_dim, dx);
      }));
}

}  // namespace phi::sparse

PD_REGISTER_KERNEL(sum_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(sum_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
