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

#include "paddle/common/ddim.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_grad_kernel_impl.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"

namespace phi::sparse {

template <typename T, typename IntT, typename Context>
void ReshapeCooCPUKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const phi::IntArray& shape,
                         SparseCooTensor* out) {
  // TODO(OccupyMars2025): Currently, reshape is only applicable to sparse dims
  int64_t x_nnz = x.nnz();

  // Use DDim::reshape to handle -1 and 0 in the argument "shape"
  std::vector<int> new_shape(shape.GetData().begin(), shape.GetData().end());
  phi::DDim out_dims = x.dims().reshape(new_shape);
  // get sparse part dimensions of x and out
  std::vector<int64_t> x_sparse_part_dims;
  std::vector<int64_t> out_sparse_part_dims;
  for (int i = 0; i < x.sparse_dim(); ++i) {
    x_sparse_part_dims.push_back(x.dims()[i]);
  }
  for (int i = 0; i < out_dims.size() - x.dense_dim(); ++i) {
    out_sparse_part_dims.push_back(out_dims[i]);
  }
  DenseTensor out_indices = Empty<IntT, Context>(
      dev_ctx, {static_cast<int64_t>(out_sparse_part_dims.size()), x_nnz});
  DenseTensor out_values(x.values());
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());

  // compute values of indices
  const DenseTensor& x_indices = x.indices();
  const auto* x_indices_data = x_indices.data<IntT>();
  auto* out_indices_data = out_indices.data<IntT>();

  const phi::DDim& x_sparse_part_strides =
      common::stride(common::make_ddim(x_sparse_part_dims));
  const phi::DDim& out_sparse_part_strides =
      common::stride(common::make_ddim(out_sparse_part_dims));
  int64_t location = 0;
  for (int64_t j = 0; j < x_nnz; ++j) {
    location = 0;
    for (int i = 0; i < x.sparse_dim(); ++i) {
      location += x_indices_data[i * x_nnz + j] * x_sparse_part_strides[i];
    }
    for (int i = 0; i < static_cast<int>(out_sparse_part_dims.size()); ++i) {
      out_indices_data[i * x_nnz + j] = location / out_sparse_part_strides[i];
      location %= out_sparse_part_strides[i];
    }
  }
}

template <typename T, typename Context>
void ReshapeCooKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const phi::IntArray& shape,
                      SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "ReshapeCooCPUKernel", ([&] {
        ReshapeCooCPUKernel<T, data_t, Context>(dev_ctx, x, shape, out);
      }));
}

template <typename T, typename Context>
void ReshapeCsrKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const phi::IntArray& shape,
                      SparseCsrTensor* out) {
  // transform csr format to coo format, and then use coo kernel
  const SparseCooTensor x_coo = CsrToCoo<T, Context>(dev_ctx, x);
  SparseCooTensor out_coo;
  ReshapeCooKernel<T, Context>(dev_ctx, x_coo, shape, &out_coo);
  CooToCsrKernel<T, Context>(dev_ctx, out_coo, out);
}

}  // namespace phi::sparse

PD_REGISTER_KERNEL(reshape_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ReshapeCooKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(reshape_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ReshapeCsrKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
