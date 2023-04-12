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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

template <typename T>
__global__ void SumCooCudaKernel(const int64_t* x_indices_data,
                                 const T* x_values_data,
                                 const int64_t x_nnz,
                                 const int64_t dense_dim,
                                 const int64_t n_dim,
                                 const int64_t axis,
                                 const int64_t keep_dim,
                                 int64_t* out_indices_data,
                                 T* out_values_data) {
  CUDA_KERNEL_LOOP_TYPE(index, x_nnz, int64_t) {
    int64_t i = 0;
    out_values_data[index] = 0;
    while (i < x_nnz) {
      bool same = true;
      for (int j = 0; j < n_dim; ++j) {
        if (j != axis && x_indices_data[index + j * x_nnz] !=
                             x_indices_data[i + j * x_nnz]) {
          same = false;
          break;
        }
      }
      if (same) {
        for (int j = 0; j < dense_dim; ++j) {
          out_values_data[j + index * dense_dim] +=
              x_values_data[j + i * dense_dim];  // j + index * dense_dim
        }
      }
      i++;
    }
    if (keep_dim) {
      for (int j = 0; j < n_dim; ++j) {
        out_indices_data[index + j * x_nnz] = x_indices_data[index + j * x_nnz];
        if (j == axis) {
          out_indices_data[index + j * x_nnz] = 0;
        }
      }
    } else {
      for (int j = 0; j < n_dim; ++j) {
        if (j < axis) {
          out_indices_data[index + j * x_nnz] =
              x_indices_data[index + j * x_nnz];
        } else if (j > axis) {
          out_indices_data[index + (j - 1) * x_nnz] =
              x_indices_data[index + j * x_nnz];
        }
      }
    }
  }
}

__global__ void SumAllCsrCudaKernel(int64_t* out_crows_data,
                                    int64_t* out_cols_data) {
  CUDA_KERNEL_LOOP_TYPE(index, 2, int64_t) {
    out_crows_data[index] = index;
    if (index == 0) {
      out_cols_data[0] = 0;
    }
  }
}

template <typename T>
__global__ void SumCsr2DCudaKernel(const int64_t* x_crows_data,
                                   const T* x_values_data,
                                   const int64_t x_dim0,
                                   int64_t* out_crows_data,
                                   int64_t* out_cols_data,
                                   T* out_values_data) {
  CUDA_KERNEL_LOOP_TYPE(index, x_dim0 + 1, int64_t) {
    out_crows_data[index] = index;
    if (index != x_dim0) {
      out_cols_data[index] = 0;
      T sum_value = 0;
      for (auto j = x_crows_data[index]; j < x_crows_data[index + 1]; ++j) {
        sum_value += x_values_data[j];
      }
      out_values_data[index] = sum_value;
    }
  }
}

template <typename T>
__global__ void SumCsr3DCudaKernel(const int64_t* x_crows_data,
                                   const T* x_values_data,
                                   const int64_t x_dim0,
                                   const int64_t x_dim1,
                                   int64_t* out_crows_data,
                                   int64_t* out_cols_data,
                                   T* out_values_data) {
  CUDA_KERNEL_LOOP_TYPE(index, x_dim0 * (x_dim1 + 1), int64_t) {
    int64_t batch = index / (x_dim1 + 1);
    int64_t number = index % (x_dim1 + 1);
    out_crows_data[index] = number;
    out_cols_data[index] = 0;

    if (number != x_dim1) {
      int64_t batch_nnz = 0;
      for (int64_t b = 1; b <= batch; ++b) {
        batch_nnz += x_crows_data[b * (x_dim1 + 1) - 1];
      }

      T sum_value = 0;
      for (int64_t j = x_crows_data[index]; j < x_crows_data[index + 1]; ++j) {
        sum_value += x_values_data[j + batch_nnz];
      }
      out_values_data[index - batch] = sum_value;
    }
  }
}

template <typename T, typename Context>
void SumCooKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCooTensor* out) {
  size_t axis_dim = axis.size();
  auto dense_dim = x.values().dims()[1];
  // create out sparse tensor
  const auto& x_dims = x.dims();
  const auto& x_indices = x.indices();
  const auto& x_values = x.values();
  DDim out_dims;
  DenseTensor out_indices;
  DenseTensor out_values;
  using x_indices_dtype = int64_t;
  //  using x_indices_dtype =
  //      typename DataTypeToCppType<x.indices().dtype()>::type;
  if (axis_dim == 0) {
    if (keep_dim) {
      out_dims = make_ddim(std::vector<int64_t>(x_dims.size(), 1));
      out_indices =
          Empty<x_indices_dtype, Context>(dev_ctx, {x_dims.size(), 1});
    } else {
      out_dims = make_ddim({1});
      out_indices = Empty<x_indices_dtype, Context>(dev_ctx, {1, 1});
    }
    phi::funcs::SetConstant<Context, x_indices_dtype> set_out_indices;
    set_out_indices(dev_ctx, &out_indices, static_cast<x_indices_dtype>(0));
    out_values = phi::Sum<T>(dev_ctx, x.values(), {}, dtype, true);
  } else {
    auto dim = axis[0] < 0 ? x_dims.size() + axis[0] : axis[0];
    auto n_dim = x.dims().size();
    std::vector<int64_t> dims;
    for (int i = 0; i < n_dim; ++i) {
      if (i == dim) {
        if (keep_dim) {
          dims.emplace_back(1);
        }
      } else {
        dims.emplace_back(x.dims()[i]);
      }
    }
    out_dims = make_ddim(dims);
    auto sparse_dim = x_indices.dims().size();
    if (keep_dim) {
      sparse_dim -= 1;
    }

    out_indices =
        Empty<x_indices_dtype, Context>(dev_ctx, {sparse_dim, x.nnz()});
    out_values = Empty<T, Context>(dev_ctx, {x.nnz(), dense_dim});

    const auto* x_indices_data = x_indices.data<x_indices_dtype>();
    const auto* x_values_data = x_values.data<T>();
    auto* out_indices_data = out_indices.data<x_indices_dtype>();
    auto* out_values_data = out_values.data<T>();

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x.nnz(), 1);
    SumCooCudaKernel<T><<<config.block_per_grid.x,
                          config.thread_per_block.x,
                          0,
                          dev_ctx.stream()>>>(x_indices_data,
                                              x_values_data,
                                              x.nnz(),
                                              dense_dim,
                                              n_dim,
                                              dim,
                                              keep_dim,
                                              out_indices_data,
                                              out_values_data);
    if (dtype != phi::DataType::UNDEFINED && dtype != x.dtype()) {
      out_values = phi::Cast<T, Context>(dev_ctx, out_values, dtype);
    }
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
      out_dims = make_ddim({1, 1, 1});
    } else {
      out_dims = make_ddim({1, 1});
    }
    out_crows = Empty<int64_t, Context>(dev_ctx, {2});  // crows = [0, 1]
    out_cols = Empty<int64_t, Context>(dev_ctx, {1});   // crows = [0]
    auto* out_crows_data = out_crows.data<int64_t>();
    auto* out_cols_data = out_cols.data<int64_t>();

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, 2, 1);
    SumAllCsrCudaKernel<<<config.block_per_grid.x,
                          config.thread_per_block.x,
                          0,
                          dev_ctx.stream()>>>(out_crows_data, out_cols_data);

    out_values = phi::Sum<T>(dev_ctx, x.values(), {}, dtype, true);
  } else {
    PADDLE_ENFORCE_EQ(axis[0],
                      -1,
                      phi::errors::Unimplemented(
                          "`axis` of SumCsrKernel only support None or -1 now."
                          "More number will be supported in the future."));
    out_crows = EmptyLike<int64_t, Context>(dev_ctx, x.crows());
    auto* out_crows_data = out_crows.data<int64_t>();

    if (x.dims().size() == 2) {
      out_cols = Empty<int64_t, Context>(dev_ctx, {x.dims()[0]});
      out_values = Empty<T, Context>(dev_ctx, {x.dims()[0]});
      auto* out_cols_data = out_cols.data<int64_t>();
      auto* out_values_data = out_values.data<T>();
      out_dims = make_ddim({x.dims()[0], 1});
      auto config =
          phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x.dims()[0] + 1, 1);
      SumCsr2DCudaKernel<T><<<config.block_per_grid.x,
                              config.thread_per_block.x,
                              0,
                              dev_ctx.stream()>>>(x_crows_data,
                                                  x_values_data,
                                                  x.dims()[0],
                                                  out_crows_data,
                                                  out_cols_data,
                                                  out_values_data);

    } else {
      out_cols = Empty<int64_t, Context>(dev_ctx, {x.dims()[0] * x.dims()[1]});
      out_values = Empty<T, Context>(dev_ctx, {x.dims()[0] * x.dims()[1]});
      auto* out_cols_data = out_cols.data<int64_t>();
      auto* out_values_data = out_values.data<T>();
      if (keep_dim) {
        out_dims = make_ddim({x.dims()[0], x.dims()[1], 1});
      } else {
        out_dims = make_ddim({x.dims()[0], x.dims()[1]});
      }
      auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
          dev_ctx, x.dims()[0] * (x.dims()[1] + 1), 1);
      SumCsr3DCudaKernel<T><<<config.block_per_grid.x,
                              config.thread_per_block.x,
                              0,
                              dev_ctx.stream()>>>(x_crows_data,
                                                  x_values_data,
                                                  x.dims()[0],
                                                  x.dims()[1],
                                                  out_crows_data,
                                                  out_cols_data,
                                                  out_values_data);
    }
    if (dtype != phi::DataType::UNDEFINED && dtype != x.dtype()) {
      out_values = phi::Cast<T, Context>(dev_ctx, out_values, dtype);
    }
  }
  out->SetMember(out_crows, out_cols, out_values, out_dims);
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sum_coo,
                   GPU,
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
                   GPU,
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
