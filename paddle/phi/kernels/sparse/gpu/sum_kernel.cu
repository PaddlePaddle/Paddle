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
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/cum_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/index_select_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
__global__ void SumCooCudaKernel(const IntT* x_indices_data,
                                 const T* x_values_data,
                                 const int64_t x_nnz,
                                 const int64_t dense_dim,
                                 const int64_t sparse_dim,
                                 const int64_t axis,
                                 const bool keep_dim,
                                 IntT* out_indices_data,
                                 T* out_values_data) {
  CUDA_KERNEL_LOOP_TYPE(index_i, x_nnz, int64_t) {
    int64_t i = 0;
    for (int j = 0; j < dense_dim; ++j) {
      out_values_data[j + index_i * dense_dim] = 0;
    }

    int64_t _index_j_ =
        static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    for (auto index_j = _index_j_; index_j < x_nnz;
         index_j += static_cast<int64_t>(blockDim.y) * gridDim.y) {
      // Determine whether the index_i and index_j elements have the same
      // indices in all dimensions except for the specified axis dimension.
      bool same = true;
      for (int j = 0; j < sparse_dim + !keep_dim; ++j) {
        if (j != axis && x_indices_data[index_i + j * x_nnz] !=
                             x_indices_data[index_j + j * x_nnz]) {
          same = false;
          break;
        }
      }
      if (same) {
        for (int j = 0; j < dense_dim; ++j) {
          phi::CudaAtomicAdd(&out_values_data[j + index_i * dense_dim],
                             x_values_data[j + index_j * dense_dim]);
        }
      }
    }
    if (_index_j_ != 0) {
      return;
    }
    if (keep_dim) {
      for (int j = 0; j < sparse_dim; ++j) {
        if (j == axis) {
          out_indices_data[index_i + j * x_nnz] = 0;
        } else {
          out_indices_data[index_i + j * x_nnz] =
              x_indices_data[index_i + j * x_nnz];
        }
      }
      return;
    }
    for (int j = 0; j < sparse_dim; ++j) {
      // out_indices_data [sparse_dim, x.nnz()]
      int64_t x_indices_data_offset;
      if (j < axis) {
        x_indices_data_offset = index_i + j * x_nnz;
      } else {
        x_indices_data_offset = index_i + (j + 1) * x_nnz;
      }
      out_indices_data[index_i + j * x_nnz] =
          x_indices_data[x_indices_data_offset];
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
                                   const int64_t* batch_nnz_data,
                                   int64_t* out_crows_data,
                                   int64_t* out_cols_data,
                                   T* out_values_data) {
  {
    CUDA_KERNEL_LOOP_TYPE(index, x_dim0 * x_dim1, int64_t) {
      out_cols_data[index] = 0;
    }
  }

  CUDA_KERNEL_LOOP_TYPE(index, x_dim0 * (x_dim1 + 1), int64_t) {
    int64_t batch = index / (x_dim1 + 1);
    int64_t number = index % (x_dim1 + 1);
    out_crows_data[index] = number;

    if (number != x_dim1) {
      T sum_value = 0;
      int64_t x_values_data_offset;
      if (batch == 0) {
        x_values_data_offset = 0;
      } else {
        x_values_data_offset = batch_nnz_data[batch - 1];
      }
      for (int64_t j = x_crows_data[index]; j < x_crows_data[index + 1]; ++j) {
        sum_value += x_values_data[j + x_values_data_offset];
      }

      // `index - batch` would never exceed x_dim0 * x_dim1.
      out_values_data[index - batch] = sum_value;
    }
  }
}

template <typename T, typename IntT, typename Context>
void SumCooGPU0Kernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const IntArray& axis,
                      DataType dtype,
                      bool keep_dim,
                      SparseCooTensor* out) {
  auto sparse_dim = x.sparse_dim();
  // create out sparse tensor
  const auto& x_dims = x.dims();
  const auto& x_indices = x.indices();
  const auto& x_values = x.values();
  DDim out_dims;
  DenseTensor out_indices;
  DenseTensor out_values;
  if (keep_dim) {
    out_dims = common::make_ddim(std::vector<int64_t>(x_dims.size(), 1));
    out_indices = Empty<IntT, Context>(dev_ctx, {sparse_dim, 1});
  } else {
    out_dims = common::make_ddim({1});
    out_indices = Empty<IntT, Context>(dev_ctx, {1, 1});
  }
  phi::funcs::SetConstant<Context, IntT> set_out_indices;
  set_out_indices(dev_ctx, &out_indices, static_cast<IntT>(0));
  out_values = phi::Sum<T>(dev_ctx, x.values(), {}, dtype, keep_dim);
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());
}

template <typename T, typename IntT, typename Context>
void SumCooGPU1Kernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const IntArray& axis,
                      DataType dtype,
                      bool keep_dim,
                      SparseCooTensor* out) {
  auto sparse_dim = x.sparse_dim();
  // create out sparse tensor
  const auto& x_dims = x.dims();
  const auto& x_indices = x.indices();
  const auto& x_values = x.values();
  DDim out_dims;
  DenseTensor out_indices;
  DenseTensor out_values;
  auto n_dim = x.dims().size();
  auto dim = axis[0] < 0 ? x_dims.size() + axis[0] : axis[0];

  std::vector<int64_t> dims;
  for (int i = 0; i < n_dim; ++i) {
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

  std::vector<int> out_values_dims;
  out_values_dims.push_back(x.nnz());
  for (auto i = 1; i < x.values().dims().size(); ++i) {
    out_values_dims.push_back(static_cast<int>(x.values().dims()[i]));
  }
  int64_t dense_dim = std::accumulate(out_values_dims.begin() + 1,
                                      out_values_dims.end(),
                                      1,
                                      std::multiplies<int64_t>());

  out_indices = Empty<IntT, Context>(dev_ctx, {sparse_dim, x.nnz()});
  out_values = Empty<T, Context>(dev_ctx, out_values_dims);

  const auto* x_indices_data = x_indices.data<IntT>();
  const auto* x_values_data = x_values.data<T>();
  auto* out_indices_data = out_indices.data<IntT>();
  auto* out_values_data = out_values.data<T>();

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig2D(dev_ctx, x.nnz(), x.nnz());
  SumCooCudaKernel<T, IntT><<<config.block_per_grid.x,
                              config.thread_per_block.x,
                              0,
                              dev_ctx.stream()>>>(x_indices_data,
                                                  x_values_data,
                                                  x.nnz(),
                                                  dense_dim,
                                                  sparse_dim,
                                                  dim,
                                                  keep_dim,
                                                  out_indices_data,
                                                  out_values_data);
  if (dtype != phi::DataType::UNDEFINED && dtype != x.dtype()) {
    out_values = phi::Cast<T, Context>(dev_ctx, out_values, dtype);
  }
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());
}

template <typename T, typename Context>
void SumCooKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCooTensor* out) {
  const size_t n_dim = axis.size();
  if (n_dim == 0) {
    PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "SumCooGPUKernel", ([&] {
                                   SumCooGPU0Kernel<T, data_t, Context>(
                                       dev_ctx, x, axis, dtype, keep_dim, out);
                                 }));
  } else {
    PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "SumCooGPUKernel", ([&] {
                                   SumCooGPU1Kernel<T, data_t, Context>(
                                       dev_ctx, x, axis, dtype, keep_dim, out);
                                 }));
  }
}

template <typename T, typename Context>
void SumCsr0Kernel(const Context& dev_ctx,
                   const SparseCsrTensor& x,
                   const IntArray& axis,
                   DataType dtype,
                   bool keep_dim,
                   SparseCsrTensor* out) {
  auto x_dim0 = x.dims()[0];
  auto x_dim1 = x.dims()[1];
  const auto& x_crows = x.crows();
  const auto& x_values = x.values();
  const auto* x_crows_data = x_crows.data<int64_t>();
  const auto* x_values_data = x_values.data<T>();

  DenseTensor out_crows, out_cols, out_values;
  DDim out_dims;
  if (keep_dim && x.dims().size() == 3) {
    out_dims = common::make_ddim({1, 1, 1});
  } else {
    out_dims = common::make_ddim({1, 1});
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
  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename Context>
void SumCsr1Kernel(const Context& dev_ctx,
                   const SparseCsrTensor& x,
                   const IntArray& axis,
                   DataType dtype,
                   bool keep_dim,
                   SparseCsrTensor* out) {
  auto x_dim0 = x.dims()[0];
  auto x_dim1 = x.dims()[1];
  const auto& x_crows = x.crows();
  const auto& x_values = x.values();
  const auto* x_crows_data = x_crows.data<int64_t>();
  const auto* x_values_data = x_values.data<T>();

  DenseTensor out_crows, out_cols, out_values;
  DDim out_dims;
  out_crows = EmptyLike<int64_t, Context>(dev_ctx, x.crows());
  auto* out_crows_data = out_crows.data<int64_t>();

  if (x.dims().size() == 2) {
    out_cols = Empty<int64_t, Context>(dev_ctx, {x_dim0});
    out_values = Empty<T, Context>(dev_ctx, {x_dim0});
    auto* out_cols_data = out_cols.data<int64_t>();
    auto* out_values_data = out_values.data<T>();
    out_dims = common::make_ddim({x_dim0, 1});
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_dim0 + 1, 1);
    SumCsr2DCudaKernel<T><<<config.block_per_grid.x,
                            config.thread_per_block.x,
                            0,
                            dev_ctx.stream()>>>(x_crows_data,
                                                x_values_data,
                                                x_dim0,
                                                out_crows_data,
                                                out_cols_data,
                                                out_values_data);

  } else {
    out_cols = Empty<int64_t, Context>(dev_ctx, {x_dim0 * x_dim1});
    out_values = Empty<T, Context>(dev_ctx, {x_dim0 * x_dim1});
    auto* out_cols_data = out_cols.data<int64_t>();
    auto* out_values_data = out_values.data<T>();
    if (keep_dim) {
      out_dims = common::make_ddim({x_dim0, x_dim1, 1});
    } else {
      out_dims = common::make_ddim({x_dim0, x_dim1});
    }

    DenseTensor x_crows_reshape =
        Reshape<int64_t, Context>(dev_ctx, x_crows, {x_dim0, x_dim1 + 1});
    DenseTensor last_indices = Empty<int64_t, Context>(dev_ctx, {1});
    phi::funcs::SetConstant<Context, int64_t> set_constant;
    set_constant(dev_ctx, &last_indices, static_cast<int64_t>(x_dim1));

    DenseTensor x_crows_last = Empty<int64_t, Context>(dev_ctx, {x_dim0, 1});
    IndexSelectKernel<int64_t, Context>(
        dev_ctx, x_crows_reshape, last_indices, 1, &x_crows_last);

    DenseTensor batch_nnz = Empty<int64_t, Context>(dev_ctx, {x_dim0, 1});
    CumsumKernel<int64_t, Context>(
        dev_ctx, x_crows_last, Scalar(0), false, false, false, &batch_nnz);
    auto* batch_nnz_data = batch_nnz.data<int64_t>();

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, x.dims()[0] * (x.dims()[1] + 1), 1);
    SumCsr3DCudaKernel<T><<<config.block_per_grid.x,
                            config.thread_per_block.x,
                            0,
                            dev_ctx.stream()>>>(x_crows_data,
                                                x_values_data,
                                                x_dim0,
                                                x_dim1,
                                                batch_nnz_data,
                                                out_crows_data,
                                                out_cols_data,
                                                out_values_data);
  }
  if (dtype != phi::DataType::UNDEFINED && dtype != x.dtype()) {
    out_values = phi::Cast<T, Context>(dev_ctx, out_values, dtype);
  }
  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename Context>
void SumCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCsrTensor* out) {
  size_t n_dim = axis.size();
  if (n_dim == 0) {
    SumCsr0Kernel<T, Context>(dev_ctx, x, axis, dtype, keep_dim, out);
  } else {
    PADDLE_ENFORCE_EQ(axis[0],
                      -1,
                      common::errors::Unimplemented(
                          "`axis` of SumCsrKernel only support None or -1 now."
                          "More number will be supported in the future."));
    SumCsr1Kernel<T, Context>(dev_ctx, x, axis, dtype, keep_dim, out);
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sum_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCooKernel,
                   float,
                   double,
                   int,
                   int64_t) {
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(sum_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCsrKernel,
                   float,
                   double,
                   int,
                   int64_t) {
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
}
