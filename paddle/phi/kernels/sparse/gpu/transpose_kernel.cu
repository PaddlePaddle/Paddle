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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

__global__ void TransposeCooCudaKernel(const int64_t *x_indices_data,
                                       const int *perm,
                                       const std::size_t n_dim,
                                       const int64_t x_nnz,
                                       int64_t *out_indices_data) {
  CUDA_KERNEL_LOOP_TYPE(index, x_nnz * n_dim, int64_t) {
    int64_t i = index / x_nnz;
    int64_t j = index % x_nnz;
    out_indices_data[index] = x_indices_data[j + perm[i] * x_nnz];
  }
}

template <typename T, typename IntT>
__global__ void TransposeCsr2DCudaKernel(const IntT *x_crows_data,
                                         const IntT *x_cols_data,
                                         const T *x_values_data,
                                         const int *perm,
                                         const int64_t *x_dims,
                                         const int64_t *out_dims,
                                         const int64_t x_nnz,
                                         IntT *out_crows_data,
                                         IntT *out_cols_data,
                                         T *out_values_data) {
  int64_t __index__ =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // compute out_crows_data by x_cols_data
  for (int64_t i = __index__; i <= out_dims[0]; i += blockDim.x * gridDim.x) {
    out_crows_data[i] = 0;
  }
  __syncthreads();
  if (__index__ == 0) {
    for (int64_t i = 0; i < x_nnz; ++i) {
      IntT j = x_cols_data[i];
      out_crows_data[j + 2]++;
    }
    for (int64_t i = 0; i < out_dims[0]; i += 1) {
      out_crows_data[i + 1] += out_crows_data[i];
    }
    // compute out_cols_data and out_values_data by out_crows_data and x
    for (int i = 0; i < x_dims[0]; ++i) {
      IntT start = x_crows_data[i];
      IntT end = x_crows_data[i + 1];
      for (IntT j = start; j < end; ++j) {
        IntT x_cols_j = x_cols_data[j] + 1;
        IntT jjj = out_crows_data[x_cols_j];
        out_cols_data[jjj] = i;
        out_values_data[jjj] = x_values_data[j];
        out_crows_data[x_cols_j]++;
      }
    }
  }
}

template <typename T, typename IntT>
__global__ void TransposeCsr3DCudaKernel(const IntT *x_crows_data,
                                         const IntT *x_cols_data,
                                         const T *x_values_data,
                                         const int *perm,
                                         const int64_t *x_dims,
                                         const int64_t *out_dims,
                                         const std::size_t n_dim,
                                         const int64_t x_nnz,
                                         IntT *out_crows_data,
                                         IntT *out_cols_data,
                                         T *out_values_data) {
  int64_t __index__ =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (__index__ == 0) {
    int out_n_rows = out_dims[1];
    int x_n_rows = x_dims[1];
    for (int k = 0; k < out_dims[0]; ++k) {
      if (perm[0] == 0) {  // dims == {0, 2, 1}
        // compute out_crows_data by x_cols_data
        for (int i = 0; i <= out_n_rows; ++i) {
          out_crows_data[i] = 0;
        }
        for (int i = 0; i < x_crows_data[x_n_rows]; ++i) {
          int j = x_cols_data[i];
          out_crows_data[j + 2]++;
        }
        for (int i = 0; i < out_n_rows; ++i) {
          out_crows_data[i + 1] += out_crows_data[i];
        }
        // compute out_cols_data and out_values_data by out_crows_data and x
        for (int i = 0; i < x_n_rows; ++i) {
          IntT start = x_crows_data[i];
          IntT end = x_crows_data[i + 1];
          for (IntT j = start; j < end; ++j) {
            IntT x_cols_j = x_cols_data[j] + 1;
            IntT jjj = out_crows_data[x_cols_j];
            out_cols_data[jjj] = i;
            out_values_data[jjj] = x_values_data[j];
            out_crows_data[x_cols_j]++;
          }
        }
        // x offset
        x_cols_data += x_crows_data[x_n_rows];
        x_values_data += x_crows_data[x_n_rows];
        x_crows_data += x_n_rows + 1;
      } else if (perm[0] == 1 && perm[1] == 0) {  // perm == {1, 0, 2}
        for (int i = 0; i < out_n_rows; ++i) {
          out_crows_data[i] = 0;
        }
        int x_cols_offset = 0;
        int out_cols_index = 0;
        for (int i = 0; i < x_dims[0]; ++i) {
          int x_crows_index = i * (x_n_rows + 1);
          int start = x_crows_data[x_crows_index + k];
          int end = x_crows_data[x_crows_index + 1 + k];
          out_crows_data[i + 1] = end - start;
          for (int j = start; j < end; ++j) {
            out_cols_data[out_cols_index] = x_cols_data[x_cols_offset + j];
            out_values_data[out_cols_index] = x_values_data[x_cols_offset + j];
            out_cols_index++;
          }
          x_cols_offset += x_crows_data[x_crows_index + x_n_rows];
        }
        for (int i = 1; i <= out_n_rows; ++i) {
          out_crows_data[i] += out_crows_data[i - 1];
        }
      }
      // out offset
      out_cols_data += out_crows_data[out_n_rows];
      out_values_data += out_crows_data[out_n_rows];
      out_crows_data += out_n_rows + 1;
    }
  }
}

template <typename T, typename Context>
void TransposeCooKernel(const Context &dev_ctx,
                        const SparseCooTensor &x,
                        const std::vector<int> &perm,
                        SparseCooTensor *out) {
  // create out sparse tensor
  int64_t x_nnz = x.nnz();
  std::size_t n_dim = perm.size();
  DDim out_dims = x.dims().transpose(perm);
  DenseTensor out_indices = EmptyLike<int64_t, Context>(dev_ctx, x.indices());
  DenseTensor out_values(x.values());
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());

  // compute values of indices
  const DenseTensor &x_indices = x.indices();
  const auto *x_indices_data = x_indices.data<int64_t>();
  auto *out_indices_data = out_indices.data<int64_t>();
  int *d_perm;

  auto d_perm_tensor = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int) * perm.size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  d_perm = reinterpret_cast<int *>(d_perm_tensor->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     d_perm,
                     phi::CPUPlace(),
                     perm.data(),
                     sizeof(int) * perm.size(),
                     dev_ctx.stream());
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_nnz * n_dim, 1);
  TransposeCooCudaKernel<<<config.block_per_grid.x,
                           config.thread_per_block.x,
                           0,
                           dev_ctx.stream()>>>(
      x_indices_data, d_perm, n_dim, x_nnz, out_indices_data);
}

template <typename T, typename IntT>
void TransposeCsrGpuKernel(const GPUContext &dev_ctx,
                           const SparseCsrTensor &x,
                           const std::vector<int> &perm,
                           SparseCsrTensor *out) {
  std::size_t n_dim = perm.size();
  const DenseTensor &x_crows = x.crows();
  const DenseTensor &x_cols = x.cols();
  const DenseTensor &x_values = x.non_zero_elements();
  DenseTensor out_crows, out_cols, out_values;
  // return a copy of x
  if (perm[0] == 0 && perm[1] == 1 && (n_dim == 2 || perm[2] == 2)) {
    out_crows = x_crows;
    out_cols = x_cols;
    out_values = x_values;
    out->SetMember(out_crows, out_cols, out_values, x.dims());
    return;
  }
  // create out sparse tensor
  DDim out_dims = x.dims().transpose(perm);
  if (n_dim == 2) {
    out_crows = Empty<IntT, GPUContext>(dev_ctx, {out_dims[0] + 1});
  } else {
    out_crows =
        Empty<IntT, GPUContext>(dev_ctx, {out_dims[0] * (out_dims[1] + 1)});
  }
  out_cols = EmptyLike<IntT, GPUContext>(dev_ctx, x.cols());
  out_values = EmptyLike<T, GPUContext>(dev_ctx, x.values());
  out->SetMember(out_crows, out_cols, out_values, out_dims);
  // transpose by two stages
  if (perm[0] == 1 && perm[1] == 2) {  // perm == {1, 2, 0}
    SparseCsrTensor temp;
    TransposeCsrKernel<T, GPUContext>(dev_ctx, x, {1, 0, 2}, &temp);
    TransposeCsrKernel<T, GPUContext>(dev_ctx, temp, {0, 2, 1}, out);
    return;
  } else if (perm[0] == 2 && perm[1] == 0) {  // perm == {2, 0, 1}
    SparseCsrTensor temp;
    TransposeCsrKernel<T, GPUContext>(dev_ctx, x, {0, 2, 1}, &temp);
    TransposeCsrKernel<T, GPUContext>(dev_ctx, temp, {1, 0, 2}, out);
    return;
  } else if (perm[0] == 2 && perm[1] == 1) {  // perm == {2, 1, 0}
    SparseCsrTensor temp;
    TransposeCsrKernel<T, GPUContext>(dev_ctx, x, {1, 0, 2}, &temp);
    TransposeCsrKernel<T, GPUContext>(dev_ctx, temp, {2, 0, 1}, out);
    return;
  }
  IntT *out_crows_data = out_crows.data<IntT>();
  IntT *out_cols_data = out_cols.data<IntT>();
  T *out_values_data = out_values.data<T>();
  const IntT *x_crows_data = x_crows.data<IntT>();
  const IntT *x_cols_data = x_cols.data<IntT>();
  const T *x_values_data = x_values.data<T>();
  int *d_perm;
  int64_t *d_x_dims, *d_out_dims;

  auto d_perm_tensor = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int) * perm.size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  d_perm = reinterpret_cast<int *>(d_perm_tensor->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     d_perm,
                     phi::CPUPlace(),
                     perm.data(),
                     sizeof(int) * perm.size(),
                     dev_ctx.stream());
  auto d_x_dims_tensor = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int64_t) * x.dims().size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  d_x_dims = reinterpret_cast<int64_t *>(d_x_dims_tensor->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     d_x_dims,
                     phi::CPUPlace(),
                     x.dims().Get(),
                     sizeof(int64_t) * x.dims().size(),
                     dev_ctx.stream());
  auto d_out_dims_tensor = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int64_t) * out_dims.size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  d_out_dims = reinterpret_cast<int64_t *>(d_out_dims_tensor->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     d_out_dims,
                     phi::CPUPlace(),
                     out_dims.Get(),
                     sizeof(int64_t) * out_dims.size(),
                     dev_ctx.stream());

  int64_t x_nnz = x.nnz();
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_dims[0], 1);
  if (perm.size() == 2) {
    TransposeCsr2DCudaKernel<T><<<config.block_per_grid.x,
                                  config.thread_per_block.x,
                                  0,
                                  dev_ctx.stream()>>>(x_crows_data,
                                                      x_cols_data,
                                                      x_values_data,
                                                      d_perm,
                                                      d_x_dims,
                                                      d_out_dims,
                                                      x_nnz,
                                                      out_crows_data,
                                                      out_cols_data,
                                                      out_values_data);
  } else {
    TransposeCsr3DCudaKernel<T><<<1, 1, 0, dev_ctx.stream()>>>(x_crows_data,
                                                               x_cols_data,
                                                               x_values_data,
                                                               d_perm,
                                                               d_x_dims,
                                                               d_out_dims,
                                                               perm.size(),
                                                               x_nnz,
                                                               out_crows_data,
                                                               out_cols_data,
                                                               out_values_data);
  }
}

template <typename T, typename Context>
void TransposeCsrKernel(const Context &dev_ctx,
                        const SparseCsrTensor &x,
                        const std::vector<int> &perm,
                        SparseCsrTensor *out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.crows().dtype(), "TransposeCsrKernel", ([&] {
                                 TransposeCsrGpuKernel<T, data_t>(
                                     dev_ctx, x, perm, out);
                               }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(transpose_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::TransposeCooKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(transpose_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::TransposeCsrKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
