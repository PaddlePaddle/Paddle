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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"
namespace phi {
namespace sparse {

template <typename T>
struct DivScalarFunctor {
  T value_;

  explicit DivScalarFunctor(T value) : value_(value) {}

  __device__ __forceinline__ T operator()(const T x) const {
    return x / value_;
  }
};

__global__ void TransposeCooCudaKernel(const int64_t *x_indices_data,
                                       const int *perm,
                                       const std::size_t n_dim,
                                       const int64_t x_nnz,
                                       int64_t *out_indices_data) {
  for (std::size_t i = 0; i < n_dim; ++i) {
    CUDA_KERNEL_LOOP_TYPE(j, x_nnz, int64_t) {
      out_indices_data[j + i * x_nnz] = x_indices_data[j + perm[i] * x_nnz];
    }
  }
}

template <typename T>
__global__ void TransposeCsr2DCudaKernel(const int64_t *x_crows_data,
                                         const int64_t *x_cols_data,
                                         const T *x_values_data,
                                         const int *perm,
                                         const int64_t *x_dims,
                                         const int64_t *out_dims,
                                         const int64_t x_nnz,
                                         int64_t *out_crows_data,
                                         int64_t *out_cols_data,
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
      int j = x_cols_data[i];
      out_crows_data[j + 2]++;
    }
    for (int64_t i = 0; i < out_dims[0]; i += 1) {
      out_crows_data[i + 1] += out_crows_data[i];
    }
    // compute out_cols_data and out_values_data by out_crows_data and x
    for (int i = 0; i < x_dims[0]; ++i) {
      int64_t start = x_crows_data[i];
      int64_t end = x_crows_data[i + 1];
      for (int64_t j = start; j < end; ++j) {
        int64_t x_cols_j = x_cols_data[j] + 1;
        int64_t jjj = out_crows_data[x_cols_j];
        out_cols_data[jjj] = i;
        out_values_data[jjj] = x_values_data[j];
        out_crows_data[x_cols_j]++;
      }
    }
  }
}

template <typename T>
__global__ void TransposeCsr3DCudaKernel(const int64_t *x_crows_data,
                                         const int64_t *x_cols_data,
                                         const T *x_values_data,
                                         const int *perm,
                                         const int64_t *x_dims,
                                         const int64_t *out_dims,
                                         const std::size_t n_dims,
                                         const int64_t x_nnz,
                                         int64_t *out_crows_data,
                                         int64_t *out_cols_data,
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
          int64_t start = x_crows_data[i];
          int64_t end = x_crows_data[i + 1];
          for (int64_t j = start; j < end; ++j) {
            int64_t x_cols_j = x_cols_data[j] + 1;
            int64_t jjj = out_crows_data[x_cols_j];
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
void DivCooScalarKernel(const Context &dev_ctx,
                        const SparseCooTensor &x,
                        float scalar,
                        SparseCooTensor *out) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);

  std::vector<const DenseTensor *> ins = {&(x.values())};
  std::vector<DenseTensor *> outs = {out->mutable_values()};
  DivScalarFunctor<T> func(static_cast<T>(scalar));
  funcs::ElementwiseKernel<T, DivScalarFunctor<T>>(dev_ctx, ins, &outs, func);
}

template <typename T, typename Context>
void DivCsrScalarKernel(const Context &dev_ctx,
                        const SparseCsrTensor &x,
                        float scalar,
                        SparseCsrTensor *out) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);

  std::vector<const DenseTensor *> ins = {&(x.values())};
  std::vector<DenseTensor *> outs = {out->mutable_values()};
  DivScalarFunctor<T> func(static_cast<T>(scalar));
  funcs::ElementwiseKernel<T, DivScalarFunctor<T>>(dev_ctx, ins, &outs, func);
}

template <typename T, typename Context>
void TransposeCooKernel(const Context &dev_ctx,
                        const SparseCooTensor &x,
                        const std::vector<int> &perm,
                        SparseCooTensor *out) {
  // create out sparse tensor
  int64_t x_nnz = x.nnz();
  DDim out_dims = x.dims().transpose(perm);
  DenseTensor out_indices = EmptyLike<int64_t, Context>(dev_ctx, x.indices());
  DenseTensor out_values(x.values());
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());

  // compute values of indices
  const DenseTensor &x_indices = x.indices();
  const auto *x_indices_data = x_indices.data<int64_t>();
  auto *out_indices_data = out_indices.data<int64_t>();
  int *d_perm;
#ifdef PADDLE_WITH_HIP
  hipMalloc(reinterpret_cast<void **>(&d_perm), sizeof(int) * perm.size());
  hipMemcpy(
      d_perm, perm.data(), sizeof(int) * perm.size(), hipMemcpyHostToDevice);
#else
  cudaMalloc(reinterpret_cast<void **>(&d_perm), sizeof(int) * perm.size());
  cudaMemcpy(
      d_perm, perm.data(), sizeof(int) * perm.size(), cudaMemcpyHostToDevice);
#endif
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_nnz, 1);
  TransposeCooCudaKernel<<<config.block_per_grid.x,
                           config.thread_per_block.x,
                           0,
                           dev_ctx.stream()>>>(
      x_indices_data, d_perm, perm.size(), x_nnz, out_indices_data);
}

template <typename T, typename Context>
void TransposeCsrKernel(const Context &dev_ctx,
                        const SparseCsrTensor &x,
                        const std::vector<int> &perm,
                        SparseCsrTensor *out) {
  unsigned int n_dim = perm.size();
  // create out sparse tensor
  DDim out_dims = x.dims().transpose(perm);
  DenseTensor out_crows;
  if (n_dim == 2) {
    out_crows = Empty<int64_t, Context>(dev_ctx, {out_dims[0] + 1});
  } else {
    out_crows =
        Empty<int64_t, Context>(dev_ctx, {out_dims[0] * (out_dims[1] + 1)});
  }
  DenseTensor out_cols = EmptyLike<int64_t, Context>(dev_ctx, x.cols());
  DenseTensor out_values = EmptyLike<T, Context>(dev_ctx, x.values());
  out->SetMember(out_crows, out_cols, out_values, out_dims);

  const DenseTensor &x_crows = x.crows();
  const DenseTensor &x_cols = x.cols();
  const DenseTensor &x_values = x.non_zero_elements();

  // return a copy of x
  if (perm[0] == 0 && perm[1] == 1 && (n_dim == 2 || perm[2] == 2)) {
    phi::Copy(dev_ctx, x_crows, dev_ctx.GetPlace(), false, &out_crows);
    phi::Copy(dev_ctx, x_cols, dev_ctx.GetPlace(), false, &out_cols);
    phi::Copy(dev_ctx, x_values, dev_ctx.GetPlace(), false, &out_values);
    return;
  }
  // transpose by two stages
  if (perm[0] == 1 && perm[1] == 2) {  // perm == {1, 2, 0}
    SparseCsrTensor temp;
    TransposeCsrKernel<T, Context>(dev_ctx, x, {1, 0, 2}, &temp);
    TransposeCsrKernel<T, Context>(dev_ctx, temp, {0, 2, 1}, out);
    return;
  } else if (perm[0] == 2 && perm[1] == 0) {  // perm == {2, 0, 1}
    SparseCsrTensor temp;
    TransposeCsrKernel<T, Context>(dev_ctx, x, {0, 2, 1}, &temp);
    TransposeCsrKernel<T, Context>(dev_ctx, temp, {1, 0, 2}, out);
    return;
  } else if (perm[0] == 2 && perm[1] == 1) {  // perm == {2, 1, 0}
    SparseCsrTensor temp;
    TransposeCsrKernel<T, Context>(dev_ctx, x, {1, 0, 2}, &temp);
    TransposeCsrKernel<T, Context>(dev_ctx, temp, {2, 0, 1}, out);
    return;
  }
  int64_t *out_crows_data = out_crows.data<int64_t>();
  int64_t *out_cols_data = out_cols.data<int64_t>();
  T *out_values_data = out_values.data<T>();
  const int64_t *x_crows_data = x_crows.data<int64_t>();
  const int64_t *x_cols_data = x_cols.data<int64_t>();
  const T *x_values_data = x_values.data<T>();
  int *d_perm;
  int64_t *d_x_dims, *d_out_dims;
#ifdef PADDLE_WITH_HIP
  hipMalloc(reinterpret_cast<void **>(&d_perm), sizeof(int) * perm.size());
  hipMemcpy(
      d_perm, perm.data(), sizeof(int) * perm.size(), hipMemcpyHostToDevice);
  hipMalloc(reinterpret_cast<void **>(&d_x_dims),
            sizeof(int64_t) * x.dims().size());
  hipMemcpy(d_x_dims,
            x.dims().Get(),
            sizeof(int64_t) * x.dims().size(),
            hipMemcpyHostToDevice);
  hipMalloc(reinterpret_cast<void **>(&d_out_dims),
            sizeof(int64_t) * out_dims.size());
  hipMemcpy(d_out_dims,
            out_dims.Get(),
            sizeof(int64_t) * out_dims.size(),
            hipMemcpyHostToDevice);
#else
  cudaMalloc(reinterpret_cast<void **>(&d_perm), sizeof(int) * perm.size());
  cudaMemcpy(
      d_perm, perm.data(), sizeof(int) * perm.size(), cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&d_x_dims),
             sizeof(int64_t) * x.dims().size());
  cudaMemcpy(d_x_dims,
             x.dims().Get(),
             sizeof(int64_t) * x.dims().size(),
             cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&d_out_dims),
             sizeof(int64_t) * out_dims.size());
  cudaMemcpy(d_out_dims,
             out_dims.Get(),
             sizeof(int64_t) * out_dims.size(),
             cudaMemcpyHostToDevice);
#endif
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
}  // namespace sparse
}  // namespace phi

#define PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(name, prefix)          \
  PD_REGISTER_KERNEL(name##_coo,                                   \
                     GPU,                                          \
                     ALL_LAYOUT,                                   \
                     phi::sparse::prefix##CooKernel,               \
                     phi::dtype::float16,                          \
                     float,                                        \
                     double) {                                     \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO); \
  }                                                                \
                                                                   \
  PD_REGISTER_KERNEL(name##_csr,                                   \
                     GPU,                                          \
                     ALL_LAYOUT,                                   \
                     phi::sparse::prefix##CsrKernel,               \
                     phi::dtype::float16,                          \
                     float,                                        \
                     double) {                                     \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR); \
  }

PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(sin, Sin)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(tan, Tan)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(asin, Asin)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(atan, Atan)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(sinh, Sinh)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(tanh, Tanh)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(asinh, Asinh)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(atanh, Atanh)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(sqrt, Sqrt)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(square, Square)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(log1p, Log1p)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(relu, Relu)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(abs, Abs)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(pow, Pow)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(scale, Scale)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(expm1, Expm1)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(relu6, Relu6)
PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(leaky_relu, LeakyRelu)

PD_REGISTER_KERNEL(divide_coo_scalar,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::DivCooScalarKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(divide_csr_scalar,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::DivCsrScalarKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(cast_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CastCooKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(cast_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CastCsrKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

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
