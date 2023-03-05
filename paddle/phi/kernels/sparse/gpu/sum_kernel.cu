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
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T>
__global__ void SetValueCudaKernel(const T value,
                                   const int64_t length,
                                   T* data) {
  CUDA_KERNEL_LOOP_TYPE(index, length, int64_t) { data[index] = value; }
}

template <typename T>
__global__ void SumCooCudaKernel(const int64_t* x_indices_data,
                                 const T* x_values_data,
                                 const std::size_t n_dim,
                                 const int64_t x_nnz,
                                 int64_t* out_indices_data,
                                 T* out_values_data) {
  CUDA_KERNEL_LOOP_TYPE(index, x_nnz * n_dim, int64_t) {}
}
//
// template <typename T>
// __global__ void SumCsr2DCudaKernel(const int64_t *x_crows_data,
//                                   const int64_t *x_cols_data,
//                                   const T *x_values_data,
//                                   const int *perm,
//                                   const int64_t *x_dims,
//                                   const int64_t *out_dims,
//                                   const int64_t x_nnz,
//                                   int64_t *out_crows_data,
//                                   int64_t *out_cols_data,
//                                   T *out_values_data) {
//  int64_t __index__ =
//      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//  // compute out_crows_data by x_cols_data
//  for (int64_t i = __index__; i <= out_dims[0]; i += blockDim.x * gridDim.x) {
//    out_crows_data[i] = 0;
//  }
//  __syncthreads();
//  if (__index__ == 0) {
//    for (int64_t i = 0; i < x_nnz; ++i) {
//      int j = x_cols_data[i];
//      out_crows_data[j + 2]++;
//    }
//    for (int64_t i = 0; i < out_dims[0]; i += 1) {
//      out_crows_data[i + 1] += out_crows_data[i];
//    }
//    // compute out_cols_data and out_values_data by out_crows_data and x
//    for (int i = 0; i < x_dims[0]; ++i) {
//      int64_t start = x_crows_data[i];
//      int64_t end = x_crows_data[i + 1];
//      for (int64_t j = start; j < end; ++j) {
//        int64_t x_cols_j = x_cols_data[j] + 1;
//        int64_t jjj = out_crows_data[x_cols_j];
//        out_cols_data[jjj] = i;
//        out_values_data[jjj] = x_values_data[j];
//        out_crows_data[x_cols_j]++;
//      }
//    }
//  }
//}
//
// template <typename T>
// __global__ void SumCsr3DCudaKernel(const int64_t *x_crows_data,
//                                   const int64_t *x_cols_data,
//                                   const T *x_values_data,
//                                   const int *perm,
//                                   const int64_t *x_dims,
//                                   const int64_t *out_dims,
//                                   const std::size_t n_dim,
//                                   const int64_t x_nnz,
//                                   int64_t *out_crows_data,
//                                   int64_t *out_cols_data,
//                                   T *out_values_data) {
//  int64_t __index__ =
//      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//  if (__index__ == 0) {
//    int out_n_rows = out_dims[1];
//    int x_n_rows = x_dims[1];
//    for (int k = 0; k < out_dims[0]; ++k) {
//      if (perm[0] == 0) {  // dims == {0, 2, 1}
//        // compute out_crows_data by x_cols_data
//        for (int i = 0; i <= out_n_rows; ++i) {
//          out_crows_data[i] = 0;
//        }
//        for (int i = 0; i < x_crows_data[x_n_rows]; ++i) {
//          int j = x_cols_data[i];
//          out_crows_data[j + 2]++;
//        }
//        for (int i = 0; i < out_n_rows; ++i) {
//          out_crows_data[i + 1] += out_crows_data[i];
//        }
//        // compute out_cols_data and out_values_data by out_crows_data and x
//        for (int i = 0; i < x_n_rows; ++i) {
//          int64_t start = x_crows_data[i];
//          int64_t end = x_crows_data[i + 1];
//          for (int64_t j = start; j < end; ++j) {
//            int64_t x_cols_j = x_cols_data[j] + 1;
//            int64_t jjj = out_crows_data[x_cols_j];
//            out_cols_data[jjj] = i;
//            out_values_data[jjj] = x_values_data[j];
//            out_crows_data[x_cols_j]++;
//          }
//        }
//        // x offset
//        x_cols_data += x_crows_data[x_n_rows];
//        x_values_data += x_crows_data[x_n_rows];
//        x_crows_data += x_n_rows + 1;
//      } else if (perm[0] == 1 && perm[1] == 0) {  // perm == {1, 0, 2}
//        for (int i = 0; i < out_n_rows; ++i) {
//          out_crows_data[i] = 0;
//        }
//        int x_cols_offset = 0;
//        int out_cols_index = 0;
//        for (int i = 0; i < x_dims[0]; ++i) {
//          int x_crows_index = i * (x_n_rows + 1);
//          int start = x_crows_data[x_crows_index + k];
//          int end = x_crows_data[x_crows_index + 1 + k];
//          out_crows_data[i + 1] = end - start;
//          for (int j = start; j < end; ++j) {
//            out_cols_data[out_cols_index] = x_cols_data[x_cols_offset + j];
//            out_values_data[out_cols_index] = x_values_data[x_cols_offset +
//            j]; out_cols_index++;
//          }
//          x_cols_offset += x_crows_data[x_crows_index + x_n_rows];
//        }
//        for (int i = 1; i <= out_n_rows; ++i) {
//          out_crows_data[i] += out_crows_data[i - 1];
//        }
//      }
//      // out offset
//      out_cols_data += out_crows_data[out_n_rows];
//      out_values_data += out_crows_data[out_n_rows];
//      out_crows_data += out_n_rows + 1;
//    }
//  }
//}

template <typename T, typename Context>
void SumCooKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCooTensor* out) {
  size_t n_dim = axis.size();
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
    } else {
      out_dims = make_ddim({1});
      out_indices_shape = {1, 1};
      out_indices = Empty<int64_t, Context>(dev_ctx, out_indices_shape);
    }
    auto* out_indices_data = out_indices.data<int64_t>();
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, out_indices.dims()[0], 1);
    SetValueCudaKernel<int64_t>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(0, out_indices.dims()[0], out_indices_data);
    out_values = phi::Sum<T>(dev_ctx, x.values(), {}, dtype, true);
  } else {
    // TODO(zrr1999)
    auto dim = axis[0] < 0 ? x_dims.size() + axis[0] : axis[0];
    const auto* x_indices_data = x_indices.data<int64_t>();
    const auto* x_values_data = x_values.data<T>();
    std::map<std::vector<int64_t>, std::vector<int64_t>> map_indices;
    for (int64_t j = 0; j < x_indices.dims()[1]; ++j) {
      std::vector<int64_t> pos;
      for (int64_t i = 0; i < x_indices.dims()[0]; ++i) {
        if (dim != i) {
          pos.emplace_back(x_indices_data[j + i * x_indices.dims()[1]]);
        } else if (keep_dim) {
          pos.emplace_back(0);
        }
      }
      map_indices[pos].emplace_back(j);
    }

    std::vector<int64_t> dims;
    if (keep_dim) {
      for (int i = 0; i < x.dims().size(); ++i) {
        if (i == dim) {
          dims.emplace_back(1);
        } else {
          dims.emplace_back(x.dims()[i]);
        }
      }
      out_dims = make_ddim(dims);

      out_indices = Empty<int64_t, Context>(
          dev_ctx, {x_dims.size(), static_cast<int>(map_indices.size())});
    } else {
      for (int i = 0; i < x.dims().size(); ++i) {
        if (i != dim) {
          dims.emplace_back(x.dims()[i]);
        }
      }
      out_dims = make_ddim(dims);
      out_indices = Empty<int64_t, Context>(
          dev_ctx, {x_dims.size() - 1, static_cast<int>(map_indices.size())});
    }
    out_values =
        Empty<T, Context>(dev_ctx, {static_cast<int>(map_indices.size())});
    auto* out_indices_data = out_indices.data<int64_t>();
    auto* out_values_data = out_values.data<T>();

    auto iter_map_indices = map_indices.begin();
    for (size_t j = 0; j < map_indices.size(); ++j) {
      std::vector<int64_t> pos = iter_map_indices->first;
      std::vector<int64_t> values_index = iter_map_indices->second;
      iter_map_indices++;
      T out_value = 0;
      for (auto index : values_index) {
        out_value += x_values_data[index];
      }
      for (auto i = 0; i < out_indices.dims()[0]; ++i) {
        out_indices_data[j + i * map_indices.size()] = pos[i];
      }
      out_values_data[j] = out_value;
    }
  }

  out->SetMember(out_indices, out_values, out_dims, x.coalesced());

  // create out sparse tensor
  //  int64_t x_nnz = x.nnz();
  //  std::size_t n_dim = perm.size();
  //  DDim out_dims = x.dims().Sum(perm);
  //  DenseTensor out_indices = EmptyLike<int64_t, Context>(dev_ctx,
  //  x.indices()); DenseTensor out_values(x.values());
  //  out->SetMember(out_indices, out_values, out_dims, x.coalesced());
  //
  //  // compute values of indices
  //  const DenseTensor &x_indices = x.indices();
  //  const auto *x_indices_data = x_indices.data<int64_t>();
  //  auto *out_indices_data = out_indices.data<int64_t>();
  //  int *d_perm;
  // #ifdef PADDLE_WITH_HIP
  //  hipMalloc(reinterpret_cast<void **>(&d_perm), sizeof(int) * perm.size());
  //  hipMemcpy(
  //      d_perm, perm.data(), sizeof(int) * perm.size(),
  //      hipMemcpyHostToDevice);
  // #else
  //  cudaMalloc(reinterpret_cast<void **>(&d_perm), sizeof(int) * perm.size());
  //  cudaMemcpy(
  //      d_perm, perm.data(), sizeof(int) * perm.size(),
  //      cudaMemcpyHostToDevice);
  // #endif
  //  auto config =
  //      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_nnz * n_dim, 1);
  //  SumCooCudaKernel<<<config.block_per_grid.x,
  //                     config.thread_per_block.x,
  //                     0,
  //                     dev_ctx.stream()>>>(
  //      x_indices_data, d_perm, n_dim, x_nnz, out_indices_data);
}

template <typename T, typename Context>
void SumCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  const IntArray& axis,
                  DataType dtype,
                  bool keep_dim,
                  SparseCsrTensor* out) {
  size_t n_dim = axis.size();
  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_values = x.values();
  const auto* x_crows_data = x_crows.data<int64_t>();
  const T* x_values_data = x_values.data<T>();

  DenseTensor out_crows, out_cols, out_values;
  DDim out_dims;
  if (n_dim == 0) {
    // TODO(zrr1999)
    out_dims = make_ddim({1, 1});
    out_crows = Empty<int64_t, Context>(dev_ctx, {2});  // crows = [0, 1]
    auto* out_crows_data = out_crows.data<int64_t>();
    out_crows_data[0] = 0;
    out_crows_data[1] = 1;

    out_cols = Empty<int64_t, Context>(dev_ctx, {1});  // crows = [0]
    auto* out_cols_data = out_cols.data<int64_t>();
    out_cols_data[0] = 0;

    out_values = phi::Sum<T>(dev_ctx, x.values(), {}, dtype, true);
  } else {
    // TODO(zrr1999)
    PADDLE_ENFORCE_EQ(axis[0],
                      -1,
                      phi::errors::Unimplemented(
                          "`axis` of SumCsrKernel only support None or -1 now."
                          "More number will be supported in the future."));
    out_dims = make_ddim({x.dims()[0], 1});
    out_crows = EmptyLike<int64_t, Context>(dev_ctx, x.crows());

    std::vector<T> out_data;
    auto* out_crows_data = out_crows.data<int64_t>();
    out_crows_data[0] = 0;
    for (int i = 0; i < x.dims()[0]; ++i) {
      if (x_crows_data[i] != x_crows_data[i + 1]) {
        T sum_value(0);
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

  // #ifdef PADDLE_WITH_HIP
  //   hipMalloc(reinterpret_cast<void **>(&d_perm), sizeof(int) * perm.size());
  //   hipMemcpy(
  //       d_perm, perm.data(), sizeof(int) * perm.size(),
  //       hipMemcpyHostToDevice);
  //   hipMalloc(reinterpret_cast<void **>(&d_x_dims),
  //             sizeof(int64_t) * x.dims().size());
  //   hipMemcpy(d_x_dims,
  //             x.dims().Get(),
  //             sizeof(int64_t) * x.dims().size(),
  //             hipMemcpyHostToDevice);
  //   hipMalloc(reinterpret_cast<void **>(&d_out_dims),
  //             sizeof(int64_t) * out_dims.size());
  //   hipMemcpy(d_out_dims,
  //             out_dims.Get(),
  //             sizeof(int64_t) * out_dims.size(),
  //             hipMemcpyHostToDevice);
  // #else
  //   cudaMalloc(reinterpret_cast<void **>(&d_perm), sizeof(int) *
  //   perm.size()); cudaMemcpy(
  //       d_perm, perm.data(), sizeof(int) * perm.size(),
  //       cudaMemcpyHostToDevice);
  //   cudaMalloc(reinterpret_cast<void **>(&d_x_dims),
  //              sizeof(int64_t) * x.dims().size());
  //   cudaMemcpy(d_x_dims,
  //              x.dims().Get(),
  //              sizeof(int64_t) * x.dims().size(),
  //              cudaMemcpyHostToDevice);
  //   cudaMalloc(reinterpret_cast<void **>(&d_out_dims),
  //              sizeof(int64_t) * out_dims.size());
  //   cudaMemcpy(d_out_dims,
  //              out_dims.Get(),
  //              sizeof(int64_t) * out_dims.size(),
  //              cudaMemcpyHostToDevice);
  // #endif
  //   int64_t x_nnz = x.nnz();
  //   auto config =
  //       phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_dims[0], 1);
  //   if (perm.size() == 2) {
  //     SumCsr2DCudaKernel<T><<<config.block_per_grid.x,
  //                             config.thread_per_block.x,
  //                             0,
  //                             dev_ctx.stream()>>>(x_crows_data,
  //                                                 x_cols_data,
  //                                                 x_values_data,
  //                                                 d_perm,
  //                                                 d_x_dims,
  //                                                 d_out_dims,
  //                                                 x_nnz,
  //                                                 out_crows_data,
  //                                                 out_cols_data,
  //                                                 out_values_data);
  //   } else {
  //     SumCsr3DCudaKernel<T><<<1, 1, 0, dev_ctx.stream()>>>(x_crows_data,
  //                                                          x_cols_data,
  //                                                          x_values_data,
  //                                                          d_perm,
  //                                                          d_x_dims,
  //                                                          d_out_dims,
  //                                                          perm.size(),
  //                                                          x_nnz,
  //                                                          out_crows_data,
  //                                                          out_cols_data,
  //                                                          out_values_data);
  //   }
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
                   bool) {}

PD_REGISTER_KERNEL(sum_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCsrKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
