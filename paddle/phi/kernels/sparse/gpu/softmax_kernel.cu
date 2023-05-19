/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/softmax_kernel.h"

#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/softmax.cu.h"
#include "paddle/phi/kernels/softmax_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT = int>
__global__ void SoftmaxGpuKernel(const IntT* x_crows,
                                 const T* x_values,
                                 T* out_values,
                                 int row_number,
                                 int total_row_number) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  int non_zero_idx = threadIdx.x;
  if (row >= total_row_number) return;
  int cur_batch = row / row_number;
  int crow_idx = cur_batch * (row_number + 1) + (row % row_number);
  int cur_batch_offset = 0;
  for (int i = 1; i < cur_batch + 1; ++i) {
    cur_batch_offset += x_crows[i * (row_number + 1) - 1];
  }
  int row_first = cur_batch_offset + static_cast<int>(x_crows[crow_idx]);
  int row_nnz = static_cast<int>(x_crows[crow_idx + 1] - x_crows[crow_idx]);
  if (row_nnz == 0) return;

  int kIteration = (row_nnz + warpSize - 1) / warpSize;

  T max_val = -std::numeric_limits<T>::infinity();
  for (int i = 0; i < kIteration; ++i) {
    int idx = non_zero_idx + i * warpSize;
    if (idx >= row_nnz) break;

    T val = x_values[row_first + idx];
    if (val > max_val) {
      max_val = val;
    }
  }
  T row_max_val = phi::funcs::WarpReduceMax<T>(max_val, 0xFFFFFFFF);

  T exp_sum = 0;
  for (int i = 0; i < kIteration; ++i) {
    int idx = non_zero_idx + i * warpSize;
    if (idx >= row_nnz) break;

    auto functor = phi::funcs::CudaExpFunctor<T>();
    T exp = functor(x_values[row_first + idx] - row_max_val);
    exp_sum += exp;
    out_values[row_first + idx] = exp;
  }
  T row_exp_sum = phi::funcs::WarpReduceSum<T>(exp_sum, 0xFFFFFFFF);

  for (int i = 0; i < kIteration; ++i) {
    int idx = non_zero_idx + i * warpSize;
    if (idx >= row_nnz) break;

    out_values[row_first + idx] = out_values[row_first + idx] / row_exp_sum;
  }
}

template <typename T, typename Context>
void SoftmaxCsrKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      int axis,
                      SparseCsrTensor* out) {
  PADDLE_ENFORCE_EQ(axis,
                    -1,
                    phi::errors::Unimplemented(
                        "SparseCsrTensor only support axis=-1 for softmax, "
                        "which is faster when reading data by row (axis=-1)"));
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);
  auto x_dim = x.dims();
  auto x_rank = x_dim.size();

  int total_row_number = 1;
  int row_number = 1;
  for (int i = 0; i < x_rank - 1; ++i) {
    total_row_number *= x_dim[i];
    if (i == x_rank - 2) {
      row_number = x_dim[i];
    }
  }

  dim3 grid((total_row_number + 3) / 4);
  dim3 block(32, 4);

  PD_VISIT_BASE_INTEGRAL_TYPES(x.crows().dtype(), "CsrSoftmaxKernel", ([&] {
                                 SoftmaxGpuKernel<T, data_t>
                                     <<<grid, block, 0, dev_ctx.stream()>>>(
                                         x.crows().data<data_t>(),
                                         x.values().data<T>(),
                                         out->mutable_values()->data<T>(),
                                         row_number,
                                         total_row_number);
                               }));
}

template <typename T, typename IntT>
__global__ void SoftmaxCooGPURawKernel(IntT* sorted_pool_indices,
                                       IntT size,
                                       IntT* pool_sizes,
                                       IntT* pool_offsets,
                                       IntT nvalues,
                                       T* mx_rows,
                                       T* input_values,
                                       T* output_values) {
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int index = tid + blkid * blksz;
  int step = blksz * gridsz;

  while (index < size) {
    IntT offset = pool_offsets[index];
    IntT* pool_indices = sorted_pool_indices + offset;
    IntT pool_indices_size = pool_sizes[index];
    T* mx_row = mx_rows + index * nvalues;

    for (IntT j = 0; j < nvalues; j++) {
      T exp_sums = 0;
      for (IntT p = 0; p < pool_indices_size; p++) {
        auto i = pool_indices[p];
        auto cur_value = input_values + i * nvalues + j;
        auto cur_out_value = output_values + i * nvalues + j;
        auto v = std::exp((*cur_value) - (*(mx_row + j)));
        *cur_out_value = v;
        exp_sums += v;
      }
      for (IntT p = 0; p < pool_indices_size; p++) {
        auto i = pool_indices[p];
        auto out_values_row = output_values + i * nvalues;
        out_values_row[j] *= 1.0 / exp_sums;
      }
    }
    index += step;
  }
}

template <typename T, typename IntT, typename Context>
void SoftmaxCooGPUKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         int axis,
                         SparseCooTensor* out) {
  auto indices = x.indices();
  auto values = x.values();
  const auto x_dims = x.dims();
  const std::vector<IntT> sizes = phi::vectorize<IntT>(x_dims);
  const auto sparse_dim = x.sparse_dim();
  const IntT x_nnz = x.nnz();
  DenseTensor out_indices(indices);
  DenseTensor out_values = EmptyLike<T, Context>(dev_ctx, values);
  out->SetMember(out_indices, out_values, x.dims(), x.coalesced());

  int dim = axis < 0 ? x_dims.size() + axis : axis;

  /* If dim is greater than or equal to sparse_dim, the dense softmax is used.
   */
  if (dim >= sparse_dim) {
    SoftmaxKernel<T, Context>(
        dev_ctx, values, dim - sparse_dim + 1, &out_values);
    return;
  }

  auto stream = dev_ctx.stream();
  IntT nvalues = std::accumulate(sizes.begin() + sparse_dim,
                                 sizes.end(),
                                 static_cast<IntT>(1),
                                 std::multiplies<>());

  auto values_2 = values.Resize({x_nnz, nvalues});
  auto out_values_2 = out_values.Resize({x_nnz, nvalues});

  /* Compute independent pools of indices */
  DenseTensor sorted_indices;
  DenseTensor pool_offsets;
  DenseTensor pool_sizes;
  DenseTensor mx_buffer;
  std::tie(sorted_indices, pool_offsets, pool_sizes, mx_buffer) =
      phi::funcs::sparse::ComputePoolMax<T, IntT, Context, true>(
          dev_ctx, indices, values_2, sizes, nvalues, static_cast<IntT>(dim));

  auto pool_size = pool_offsets.dims()[0];
  int block_size = phi::funcs::sparse::GetNumThreads(pool_size);
  const int grid_size = (pool_size + block_size - 1) / block_size;
  auto out_values_ptr = out_values.data<T>();
  auto values_ptr = values.data<T>();

  /* Compute softmax results with pool indices */
  SoftmaxCooGPURawKernel<T, IntT>
      <<<grid_size, block_size, 0, stream>>>(sorted_indices.data<IntT>(),
                                             pool_size,
                                             pool_sizes.data<IntT>(),
                                             pool_offsets.data<IntT>(),
                                             nvalues,
                                             mx_buffer.data<T>(),
                                             values_ptr,
                                             out_values_ptr);
}

template <typename T, typename Context>
void SoftmaxCooKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      int axis,
                      SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "SoftmaxCooGPUKernel", ([&] {
        SoftmaxCooGPUKernel<T, data_t, Context>(dev_ctx, x, axis, out);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(softmax_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(softmax_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
