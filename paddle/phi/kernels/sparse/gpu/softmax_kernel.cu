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

static int GetNumThreads(int nElem) {
#if defined(PADLDE_WITH_ROCM)
  int threadSizes[5] = {16, 32, 64, 128, 256};
#else
  int threadSizes[5] = {32, 64, 128, 256, 512};
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return threadSizes[4];
}

/* Given the indices of a sparse tensor, return a vector of offsets
for the entries in the equivalent dense tensor. */
template <typename IntT, typename Context>
DenseTensor GetOffsets(const Context& dev_ctx,
                       const DenseTensor& indices,
                       const std::vector<IntT>& sizes,
                       const IntT dim) {
#ifdef __HIPCC__
  const auto& policy = thrust::hip::par.on(dev_ctx.stream());
#else
  const auto& policy = thrust::cuda::par.on(dev_ctx.stream());
#endif

  auto ndim = indices.dims()[0];
  auto nnz = indices.dims()[1];
  std::vector<IntT> host_strides(ndim, 1);
  if (ndim > 1) {
    for (IntT i = ndim - 2; i >= 0; i--) {
      host_strides[i] = host_strides[i + 1] * (i + 1 == dim ? 1 : sizes[i + 1]);
    }
  }

  const IntArray strides_shape(phi::vectorize<IntT>(indices.dims()));
  DenseTensor strides = phi::Empty<IntT>(dev_ctx, strides_shape);
  auto strides_ptr = strides.data<IntT>();
  memory_utils::Copy(dev_ctx.GetPlace(),
                     strides_ptr,
                     phi::CPUPlace(),
                     host_strides.data(),
                     sizeof(IntT) * host_strides.size(),
                     dev_ctx.stream());

  DenseTensor offsets = phi::Empty<IntT>(dev_ctx, {nnz});
  auto indices_ptr = indices.data<IntT>();

  thrust::transform(
      policy,
      thrust::make_counting_iterator(IntT(0)),
      thrust::make_counting_iterator(IntT(nnz)),
      thrust::device_ptr<IntT>(offsets.data<IntT>()),
      [strides_ptr, indices_ptr, nnz, dim, ndim] __device__(IntT x) {
        IntT pool_index = 0;
        for (IntT j = 0; j < ndim; j++) {
          if (j != dim) {
            auto indice_cur_ptr = indices_ptr + j * nnz + x;
            auto stride = strides_ptr[j];
            pool_index += stride * (*indice_cur_ptr);
          }
        }
        return pool_index;
      });
  return offsets;
}

/* Return pools of indices that align with the given dimension and the
corresponding max values for each pool. */
template <typename T,
          typename IntT,
          typename Context,
          bool requireMxRows = true>
std::tuple<DenseTensor, DenseTensor, DenseTensor, DenseTensor> ComputePoolMax(
    const Context& dev_ctx,
    const DenseTensor& indices,
    const DenseTensor& values,
    const std::vector<IntT>& sizes,
    IntT nvalues,
    const IntT dim) {
#ifdef __HIPCC__
  const auto& policy = thrust::hip::par.on(dev_ctx.stream());
#else
  const auto& policy = thrust::cuda::par.on(dev_ctx.stream());
#endif
  using thrust_ptr = thrust::device_ptr<IntT>;
  auto nnz = indices.dims()[1];
  DenseTensor offsets = GetOffsets<IntT, Context>(dev_ctx, indices, sizes, dim);
  auto offsets_ptr = offsets.data<IntT>();

  phi::DenseTensor sorted_indices = phi::Empty<IntT>(dev_ctx, {nnz});
  thrust_ptr sorted_indices_thrust_ptr(sorted_indices.data<IntT>());
  thrust::sequence(
      policy, sorted_indices_thrust_ptr, sorted_indices_thrust_ptr + nnz, 0);

  /* sort indices corresponding to offsets */
  thrust::sort(policy,
               sorted_indices_thrust_ptr,
               sorted_indices_thrust_ptr + nnz,
               [offsets_ptr] __device__(IntT x, IntT y) {
                 return offsets_ptr[x] < offsets_ptr[y];
               });

  DenseTensor pool_sizes = phi::Empty<IntT>(dev_ctx, {nnz});

  /* reduce the elements which are groupped by pool index,
  returns all the pool indexes with unique offset value for each. */
  auto new_end =
      thrust::reduce_by_key(policy,
                            sorted_indices_thrust_ptr,
                            sorted_indices_thrust_ptr + nnz,
                            thrust::make_constant_iterator(IntT(1)),
                            thrust::make_discard_iterator(),
                            thrust_ptr(pool_sizes.data<IntT>()),
                            [offsets_ptr] __device__(IntT x, IntT y) {
                              return offsets_ptr[x] == offsets_ptr[y];
                            });
  auto new_sz =
      thrust::distance(thrust_ptr(pool_sizes.data<IntT>()), new_end.second);
  pool_sizes.Resize(phi::make_ddim({new_sz}));

  DenseTensor pool_offsets;
  pool_offsets.Resize(phi::make_ddim({new_sz}));
  dev_ctx.template Alloc<T>(&pool_offsets);
  phi::Copy(dev_ctx, pool_sizes, dev_ctx.GetPlace(), false, &pool_offsets);

  /* accumulate value for each pool index */
  thrust_ptr pool_offsets_thrust_ptr(pool_offsets.data<IntT>());
  thrust::exclusive_scan(policy,
                         pool_offsets_thrust_ptr,
                         pool_offsets_thrust_ptr + new_sz,
                         pool_offsets_thrust_ptr);

  DenseTensor mx_buffer;
  if (requireMxRows) {
    mx_buffer = phi::Full<T>(
        dev_ctx, {new_sz * nvalues}, -std::numeric_limits<T>::infinity());
    auto mx_buffer_ptr = mx_buffer.data<T>();

    auto pool_sizes_ptr = pool_sizes.data<IntT>();
    auto sorted_indices_ptr = sorted_indices.data<IntT>();
    auto pool_offsets_ptr = pool_offsets.data<IntT>();
    auto values_ptr = values.data<T>();

    /* calculate max value in each pool. */
    thrust::for_each(policy,
                     thrust::make_counting_iterator(IntT(0)),
                     thrust::make_counting_iterator(IntT(new_sz)),
                     [sorted_indices_ptr,
                      pool_sizes_ptr,
                      pool_offsets_ptr,
                      mx_buffer_ptr,
                      values_ptr,
                      nvalues] __device__(IntT index) {
                       IntT curr_pool_size = pool_sizes_ptr[index];
                       auto mx_row = mx_buffer_ptr + index * nvalues;
                       IntT offset = pool_offsets_ptr[index];
                       for (IntT p = 0; p < curr_pool_size; p++) {
                         IntT i = *(sorted_indices_ptr + offset + p);
                         for (IntT j = 0; j < nvalues; j++) {
                           auto value_tmp = *(values_ptr);
                           mx_row[j] = std::max(mx_row[j], value_tmp);
                         }
                       }
                     });
  }
  return std::make_tuple(sorted_indices, pool_offsets, pool_sizes, mx_buffer);
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
      ComputePoolMax<T, IntT, Context, true>(
          dev_ctx, indices, values_2, sizes, nvalues, static_cast<IntT>(dim));

  auto pool_size = pool_offsets.dims()[0];
  int block_size = GetNumThreads(pool_size);
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
