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

template <typename Context>
DenseTensor GetOffsets(const Context& dev_ctx,
                       const DenseTensor& indices,
                       const std::vector<int64_t>& sizes,
                       const int64_t dim) {
#ifdef __HIPCC__
  const auto& policy = thrust::hip::par.on(dev_ctx.stream());
#else
  const auto& policy = thrust::cuda::par.on(dev_ctx.stream());
#endif

  auto ndim = indices.dims()[0];
  auto nnz = indices.dims()[1];
  std::vector<int64_t> host_strides(ndim, 1);
  if (ndim > 1) {
    for (int64_t i = ndim - 2; i >= 0; i--) {
      host_strides[i] = host_strides[i + 1] * (i + 1 == dim ? 1 : sizes[i + 1]);
    }
  }

  DenseTensorMeta meta(DataType::INT64, indices.dims(), indices.layout());
  DenseTensor strides = phi::Empty(dev_ctx, std::move(meta));

  auto strides_ptr = strides.data<int64_t>();

#ifdef __HIPCC__
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipMemcpyAsync(strides_ptr,
                     host_strides.data(),
                     host_strides.size() * sizeof(int64_t),
                     hipMemcpyHostToDevice,
                     dev_ctx.stream()));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpyAsync(strides_ptr,
                      host_strides.data(),
                      host_strides.size() * sizeof(int64_t),
                      cudaMemcpyHostToDevice,
                      dev_ctx.stream()));
#endif

  DenseTensorMeta offsets_meta(DataType::INT64, {nnz}, indices.layout());
  DenseTensor offsets = phi::Empty(dev_ctx, std::move(offsets_meta));
  auto indices_ptr = indices.data<int64_t>();

  thrust::transform(
      policy,
      thrust::make_counting_iterator(int64_t(0)),
      thrust::make_counting_iterator(int64_t(nnz)),
      thrust::device_ptr<int64_t>(offsets.data<int64_t>()),
      [strides_ptr, indices_ptr, nnz, dim, ndim] __device__(int64_t x) {
        int64_t pool_index = 0;
        for (int64_t j = 0; j < ndim; j++) {
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

using thrust_ptr = thrust::device_ptr<int64_t>;

template <typename T, typename Context, bool requireMxRows = true>
std::tuple<DenseTensor, DenseTensor, DenseTensor, DenseTensor> ComputePoolMax(
    const Context& dev_ctx,
    const DenseTensor& indices,
    const DenseTensor& values,
    const std::vector<int64_t>& sizes,
    int64_t nvalues,
    const int64_t dim) {
#ifdef __HIPCC__
  const auto& policy = thrust::hip::par.on(dev_ctx.stream());
#else
  const auto& policy = thrust::cuda::par.on(dev_ctx.stream());
#endif

  auto nnz = indices.dims()[1];
  DenseTensor offsets = GetOffsets<Context>(dev_ctx, indices, sizes, dim);
  auto offsets_ptr = offsets.data<int64_t>();

  DenseTensor sorted_indices = phi::Empty<GPUContext>(
      dev_ctx, DenseTensorMeta(DataType::INT64, {nnz}, DataLayout::NCHW));

  thrust_ptr sorted_indices_thrust_ptr(sorted_indices.data<int64_t>());
  thrust::sequence(
      policy, sorted_indices_thrust_ptr, sorted_indices_thrust_ptr + nnz, 0);

  thrust::sort(policy,
               sorted_indices_thrust_ptr,
               sorted_indices_thrust_ptr + nnz,
               [offsets_ptr] __device__(int64_t x, int64_t y) {
                 return offsets_ptr[x] < offsets_ptr[y];
               });

  DenseTensor pool_sizes = phi::Empty<GPUContext>(
      dev_ctx, DenseTensorMeta(DataType::INT64, {nnz}, DataLayout::NCHW));

  auto new_end =
      thrust::reduce_by_key(policy,
                            sorted_indices_thrust_ptr,
                            sorted_indices_thrust_ptr + nnz,
                            thrust::make_constant_iterator(int64_t(1)),
                            thrust::make_discard_iterator(),
                            thrust_ptr(pool_sizes.data<int64_t>()),
                            [offsets_ptr] __device__(int64_t x, int64_t y) {
                              return offsets_ptr[x] == offsets_ptr[y];
                            });
  auto new_sz =
      thrust::distance(thrust_ptr(pool_sizes.data<int64_t>()), new_end.second);
  pool_sizes.Resize(phi::make_ddim({new_sz}));

  DenseTensor pool_offsets;
  pool_offsets.Resize(phi::make_ddim({new_sz}));
  dev_ctx.template Alloc<T>(&pool_offsets);
  phi::Copy(dev_ctx, pool_sizes, dev_ctx.GetPlace(), false, &pool_offsets);

  thrust_ptr pool_offsets_thrust_ptr(pool_offsets.data<int64_t>());
  thrust::exclusive_scan(policy,
                         pool_offsets_thrust_ptr,
                         pool_offsets_thrust_ptr + new_sz,
                         pool_offsets_thrust_ptr);

  DenseTensor mx_buffer;
  if (requireMxRows) {
    mx_buffer = phi::Full<T>(
        dev_ctx, {new_sz * nvalues}, -std::numeric_limits<T>::infinity());
    auto mx_buffer_ptr = mx_buffer.data<T>();

    auto pool_sizes_ptr = pool_sizes.data<int64_t>();
    auto sorted_indices_ptr = sorted_indices.data<int64_t>();
    auto pool_offsets_ptr = pool_offsets.data<int64_t>();
    auto values_ptr = values.data<T>();
    thrust::for_each(policy,
                     thrust::make_counting_iterator(int64_t(0)),
                     thrust::make_counting_iterator(int64_t(new_sz)),
                     [sorted_indices_ptr,
                      pool_sizes_ptr,
                      pool_offsets_ptr,
                      mx_buffer_ptr,
                      values_ptr,
                      nvalues] __device__(int64_t index) {
                       int64_t curr_pool_size = pool_sizes_ptr[index];
                       auto mx_row = mx_buffer_ptr + index * nvalues;
                       int64_t offset = pool_offsets_ptr[index];
                       for (int64_t p = 0; p < curr_pool_size; p++) {
                         int64_t i = *(sorted_indices_ptr + offset + p);
                         for (int64_t j = 0; j < nvalues; j++) {
                           auto value_tmp = *(values_ptr);
                           mx_row[j] = std::max(mx_row[j], value_tmp);
                         }
                       }
                     });
  }
  return std::make_tuple(sorted_indices, pool_offsets, pool_sizes, mx_buffer);
}

template <typename T>
__global__ void SoftmaxCooGpuKernel(int64_t* sorted_pool_indices,
                                    int64_t size,
                                    int64_t* pool_sizes,
                                    int64_t* pool_offsets,
                                    int64_t nvalues,
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
    int64_t offset = pool_offsets[index];
    int64_t* pool_indices = sorted_pool_indices + offset;
    int64_t pool_indices_size = pool_sizes[index];
    T* mx_row = mx_rows + index * nvalues;

    for (int64_t j = 0; j < nvalues; j++) {
      T exp_sums = 0;
      for (int64_t p = 0; p < pool_indices_size; p++) {
        auto i = pool_indices[p];
        auto cur_value = input_values + i * nvalues + j;
        auto cur_out_value = output_values + i * nvalues + j;
        auto v = std::exp((*cur_value) - (*(mx_row + j)));
        *cur_out_value = v;
        exp_sums += v;
      }
      for (int64_t p = 0; p < pool_indices_size; p++) {
        auto i = pool_indices[p];
        auto out_values_row = output_values + i * nvalues;
        out_values_row[j] *= 1.0 / exp_sums;
      }
    }
    index += step;
  }
}

template <typename T, typename Context>
void SoftmaxCooKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      int axis,
                      SparseCooTensor* out) {
  // create out sparse DenseTensor
  auto indices = x.indices();
  auto values = x.values();
  const auto x_dims = x.dims();
  const std::vector<int64_t> sizes = phi::vectorize<int64_t>(x_dims);
  const auto sparse_dim = x.sparse_dim();
  const int64_t x_nnz = x.nnz();
  DenseTensor out_indices(indices);
  DenseTensor out_values = EmptyLike<T, Context>(dev_ctx, values);
  out->SetMember(out_indices, out_values, x.dims(), x.coalesced());

  int dim = axis < 0 ? x_dims.size() + axis : axis;
  if (dim >= sparse_dim) {
    SoftmaxKernel<T, Context>(
        dev_ctx, values, dim - sparse_dim + 1, &out_values);
    return;
  }

  auto stream = dev_ctx.stream();
  int64_t nvalues = std::accumulate(sizes.begin() + sparse_dim,
                                    sizes.end(),
                                    static_cast<int64_t>(1),
                                    std::multiplies<>());

  auto values_2 = values.Resize({x_nnz, nvalues});
  auto out_values_2 = out_values.Resize({x_nnz, nvalues});

  DenseTensor sorted_indices;
  DenseTensor pool_offsets;
  DenseTensor pool_sizes;
  DenseTensor mx_buffer;
  std::tie(sorted_indices, pool_offsets, pool_sizes, mx_buffer) =
      ComputePoolMax<T, Context, true>(
          dev_ctx, indices, values_2, sizes, nvalues, dim);

  auto pool_size = pool_offsets.dims()[0];
  int block_size = GetNumThreads(pool_size);
  const int grid_size = (pool_size + block_size - 1) / block_size;
  auto out_values_ptr = out_values.data<T>();
  auto values_ptr = values.data<T>();
  SoftmaxCooGpuKernel<T>
      <<<grid_size, block_size, 0, stream>>>(sorted_indices.data<int64_t>(),
                                             pool_size,
                                             pool_sizes.data<int64_t>(),
                                             pool_offsets.data<int64_t>(),
                                             nvalues,
                                             mx_buffer.data<T>(),
                                             values_ptr,
                                             out_values_ptr);
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
