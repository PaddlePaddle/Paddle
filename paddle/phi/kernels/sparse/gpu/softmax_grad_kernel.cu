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

#include "paddle/phi/kernels/sparse/softmax_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/softmax_grad_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace phi {
namespace sparse {

template <typename T, typename IntT = int>
__global__ void SoftmaxGradGpuKernel(const IntT* out_crows,
                                     const T* out_values,
                                     const T* dout_values,
                                     T* dx_values,
                                     int row_number,
                                     int total_row_number) {
  // dx = (dout - sum(dout * out)) * out
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  int non_zero_idx = threadIdx.x;
  if (row >= total_row_number) return;
  int cur_batch = row / row_number;
  int crow_idx = cur_batch * (row_number + 1) + (row % row_number);
  int cur_batch_offset = 0;
  for (int i = 1; i < cur_batch + 1; ++i) {
    cur_batch_offset += out_crows[i * (row_number + 1) - 1];
  }
  int row_first = cur_batch_offset + static_cast<int>(out_crows[crow_idx]);
  int row_nnz = static_cast<int>(out_crows[crow_idx + 1] - out_crows[crow_idx]);
  if (row_nnz == 0) return;

  int kIteration = (row_nnz + warpSize - 1) / warpSize;

  T mul_result = 0;
  for (int i = 0; i < kIteration; ++i) {
    int idx = non_zero_idx + i * warpSize;
    if (idx >= row_nnz) break;

    mul_result += out_values[row_first + idx] * dout_values[row_first + idx];
  }
  T sum = phi::funcs::WarpReduceSum<T>(mul_result, 0xFFFFFFFF);

  for (int i = 0; i < kIteration; ++i) {
    int idx = non_zero_idx + i * warpSize;
    if (idx >= row_nnz) break;

    dx_values[row_first + idx] =
        (dout_values[row_first + idx] - sum) * out_values[row_first + idx];
  }
}

template <typename T, typename Context>
void SoftmaxCsrGradKernel(const Context& dev_ctx,
                          const SparseCsrTensor& out,
                          const SparseCsrTensor& dout,
                          int axis,
                          SparseCsrTensor* dx) {
  PADDLE_ENFORCE_EQ(axis,
                    -1,
                    phi::errors::Unimplemented(
                        "SparseCsrTensor only support axis=-1 for softmax, "
                        "which is faster when reading data by row (axis=-1)"));
  EmptyLikeCsrKernel<T, Context>(dev_ctx, dout, dx);

  auto out_dim = out.dims();
  auto out_rank = out_dim.size();

  int total_row_number = 1;
  int row_number = 1;
  for (int i = 0; i < out_rank - 1; ++i) {
    total_row_number *= out_dim[i];
    if (i == out_rank - 2) {
      row_number = out_dim[i];
    }
  }

  dim3 grid((total_row_number + 3) / 4);
  dim3 block(32, 4);

  PD_VISIT_BASE_INTEGRAL_TYPES(
      out.crows().dtype(), "SoftmaxCsrGradKernel", ([&] {
        SoftmaxGradGpuKernel<T, data_t><<<grid, block, 0, dev_ctx.stream()>>>(
            out.crows().data<data_t>(),
            out.values().data<T>(),
            dout.values().data<T>(),
            dx->mutable_values()->data<T>(),
            row_number,
            total_row_number);
      }));
}

//============================================================================

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
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

using thrust_ptr = thrust::device_ptr<int64_t>;

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
__global__ void SoftmaxCooGradGpuKernel(int64_t* sorted_pool_indices,
                                        int64_t size,
                                        int64_t* pool_sizes,
                                        int64_t* pool_offsets,
                                        int64_t nvalues,
                                        int64_t grad_nnz,
                                        int64_t* grad_offsets,
                                        int64_t* out_offsets,
                                        int64_t* lower_bound_values,
                                        T* values,
                                        T* out_values,
                                        T* grad_values) {
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

    for (int64_t k = 0; k < nvalues; k++) {
      T tmp_row{0};

      /* Compute tmp = - sum_j output_j * grad_j */
      for (int64_t p = 0; p < pool_indices_size; p++) {
        auto i = pool_indices[p];
        auto cur_out_value = out_values + i * nvalues;
        auto j = lower_bound_values[i];

        /* Update `tmp_row` accumulator only when limits and pools are valid */
        if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
          auto cur_grad_value = grad_values + j * nvalues;
          tmp_row -= (*(cur_out_value + k)) * (*(cur_grad_value + k));
        }
      }

      /* Compute grad_input = output * (grad + tmp)*/
      for (int64_t p = 0; p < pool_indices_size; p++) {
        auto i = pool_indices[p];
        auto cur_out_value = out_values + i * nvalues;
        auto cur_value = values + i * nvalues;
        auto j = lower_bound_values[i];
        if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
          auto cur_grad_value = grad_values + j * nvalues;
          cur_value[k] =
              (*(cur_out_value + k)) * (*(cur_grad_value + k) + tmp_row);
        } else {
          cur_value[k] = (*(cur_out_value + k)) * tmp_row;
        }
      }
    }
    index += step;
  }
}

template <typename T, typename Context>
void SoftmaxCooGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& out,
                          const SparseCooTensor& dout,
                          int axis,
                          SparseCooTensor* dx) {
  auto out_indices = out.indices();
  auto out_values = out.values();
  auto out_values_ptr = out_values.data<T>();
  const auto output_indices_dims = out.indices().dims();
  const auto out_dims = out.dims();
  auto sparse_dim = out.sparse_dim();
  auto sizes = phi::vectorize(out_dims);
  auto grad_indices = dout.indices();
  auto grad_values = dout.values();
  auto grad_values_ptr = grad_values.data<T>();
  auto out_nnz = out.nnz();
  auto grad_nnz = dout.nnz();
  auto place = dev_ctx.GetPlace();
  auto stream = dev_ctx.stream();

  *(dx->mutable_indices()) = out_indices;
  DenseTensor* values = dx->mutable_values();
  values->Resize(out_dims);
  values->set_meta(out_values.meta());
  dev_ctx.template Alloc<T>(values);
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  set_zero(dev_ctx, values, static_cast<T>(0.0f));

  DenseTensor out_offsets = GetOffsets(dev_ctx, out_indices, sizes, -1);
  auto out_offsets_ptr = out_offsets.data<int64_t>();
  DenseTensor grad_offsets = GetOffsets(dev_ctx, grad_indices, sizes, -1);
  auto grad_offsets_ptr = grad_offsets.data<int64_t>();

#ifdef PADDLE_WITH_HIP
  const auto& policy = thrust::hip::par.on(dev_ctx.stream());
  bool is_same_offset = thrust::equal(thrust::hip::par.on(dev_ctx.stream()),
#else
  const auto& policy = thrust::cuda::par.on(dev_ctx.stream());
  bool is_same_offset = thrust::equal(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                                      out_offsets_ptr,
                                      out_offsets_ptr + out_offsets.numel(),
                                      grad_offsets_ptr);

  int dim = axis < 0 ? out_dims.size() + axis : axis;
  if (dim >= sparse_dim) {
    if (is_same_offset) {
      SoftmaxGradKernel<T, Context>(
          dev_ctx, out_values, grad_values, dim - sparse_dim + 1, values);
    } else {
      DenseTensor cur_out_values, cur_grad_values, cur_values;
      cur_out_values.Resize(phi::make_ddim({grad_nnz}));
      dev_ctx.template Alloc<T>(&cur_out_values);
      cur_grad_values.Resize(phi::make_ddim({grad_nnz}));
      dev_ctx.template Alloc<T>(&cur_grad_values);
      cur_values.Resize(phi::make_ddim({grad_nnz}));
      dev_ctx.template Alloc<T>(&cur_values);

      for (int64_t i = 0; i < out_nnz; i++) {
        auto low =
            thrust::lower_bound(grad_offsets_ptr,
                                grad_offsets_ptr + grad_offsets.dims()[0],
                                out_offsets_ptr[i]);

        auto j = *low - (*grad_offsets_ptr);
        if (j < grad_nnz && out_offsets_ptr[i] == grad_offsets_ptr[j]) {
          memory_utils::Copy(place,
                             out_values_ptr + i * grad_nnz,
                             place,
                             cur_out_values.data<T>(),
                             grad_nnz * sizeof(T),
                             stream);

          memory_utils::Copy(place,
                             grad_values_ptr + i * grad_nnz,
                             place,
                             cur_grad_values.data<T>(),
                             grad_nnz * sizeof(T),
                             stream);

          SoftmaxGradKernel<T, Context>(dev_ctx,
                                        cur_out_values,
                                        cur_grad_values,
                                        dim - sparse_dim,
                                        &cur_values);

          memory_utils::Copy(place,
                             cur_values.data<T>(),
                             place,
                             values->data<T>() + i * grad_nnz,
                             grad_nnz * sizeof(T),
                             stream);
        }
      }
    }
    return;
  }

  auto nnz = out.nnz();
  int64_t nvalues = std::accumulate(sizes.begin() + sparse_dim,
                                    sizes.end(),
                                    static_cast<int64_t>(1),
                                    std::multiplies<>());

  DenseTensor values_2(*values);
  values_2.Resize(phi::make_ddim({nnz, nvalues}));

  DenseTensor out_values_2(out_values);
  out_values_2.Resize(phi::make_ddim({nnz, nvalues}));

  DenseTensor grad_values_2(grad_values);
  grad_values_2.Resize(phi::make_ddim({nnz, nvalues}));

  DenseTensor sorted_indices;
  DenseTensor pool_offsets;
  DenseTensor pool_sizes;

  std::tie(sorted_indices, pool_offsets, pool_sizes, std::ignore) =
      ComputePoolMax<T, Context, false>(
          dev_ctx, out_indices, values_2, sizes, nvalues, dim);

  DenseTensor bound =
      phi::Empty(dev_ctx,
                 DenseTensorMeta(DataType::INT64,
                                 {static_cast<int>(out_offsets.dims()[0])},
                                 DataLayout::NCHW));

  int64_t* bound_ptr = bound.data<int64_t>();
  thrust::lower_bound(policy,
                      thrust_ptr(grad_offsets_ptr),
                      thrust_ptr(grad_offsets_ptr + grad_offsets.dims()[0]),
                      thrust_ptr(out_offsets_ptr),
                      thrust_ptr(out_offsets_ptr) + out_offsets.dims()[0],
                      thrust_ptr(bound.data<int64_t>()));

  auto pool_size = pool_offsets.dims()[0];
  int block_size = GetNumThreads(pool_size);
  const int grid_size = (pool_size + block_size - 1) / block_size;
  SoftmaxCooGradGpuKernel<T>
      <<<grid_size, block_size, 0, stream>>>(sorted_indices.data<int64_t>(),
                                             pool_size,
                                             pool_sizes.data<int64_t>(),
                                             pool_offsets.data<int64_t>(),
                                             nvalues,
                                             grad_nnz,
                                             grad_offsets.data<int64_t>(),
                                             out_offsets.data<int64_t>(),
                                             bound_ptr,
                                             values_2.data<T>(),
                                             out_values_2.data<T>(),
                                             grad_values_2.data<T>());
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(softmax_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCsrGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(softmax_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCooGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
