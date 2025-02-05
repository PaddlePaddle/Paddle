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

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
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
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/softmax.cu.h"
#include "paddle/phi/kernels/softmax_grad_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

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
                    common::errors::Unimplemented(
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

template <typename T, typename IntT>
__global__ void SoftmaxCooGradGPURawKernel(IntT* sorted_pool_indices,
                                           IntT size,
                                           IntT* pool_sizes,
                                           IntT* pool_offsets,
                                           IntT nvalues,
                                           IntT grad_nnz,
                                           IntT* grad_offsets,
                                           IntT* out_offsets,
                                           IntT* lower_bound_values,
                                           T* values,
                                           T* out_values,
                                           T* grad_values,
                                           int total_rows) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= total_rows) return;

  int tid = threadIdx.x;
  int index = row / nvalues;
  int nval = row % nvalues;
  IntT offset = pool_offsets[index];
  IntT* pool_indices = sorted_pool_indices + offset;
  IntT pool_indices_size = pool_sizes[index];

  int kIteration = (pool_indices_size + warpSize - 1) / warpSize;
  T mul_result = 0;
  for (int k = 0; k < kIteration; ++k) {
    int idx = tid + k * warpSize;
    if (idx >= pool_indices_size) break;

    auto i = pool_indices[idx];
    auto cur_out_value = out_values + i * nvalues;
    auto j = lower_bound_values[i];
    if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
      auto cur_grad_value = grad_values + j * nvalues;
      mul_result += (*(cur_out_value + nval)) * (*(cur_grad_value + nval));
    }
  }
  T sum = phi::funcs::WarpReduceSum<T>(mul_result, 0xFFFFFFFF);

  for (int k = 0; k < kIteration; ++k) {
    int idx = tid + k * warpSize;
    if (idx >= pool_indices_size) break;

    auto i = pool_indices[idx];
    auto j = lower_bound_values[i];
    auto cur_out_value = out_values + i * nvalues;
    auto cur_value = values + i * nvalues;
    auto cur_grad_value = grad_values + j * nvalues;
    if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
      cur_value[nval] =
          (*(cur_out_value + nval)) * (*(cur_grad_value + nval) - sum);
    } else {
      cur_value[nval] = -(*(cur_out_value + nval)) * sum;
    }
  }
}

template <typename T, typename IntT, typename Context>
void SoftmaxCooGradGPUKernel(const Context& dev_ctx,
                             const SparseCooTensor& out,
                             const SparseCooTensor& dout,
                             int axis,
                             SparseCooTensor* dx) {
  using thrust_ptr = thrust::device_ptr<IntT>;
  auto out_indices = out.indices();
  auto out_values = out.values();
  auto out_values_ptr = out_values.data<T>();
  const auto output_indices_dims = out.indices().dims();
  const auto out_dims = out.dims();
  auto sparse_dim = out.sparse_dim();
  auto sizes = common::vectorize<IntT>(out_dims);
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

  DenseTensor out_offsets = phi::funcs::sparse::GetOffsets<IntT, Context>(
      dev_ctx, out_indices, sizes, static_cast<IntT>(-1));
  auto out_offsets_ptr = out_offsets.data<IntT>();
  DenseTensor grad_offsets = phi::funcs::sparse::GetOffsets<IntT, Context>(
      dev_ctx, grad_indices, sizes, static_cast<IntT>(-1));
  auto grad_offsets_ptr = grad_offsets.data<IntT>();

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
    PADDLE_ENFORCE_EQ(
        is_same_offset,
        true,
        common::errors::Unimplemented(
            "SparseCooTensor only support same offsets for softmax."));
    SoftmaxGradKernel<T, Context>(
        dev_ctx, out_values, grad_values, dim - sparse_dim + 1, values);
    return;
  }

  auto nnz = out.nnz();
  IntT nvalues = std::accumulate(sizes.begin() + sparse_dim,
                                 sizes.end(),
                                 static_cast<IntT>(1),
                                 std::multiplies<>());

  DenseTensor values_2(*values);
  values_2.Resize(common::make_ddim({nnz, nvalues}));

  DenseTensor sorted_indices;
  DenseTensor pool_offsets;
  DenseTensor pool_sizes;

  std::tie(sorted_indices, pool_offsets, pool_sizes, std::ignore) =
      phi::funcs::sparse::ComputePoolMax<T, IntT, Context, false>(
          dev_ctx, out_indices, values_2, sizes, nvalues, dim);

  DenseTensor bound =
      phi::Empty<IntT>(dev_ctx, {static_cast<IntT>(out_offsets.dims()[0])});
  IntT* bound_ptr = bound.data<IntT>();
  thrust::lower_bound(policy,
                      thrust_ptr(grad_offsets_ptr),
                      thrust_ptr(grad_offsets_ptr + grad_offsets.dims()[0]),
                      thrust_ptr(out_offsets_ptr),
                      thrust_ptr(out_offsets_ptr) + out_offsets.dims()[0],
                      thrust_ptr(bound.data<IntT>()));

  auto pool_size = pool_offsets.dims()[0];
  int total_rows = pool_size * nvalues;
  dim3 grid((total_rows + 15) / 16);
  dim3 block(32, 16);
  SoftmaxCooGradGPURawKernel<T, IntT>
      <<<grid, block, 0, stream>>>(sorted_indices.data<IntT>(),
                                   pool_size,
                                   pool_sizes.data<IntT>(),
                                   pool_offsets.data<IntT>(),
                                   nvalues,
                                   grad_nnz,
                                   grad_offsets.data<IntT>(),
                                   out_offsets.data<IntT>(),
                                   bound_ptr,
                                   values_2.data<T>(),
                                   out_values.data<T>(),
                                   grad_values.data<T>(),
                                   total_rows);
}

template <typename T, typename Context>
void SoftmaxCooGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& out,
                          const SparseCooTensor& dout,
                          int axis,
                          SparseCooTensor* dx) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      out.indices().dtype(), "SoftmaxCooGradGPUKernel", ([&] {
        SoftmaxCooGradGPUKernel<T, data_t, Context>(
            dev_ctx, out, dout, axis, dx);
      }));
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
