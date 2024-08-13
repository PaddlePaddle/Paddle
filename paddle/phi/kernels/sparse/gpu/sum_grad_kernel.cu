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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"

namespace phi {
namespace sparse {

template <typename T>
__global__ void SetValueCudaKernel(const T* value,
                                   const int64_t length,
                                   T* data) {
  CUDA_KERNEL_LOOP_TYPE(index, length, int64_t) { data[index] = value[0]; }
}

template <typename T>
__global__ void SumCsr2DGradCudaKernel(const int64_t* x_crows_data,
                                       const T* dout_values_data,
                                       const int64_t x_dim0,
                                       T* dx_values_data) {
  // dout_crows_data[index] should be equal to index;
  CUDA_KERNEL_LOOP_TYPE(index, x_dim0, int64_t) {
    T value = dout_values_data[index];
    for (auto i = x_crows_data[index]; i < x_crows_data[index + 1]; ++i) {
      dx_values_data[i] = value;
    }
  }
}

template <typename T>
__global__ void SumCsr3DGradCudaKernel(const int64_t* x_crows_data,
                                       const T* dout_values_data,
                                       const int64_t x_dim0,
                                       const int64_t x_dim1,
                                       T* dx_values_data) {
  // dout_crows_data[index] should be equal to number;
  CUDA_KERNEL_LOOP_TYPE(index, x_dim0 * (x_dim1 + 1) - 1, int64_t) {
    int64_t batch = index / (x_dim1 + 1);
    int64_t number = index % (x_dim1 + 1);

    // compute offset of dx_values_data in every batch
    int64_t batch_offset = 0;
    for (int64_t b = 1; b <= batch; ++b) {
      batch_offset += x_crows_data[b * (x_dim1 + 1) - 1];
    }

    T value = dout_values_data[index - batch];
    for (auto i = x_crows_data[index]; i < x_crows_data[index + 1]; ++i) {
      dx_values_data[i + batch_offset] = value;
    }
  }
}

template <typename T, typename IntT, typename Context>
void SumCooGradGPUKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const SparseCooTensor& dout,
                         const IntArray& axis,
                         bool keep_dim,
                         SparseCooTensor* dx) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, dx);
  unsigned int n_dim = axis.size();

  const DenseTensor& x_indices = x.indices();
  const DenseTensor& dout_indices = dout.indices();
  const DenseTensor& dout_values = dout.values();
  const auto* dout_indices_data = dout_indices.data<IntT>();
  const auto* dout_values_data = dout_values.data<T>();

  DenseTensor* dx_indices = dx->mutable_indices();
  DenseTensor* dx_values = dx->mutable_values();
  *dx_indices = x_indices;

  const auto* dx_indices_data = dx_indices->data<IntT>();
  auto* dx_values_data = dx_values->data<T>();

  if (n_dim == 0) {
    auto length = dx->nnz();
    for (auto i = 1; i < x.values().dims().size(); ++i) {
      length *= x.values().dims()[i];
    }
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, length, 1);

    SetValueCudaKernel<T>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(dout_values_data, length, dx_values_data);

    if (dx_values->dtype() != dx->dtype()) {
      *dx_values = phi::Cast<T, Context>(dev_ctx, *dx_values, dx->dtype());
    }
    return;
  }

  auto dim = axis[0] < 0 ? x.dims().size() + axis[0] : axis[0];
  auto sparse_dim = x.sparse_dim();
  if (dim >= sparse_dim) {
    dim = dim - sparse_dim + 1;
    phi::ReduceSumGradKernel<T, Context>(
        dev_ctx, x.values(), dout.values(), {dim}, keep_dim, false, dx_values);
  } else {
    *dx_values = dout_values;
  }
  if (dx_values->dtype() != dx->dtype()) {
    *dx_values = phi::Cast<T, Context>(dev_ctx, *dx_values, dx->dtype());
  }
}

template <typename T, typename Context>
void SumCsrGradKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const SparseCsrTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCsrTensor* dx) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, dx);
  size_t n_dim = axis.size();

  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_cols = x.cols();
  const DenseTensor& dout_values = dout.values();

  DenseTensor* dx_crows = dx->mutable_crows();
  DenseTensor* dx_cols = dx->mutable_cols();
  DenseTensor* dx_values = dx->mutable_values();

  const auto* x_crows_data = x_crows.data<int64_t>();
  const auto* dout_values_data = dout_values.data<T>();
  auto* dx_values_data = dx_values->data<T>();

  *dx_crows = x_crows;
  *dx_cols = x_cols;

  if (n_dim == 0) {
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, dx->nnz(), 1);
    SetValueCudaKernel<T>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(dout_values_data, dx->nnz(), dx_values_data);

    if (dx_values->dtype() != dx->dtype()) {
      *dx_values = phi::Cast<T, Context>(dev_ctx, *dx_values, dx->dtype());
    }
    return;
  }
  PADDLE_ENFORCE_EQ(axis[0],
                    -1,
                    common::errors::Unimplemented(
                        "`axis` of SumCsrKernel only support None or -1 now."
                        "More number will be supported in the future."));
  if (x.dims().size() == 2) {
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x.dims()[0], 1);
    SumCsr2DGradCudaKernel<T><<<config.block_per_grid.x,
                                config.thread_per_block.x,
                                0,
                                dev_ctx.stream()>>>(
        x_crows_data, dout_values_data, x.dims()[0], dx_values_data);
  } else {
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, x.dims()[0] * (x.dims()[1] + 1), 1);
    SumCsr3DGradCudaKernel<T><<<config.block_per_grid.x,
                                config.thread_per_block.x,
                                0,
                                dev_ctx.stream()>>>(x_crows_data,
                                                    dout_values_data,
                                                    x.dims()[0],
                                                    x.dims()[1],
                                                    dx_values_data);
  }
  if (dx_values->dtype() != dx->dtype()) {
    *dx_values = phi::Cast<T, Context>(dev_ctx, *dx_values, dx->dtype());
  }
}

template <typename T, typename Context>
void SumCooGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const SparseCooTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCooTensor* dx) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "SumCooGradGPUKernel", ([&] {
        SumCooGradGPUKernel<T, data_t, Context>(
            dev_ctx, x, dout, axis, keep_dim, dx);
      }));
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sum_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(sum_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
