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

#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {
namespace sparse {

template <typename T>
__global__ void GetCooInputGradCudaKernel(const int64_t* out_grad_indices_data,
                                          const T* out_grad_values_data,
                                          const int64_t* axes,
                                          const int64_t* starts,
                                          const int64_t axes_size,
                                          const int64_t sparse_dim,
                                          const int64_t out_grad_nnz,
                                          int64_t* dx_indices_data,
                                          T* dx_values_data) {
  CUDA_KERNEL_LOOP_TYPE(j, out_grad_nnz, int64_t) {
    // set indices
    for (int64_t i = 0; i < sparse_dim; ++i) {
      dx_indices_data[i * out_grad_nnz + j] =
          out_grad_indices_data[i * out_grad_nnz + j];
    }
    for (size_t ii = 0; ii < axes_size; ++ii) {
      int64_t i = axes[ii];
      dx_indices_data[i * out_grad_nnz + j] += starts[ii];
    }
    // set value
    dx_values_data[j] = out_grad_values_data[j];
  }
}
template <typename T, typename Context>
void SliceCooGradCompute(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const SparseCooTensor& out_grad,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& starts,
                         const std::vector<int64_t>& ends,
                         SparseCooTensor* x_grad) {
  const phi::DDim& x_dims = x.dims();

  // copy axes to device
  auto d_axes_tensor = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int64_t) * axes.size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int64_t* d_axes = reinterpret_cast<int64_t*>(d_axes_tensor->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     d_axes,
                     phi::CPUPlace(),
                     axes.data(),
                     sizeof(int64_t) * axes.size(),
                     dev_ctx.stream());

  // copy starts to device
  auto d_starts_tensor = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int64_t) * starts.size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int64_t* d_starts = reinterpret_cast<int64_t*>(d_starts_tensor->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     d_starts,
                     phi::CPUPlace(),
                     starts.data(),
                     sizeof(int64_t) * starts.size(),
                     dev_ctx.stream());

  // Step2: Set indices and values of x_grad
  const int64_t out_grad_nnz = out_grad.nnz();
  auto sparse_dim = static_cast<int64_t>(out_grad.sparse_dim());
  DenseTensor dx_indices =
      phi::Empty<int64_t, Context>(dev_ctx, {sparse_dim, out_grad_nnz});
  DenseTensor dx_values = phi::Empty<T, Context>(dev_ctx, {out_grad_nnz});
  auto* dx_indices_data = dx_indices.data<int64_t>();
  auto* dx_values_data = dx_values.data<T>();

  const auto* out_grad_indices_data = out_grad.indices().data<int64_t>();
  const auto* out_grad_values_data = out_grad.values().data<T>();

  x_grad->SetMember(dx_indices, dx_values, x.dims(), x.coalesced());

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_grad_nnz + 1, 1);
  GetCooInputGradCudaKernel<T><<<config.block_per_grid.x,
                                 config.thread_per_block.x,
                                 0,
                                 dev_ctx.stream()>>>(out_grad_indices_data,
                                                     out_grad_values_data,
                                                     d_axes,
                                                     d_starts,
                                                     axes.size(),
                                                     sparse_dim,
                                                     out_grad_nnz,
                                                     dx_indices_data,
                                                     dx_values_data);
}

template <typename T, typename Context>
void SliceCooGradKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const SparseCooTensor& out_grad,
                        const phi::IntArray& axes,
                        const phi::IntArray& starts,
                        const phi::IntArray& ends,
                        SparseCooTensor* x_grad) {
  const phi::DDim& x_dims = x.dims();
  std::vector<int64_t> axes_vec = axes.GetData();
  std::vector<int64_t> starts_vec = starts.GetData();
  std::vector<int64_t> ends_vec = ends.GetData();
  // Check and update sparse slice attrs
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(
      x_dims, &axes_vec, &starts_vec, &ends_vec);

  SliceCooGradCompute<T, Context>(
      dev_ctx, x, out_grad, axes_vec, starts_vec, ends_vec, x_grad);
}

template <typename T>
__global__ void GetCsrInputColsValuesCudaKernel(
    const int64_t* out_grad_cols_data,
    const T* out_grad_values_data,
    const int64_t out_grad_nnz,
    const int64_t cols_start,
    int64_t* dx_cols_data,
    T* dx_values_data) {
  CUDA_KERNEL_LOOP_TYPE(i, out_grad_nnz, int64_t) {
    dx_cols_data[i] = out_grad_cols_data[i] + cols_start;
    dx_values_data[i] = out_grad_values_data[i];
  }
}

__global__ void GetCsrInputCrowsCudaKernel(
    const int64_t* out_grad_crows_data,
    const int64_t out_grad_n_rows,
    const int64_t out_grad_nnz,
    const int64_t x_n_rows,
    const int64_t rows_start,
    const int64_t rows_end,
    int64_t* dx_crows_data,
    const int64_t dx_crows_offset = 0,
    const int64_t out_grad_crows_offset = 0) {
  CUDA_KERNEL_LOOP_TYPE(i, x_n_rows + 1, int64_t) {
    int64_t idx = i + dx_crows_offset;
    if (i < rows_start) {
      dx_crows_data[idx] = 0;
    } else if (i < rows_start + out_grad_n_rows + 1) {
      int64_t out_grad_idx = out_grad_crows_offset + (i - rows_start);
      dx_crows_data[idx] = out_grad_crows_data[out_grad_idx];
    } else {
      int64_t out_grad_idx = out_grad_crows_offset + out_grad_n_rows;
      dx_crows_data[idx] = out_grad_crows_data[out_grad_idx];
    }
  }
}

template <typename T, typename Context>
void SliceCsrGrad2D(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    const SparseCsrTensor& out_grad,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends,
                    SparseCsrTensor* x_grad) {
  const int64_t out_grad_nnz = out_grad.nnz();
  const int64_t n_rows = x.dims()[0];
  const auto* out_grad_crows_data = out_grad.crows().data<int64_t>();
  const auto* out_grad_cols_data = out_grad.cols().data<int64_t>();
  const auto* out_grad_values_data = out_grad.values().data<T>();

  DenseTensor dx_crows = phi::Empty<int64_t>(dev_ctx, {n_rows + 1});
  DenseTensor dx_cols = phi::Empty<int64_t>(dev_ctx, {out_grad_nnz});
  DenseTensor dx_values = phi::Empty<T, Context>(dev_ctx, {out_grad_nnz});
  auto* dx_crows_data = dx_crows.data<int64_t>();
  auto* dx_cols_data = dx_cols.data<int64_t>();
  auto* dx_values_data = dx_values.data<T>();
  x_grad->SetMember(dx_crows, dx_cols, dx_values, x.dims());

  // set cols and values
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_grad_nnz + 1, 1);
  GetCsrInputColsValuesCudaKernel<T><<<config.block_per_grid.x,
                                       config.thread_per_block.x,
                                       0,
                                       dev_ctx.stream()>>>(out_grad_cols_data,
                                                           out_grad_values_data,
                                                           out_grad_nnz,
                                                           starts[1],
                                                           dx_cols_data,
                                                           dx_values_data);
  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n_rows + 1, 1);
  GetCsrInputCrowsCudaKernel<<<config.block_per_grid.x,
                               config.thread_per_block.x,
                               0,
                               dev_ctx.stream()>>>(out_grad_crows_data,
                                                   out_grad.dims()[0],
                                                   out_grad_nnz,
                                                   x.dims()[0],
                                                   starts[0],
                                                   ends[0],
                                                   dx_crows_data,
                                                   0,
                                                   0);
}

__global__ void GetCsrInputCrowsPart1CudaKernl(const int64_t n_rows,
                                               const int64_t dim0_idx,
                                               int64_t* dx_crows_data) {
  CUDA_KERNEL_LOOP_TYPE(j, n_rows + 1, int64_t) {
    dx_crows_data[dim0_idx * (n_rows + 1) + j] = 0;
  }
}

template <typename T, typename Context>
void SliceCsrGrad3D(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    const SparseCsrTensor& out_grad,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends,
                    SparseCsrTensor* x_grad) {
  const int64_t dim0 = x.dims()[0], n_rows = x.dims()[1];
  const int64_t out_grad_nnz = out_grad.nnz();
  const auto* out_grad_crows_data = out_grad.crows().data<int64_t>();
  const auto* out_grad_cols_data = out_grad.cols().data<int64_t>();
  const auto* out_grad_values_data = out_grad.values().data<T>();

  DenseTensor dx_crows = phi::Empty<int64_t>(dev_ctx, {dim0 * (n_rows + 1)});
  DenseTensor dx_cols = phi::Empty<int64_t>(dev_ctx, {out_grad_nnz});
  DenseTensor dx_values = phi::Empty<T, Context>(dev_ctx, {out_grad_nnz});
  auto* dx_crows_data = dx_crows.data<int64_t>();
  auto* dx_cols_data = dx_cols.data<int64_t>();
  auto* dx_values_data = dx_values.data<T>();
  x_grad->SetMember(dx_crows, dx_cols, dx_values, x.dims());

  // set cols and values
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_grad_nnz + 1, 1);
  GetCsrInputColsValuesCudaKernel<T><<<config.block_per_grid.x,
                                       config.thread_per_block.x,
                                       0,
                                       dev_ctx.stream()>>>(out_grad_cols_data,
                                                           out_grad_values_data,
                                                           out_grad_nnz,
                                                           starts[2],
                                                           dx_cols_data,
                                                           dx_values_data);
  // set crows
  int64_t out_grad_n_rows = out_grad.dims()[1];
  for (int64_t i = 0; i < dim0; ++i) {
    if (i < starts[0] || i >= ends[0]) {
      config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n_rows + 1, 1);
      GetCsrInputCrowsPart1CudaKernl<<<config.block_per_grid.x,
                                       config.thread_per_block.x,
                                       0,
                                       dev_ctx.stream()>>>(
          n_rows, i, dx_crows_data);
    } else {
      int64_t dx_crows_offset = i * (n_rows + 1);
      int64_t out_grad_crows_offset = (i - starts[0]) * (out_grad_n_rows + 1);
      config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n_rows + 1, 1);
      GetCsrInputCrowsCudaKernel<<<config.block_per_grid.x,
                                   config.thread_per_block.x,
                                   0,
                                   dev_ctx.stream()>>>(out_grad_crows_data,
                                                       out_grad_n_rows,
                                                       out_grad_nnz,
                                                       n_rows,
                                                       starts[1],
                                                       ends[1],
                                                       dx_crows_data,
                                                       dx_crows_offset,
                                                       out_grad_crows_offset);
    }
  }
}

template <typename T, typename Context>
void SliceCsrGradCompute(const Context& dev_ctx,
                         const SparseCsrTensor& x,
                         const SparseCsrTensor& out_grad,
                         const std::vector<int64_t>& axes,
                         const std::vector<int64_t>& starts,
                         const std::vector<int64_t>& ends,
                         SparseCsrTensor* x_grad) {
  const phi::DDim& x_dims = x.dims();

  // construct new axes, starts, and ends
  std::vector<int64_t> new_axes(3), new_starts(3), new_ends(3);
  funcs::ConstructNewSliceAttrs(
      x_dims, axes, starts, ends, &new_axes, &new_starts, &new_ends);

  const int64_t sparse_dim = x_dims.size();
  if (sparse_dim == 2) {
    SliceCsrGrad2D<T, Context>(
        dev_ctx, x, out_grad, new_axes, new_starts, new_ends, x_grad);
  } else if (sparse_dim == 3) {
    SliceCsrGrad3D<T, Context>(
        dev_ctx, x, out_grad, new_axes, new_starts, new_ends, x_grad);
  } else {
    // throw exception
    common::errors::InvalidArgument(
        "Slice grad for Sparse CSR Tensor only support 2-D or 3-D, but got "
        "%d-D.",
        x_dims.size());
  }
}

template <typename T, typename Context>
void SliceCsrGradKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const SparseCsrTensor& out_grad,
                        const phi::IntArray& axes,
                        const phi::IntArray& starts,
                        const phi::IntArray& ends,
                        SparseCsrTensor* x_grad) {
  const phi::DDim& x_dims = x.dims();
  std::vector<int64_t> axes_vec = axes.GetData();
  std::vector<int64_t> starts_vec = starts.GetData();
  std::vector<int64_t> ends_vec = ends.GetData();
  // update starts and ends
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(
      x_dims, &axes_vec, &starts_vec, &ends_vec);

  SliceCsrGradCompute<T, Context>(
      dev_ctx, x, out_grad, axes_vec, starts_vec, ends_vec, x_grad);
}

}  // namespace sparse
}  // namespace phi
PD_REGISTER_KERNEL(slice_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SliceCooGradKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(slice_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SliceCsrGradKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
