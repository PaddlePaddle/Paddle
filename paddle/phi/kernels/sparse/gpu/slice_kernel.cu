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

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {
namespace sparse {

template <typename IntT>
__global__ void GetCooNonZeroNumberCudaKernel(const IntT* x_indices_data,
                                              const int64_t* axes,
                                              const int64_t* starts,
                                              const int64_t* ends,
                                              const int64_t axes_size,
                                              const int64_t x_nnz,
                                              int* out_nnz,
                                              IntT* out_nnz_indices) {
  CUDA_KERNEL_LOOP_TYPE(j, x_nnz, int64_t) {
    bool hit = true;
    for (size_t ii = 0; ii < axes_size; ++ii) {
      auto item = x_indices_data[axes[ii] * x_nnz + j];
      if (!(starts[ii] <= item && item < ends[ii])) {
        hit = false;
        break;
      }
    }
    if (!hit) continue;
    int old_value = atomicAdd(out_nnz, 1);
    out_nnz_indices[old_value] = j;
  }
}

template <typename T, typename IntT>
__global__ void GetCooOutCudaKernel(const IntT* x_indices_data,
                                    const T* x_values_data,
                                    const int64_t* axes,
                                    const int64_t* starts,
                                    const int64_t axes_size,
                                    const int64_t sparse_dim,
                                    const int64_t x_nnz,
                                    const int64_t out_nnz,
                                    const IntT* out_nnz_indices,
                                    IntT* out_indices_data,
                                    T* out_values_data) {
  CUDA_KERNEL_LOOP_TYPE(index, out_nnz, int64_t) {
    // index is in the order of the non-zero elements in out
    // out_nnz_indices[index] is the valid index in x's non-zero elements, where
    // the `hit` is true.
    IntT j = out_nnz_indices[index];
    // set value
    out_values_data[index] = x_values_data[j];
    // set coordinate
    for (int64_t i = 0; i < sparse_dim; ++i) {
      out_indices_data[i * out_nnz + index] = x_indices_data[i * x_nnz + j];
    }
    for (size_t ii = 0; ii < axes_size; ++ii) {
      auto i = axes[ii];
      out_indices_data[i * out_nnz + index] -= starts[ii];
    }
  }
}

template <typename T, typename IntT, typename Context>
void SliceCooGPUCompute(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& starts,
                        const std::vector<int64_t>& ends,
                        SparseCooTensor* out) {
  const phi::DDim& x_dims = x.dims();

  // Step1: Infer output dims
  auto out_dims = funcs::GetSliceDims<int64_t>(
      x_dims, axes, starts, ends, nullptr, nullptr);

  // Step2: Get the number of non zero elements
  DenseTensor d_out_nnz = phi::Empty<int32_t>(dev_ctx, {1});
  int* d_out_nnz_ptr = d_out_nnz.data<int32_t>();
  phi::backends::gpu::GpuMemsetAsync(
      d_out_nnz_ptr, 0, sizeof(int32_t), dev_ctx.stream());

  // out_nnz_indices is the indices where the data is valid in out
  // the length of the out_nnz_indices must be less than x.nnz()
  DenseTensor d_out_nnz_indices = phi::Empty<IntT>(dev_ctx, {x.nnz()});
  auto* d_out_nnz_indices_ptr = d_out_nnz_indices.data<IntT>();
  phi::backends::gpu::GpuMemsetAsync(
      d_out_nnz_indices_ptr, 0, sizeof(IntT), dev_ctx.stream());

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

  // copy ends to device
  auto d_ends_tensor = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int64_t) * ends.size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int64_t* d_ends = reinterpret_cast<int64_t*>(d_ends_tensor->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     d_ends,
                     phi::CPUPlace(),
                     ends.data(),
                     sizeof(int64_t) * ends.size(),
                     dev_ctx.stream());

  const auto* x_indices_data = x.indices().data<IntT>();

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x.nnz() + 1, 1);
  GetCooNonZeroNumberCudaKernel<IntT>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(x_indices_data,
                             d_axes,
                             d_starts,
                             d_ends,
                             axes.size(),
                             x.nnz(),
                             d_out_nnz_ptr,
                             d_out_nnz_indices_ptr);

  // copy d_out_nnz from device to host (out_nnz)
  int32_t out_nnz = 0;
  phi::backends::gpu::GpuMemcpyAsync(&out_nnz,
                                     d_out_nnz_ptr,
                                     sizeof(int32_t),
                                     gpuMemcpyDeviceToHost,
                                     dev_ctx.stream());
  dev_ctx.Wait();
  // sort `d_out_nnz_indices_ptr`
  d_out_nnz_indices.Resize({out_nnz});
#ifdef PADDLE_WITH_HIP
  thrust::sort(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::sort(thrust::cuda::par.on(dev_ctx.stream()),
#endif
               d_out_nnz_indices_ptr,
               d_out_nnz_indices_ptr + out_nnz);

  // Step3: Get the values and indices of output
  auto sparse_dim = static_cast<int64_t>(x.sparse_dim());
  DenseTensor out_indices =
      phi::Empty<IntT, Context>(dev_ctx, {sparse_dim, out_nnz});
  DenseTensor out_values = phi::Empty<T, Context>(dev_ctx, {out_nnz});
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());

  auto* out_indices_data = out_indices.data<IntT>();
  auto* out_values_data = out_values.data<T>();
  const auto* x_values_data = x.values().data<T>();

  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_nnz + 1, 1);
  GetCooOutCudaKernel<T, IntT>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(x_indices_data,
                             x_values_data,
                             d_axes,
                             d_starts,
                             axes.size(),
                             sparse_dim,
                             x.nnz(),
                             static_cast<int64_t>(out_nnz),
                             d_out_nnz_indices_ptr,
                             out_indices_data,
                             out_values_data);
}

template <typename T, typename IntT, typename Context>
void SliceCooGPUKernel(const Context& dev_ctx,
                       const SparseCooTensor& x,
                       const phi::IntArray& axes,
                       const phi::IntArray& starts,
                       const phi::IntArray& ends,
                       SparseCooTensor* out) {
  const phi::DDim& x_dims = x.dims();
  std::vector<int64_t> axes_vec = axes.GetData();
  std::vector<int64_t> starts_vec = starts.GetData();
  std::vector<int64_t> ends_vec = ends.GetData();
  // Check and update attr
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(
      x_dims, &axes_vec, &starts_vec, &ends_vec);
  SliceCooGPUCompute<T, IntT, Context>(
      dev_ctx, x, axes_vec, starts_vec, ends_vec, out);
}

template <typename T, typename Context>
void SliceCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const phi::IntArray& axes,
                    const phi::IntArray& starts,
                    const phi::IntArray& ends,
                    SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "SliceCooGPUKernel", ([&] {
                                 SliceCooGPUKernel<T, data_t, Context>(
                                     dev_ctx, x, axes, starts, ends, out);
                               }));
}

__global__ void GetCsr2DNonZeroNumberCudaKernel(const int64_t* x_crows_data,
                                                const int64_t* x_cols_data,
                                                const int64_t x_crows_start,
                                                const int64_t x_crows_end,
                                                const int64_t min_col,
                                                const int64_t max_col,
                                                int64_t* out_crows_data) {
  CUDA_KERNEL_LOOP_TYPE(i, x_crows_end - x_crows_start, int64_t) {
    if (i == 0) {
      out_crows_data[0] = 0;
    }
    int64_t st = x_crows_data[x_crows_start + i];
    int64_t ed = x_crows_data[x_crows_start + i + 1];
    out_crows_data[i + 1] = 0;
    for (int64_t jj = st; jj < ed; ++jj) {
      if (x_cols_data[jj] >= min_col && x_cols_data[jj] < max_col) {
        out_crows_data[i + 1] += 1;
      }
    }
  }
}

template <typename T>
__global__ void GetCsr2DCudaKernel(const int64_t* x_crows_data,
                                   const int64_t* x_cols_data,
                                   const T* x_values_data,
                                   const int64_t x_crows_start,
                                   const int64_t x_crows_end,
                                   const int64_t min_col,
                                   const int64_t max_col,
                                   const int64_t* out_crows_data,
                                   int64_t* out_cols_data,
                                   T* out_values_data) {
  CUDA_KERNEL_LOOP_TYPE(i, x_crows_end - x_crows_start, int64_t) {
    int64_t st = x_crows_data[x_crows_start + i];
    int64_t ed = x_crows_data[x_crows_start + i + 1];
    int64_t index = out_crows_data[i];
    for (int64_t jj = st; jj < ed; ++jj) {
      if (x_cols_data[jj] >= min_col && x_cols_data[jj] < max_col) {
        out_cols_data[index] = x_cols_data[jj] - min_col;
        out_values_data[index] = x_values_data[jj];
        index++;
      }
    }
  }
}

template <typename T, typename Context>
void SliceCsrTensor2D(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const std::vector<int64_t>& axes,
                      const std::vector<int64_t>& starts,
                      const std::vector<int64_t>& ends,
                      const phi::DDim& out_dims,
                      SparseCsrTensor* out) {
  const auto* x_crows_data = x.crows().data<int64_t>();
  const auto* x_cols_data = x.cols().data<int64_t>();
  const auto* x_values_data = x.values().data<T>();
  // Step1: Get the number of non zero elements for out and out_crows
  int64_t out_n_rows = ends[0] - starts[0];
  DenseTensor out_crows =
      phi::Empty<int64_t, Context>(dev_ctx, {out_n_rows + 1});
  auto* out_crows_data = out_crows.data<int64_t>();

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, ends[0] - starts[0] + 1, 1);
  GetCsr2DNonZeroNumberCudaKernel<<<config.block_per_grid.x,
                                    config.thread_per_block.x,
                                    0,
                                    dev_ctx.stream()>>>(x_crows_data,
                                                        x_cols_data,
                                                        starts[0],
                                                        ends[0],
                                                        starts[1],
                                                        ends[1],
                                                        out_crows_data);
#ifdef PADDLE_WITH_HIP
  thrust::inclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::inclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                         out_crows_data,
                         out_crows_data + out_n_rows + 1,
                         out_crows_data);
  int64_t out_nnz = 0;
  phi::backends::gpu::GpuMemcpyAsync(&out_nnz,
                                     &out_crows_data[out_n_rows],
                                     sizeof(int64_t),
                                     gpuMemcpyDeviceToHost,
                                     dev_ctx.stream());
  dev_ctx.Wait();
  // Step2: Set out
  DenseTensor out_cols = phi::Empty<int64_t, Context>(dev_ctx, {out_nnz});
  DenseTensor out_values = phi::Empty<T, Context>(dev_ctx, {out_nnz});
  out->SetMember(out_crows, out_cols, out_values, out_dims);
  config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, ends[0] - starts[0] + 1, 1);
  GetCsr2DCudaKernel<T><<<config.block_per_grid.x,
                          config.thread_per_block.x,
                          0,
                          dev_ctx.stream()>>>(x_crows_data,
                                              x_cols_data,
                                              x_values_data,
                                              starts[0],
                                              ends[0],
                                              starts[1],
                                              ends[1],
                                              out_crows.data<int64_t>(),
                                              out_cols.data<int64_t>(),
                                              out_values.data<T>());
}

__global__ void GetXColsOffsetsCudaKernel(const int64_t* x_crows_data,
                                          const int64_t x_n_rows,
                                          const int64_t x_dim0,
                                          int64_t* x_cols_offsets) {
  CUDA_KERNEL_LOOP_TYPE(i, x_dim0, int64_t) {
    if (i == 0) {
      x_cols_offsets[i] = 0;
    }
    x_cols_offsets[i + 1] = x_crows_data[(i + 1) * (x_n_rows + 1) - 1];
  }
}

__global__ void GetCsr3DNonZeroNumberCudaKernel(const int64_t* x_crows_data,
                                                const int64_t* x_cols_data,
                                                const int64_t x_dim0,
                                                const int64_t x_n_rows,
                                                const int64_t* x_cols_offsets,
                                                const int64_t* starts,
                                                const int64_t* ends,
                                                const int64_t out_n_rows,
                                                int64_t* out_crows_data) {
  CUDA_KERNEL_LOOP_TYPE(i, x_dim0 * (x_n_rows + 1), int64_t) {
    int64_t dim0_i = i / (x_n_rows + 1);
    int64_t dim1_i = i % (x_n_rows + 1);
    if (!(dim0_i >= starts[0] && dim0_i < ends[0])) {
      continue;
    }
    if (!(dim1_i >= starts[1] && dim1_i < ends[1])) {
      continue;
    }
    // the starting index of current 2D Tensor in out_crows
    int64_t out_dim0_start = (dim0_i - starts[0]) * (out_n_rows + 1);
    if (dim1_i == starts[1]) {
      out_crows_data[out_dim0_start] = 0;
    }
    int64_t out_crows_idx = out_dim0_start + (dim1_i - starts[1]);
    int64_t st = x_crows_data[i] + x_cols_offsets[dim0_i];
    int64_t ed = x_crows_data[i + 1] + x_cols_offsets[dim0_i];
    out_crows_data[out_crows_idx + 1] = 0;
    for (int64_t jj = st; jj < ed; ++jj) {
      if (x_cols_data[jj] >= starts[2] && x_cols_data[jj] < ends[2]) {
        out_crows_data[out_crows_idx + 1] += 1;
      }
    }
  }
}

template <typename T>
__global__ void GetCsr3DCudaKernel(const int64_t* x_crows_data,
                                   const int64_t* x_cols_data,
                                   const T* x_values_data,
                                   const int64_t* x_cols_offsets,
                                   const int64_t x_dim0,
                                   const int64_t x_n_rows,
                                   const int64_t* starts,
                                   const int64_t* ends,
                                   const int64_t out_n_rows,
                                   const int64_t* out_cols_offsets,
                                   const int64_t* out_crows_data,
                                   int64_t* out_cols_data,
                                   T* out_values_data) {
  CUDA_KERNEL_LOOP_TYPE(i, x_dim0 * (x_n_rows + 1), int64_t) {
    int dim0_i = i / (x_n_rows + 1);
    int dim1_i = i % (x_n_rows + 1);
    if (!(dim0_i >= starts[0] && dim0_i < ends[0])) {
      continue;
    }
    if (!(dim1_i >= starts[1] && dim1_i < ends[1])) {
      continue;
    }
    // the starting index of current 2D Tensor in out_crows
    int64_t out_dim0_start = (dim0_i - starts[0]) * (out_n_rows + 1);
    int64_t out_crows_idx = out_dim0_start + (dim1_i - starts[1]);
    int64_t st = x_crows_data[i] + x_cols_offsets[dim0_i];
    int64_t ed = x_crows_data[i + 1] + x_cols_offsets[dim0_i];
    int64_t index = out_crows_data[out_crows_idx];
    for (int64_t jj = st; jj < ed; ++jj) {
      if (x_cols_data[jj] >= starts[2] && x_cols_data[jj] < ends[2]) {
        out_cols_data[out_cols_offsets[out_dim0_start] + index] =
            x_cols_data[jj] - starts[2];
        out_values_data[out_cols_offsets[out_dim0_start] + index] =
            x_values_data[jj];
        index++;
      }
    }
  }
}

template <typename T, typename Context>
void SliceCsrTensor3D(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const std::vector<int64_t>& axes,
                      const std::vector<int64_t>& starts,
                      const std::vector<int64_t>& ends,
                      const phi::DDim& out_dims,
                      SparseCsrTensor* out) {
  const auto* x_crows_data = x.crows().data<int64_t>();
  const auto* x_cols_data = x.cols().data<int64_t>();
  const auto* x_values_data = x.values().data<T>();
  const int64_t x_dim0 = x.dims()[0], x_n_rows = x.dims()[1];

  // get x_cols_offsets
  DenseTensor x_cols_offsets = phi::Empty<int64_t>(dev_ctx, {x_dim0 + 1});
  auto* x_cols_offsets_data = x_cols_offsets.data<int64_t>();

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_dim0 + 1, 1);
  GetXColsOffsetsCudaKernel<<<config.block_per_grid.x,
                              config.thread_per_block.x,
                              0,
                              dev_ctx.stream()>>>(
      x_crows_data, x_n_rows, x_dim0, x_cols_offsets_data);

#ifdef PADDLE_WITH_HIP
  thrust::inclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::inclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                         x_cols_offsets_data,
                         x_cols_offsets_data + x_dim0 + 1,
                         x_cols_offsets_data);

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

  // copy ends to device
  auto d_ends_tensor = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int64_t) * ends.size(),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int64_t* d_ends = reinterpret_cast<int64_t*>(d_ends_tensor->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     d_ends,
                     phi::CPUPlace(),
                     ends.data(),
                     sizeof(int64_t) * ends.size(),
                     dev_ctx.stream());

  // get out_nnz
  const int64_t out_dim0 = out_dims[0], out_n_rows = out_dims[1];
  DenseTensor out_crows =
      phi::Empty<int64_t, Context>(dev_ctx, {out_dim0 * (out_n_rows + 1)});
  auto* out_crows_data = out_crows.data<int64_t>();
  config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, x_dim0 * (x_n_rows + 1) + 1, 1);
  GetCsr3DNonZeroNumberCudaKernel<<<config.block_per_grid.x,
                                    config.thread_per_block.x,
                                    0,
                                    dev_ctx.stream()>>>(x_crows_data,
                                                        x_cols_data,
                                                        x_dim0,
                                                        x_n_rows,
                                                        x_cols_offsets_data,
                                                        d_starts,
                                                        d_ends,
                                                        out_n_rows,
                                                        out_crows_data);
  DenseTensor out_cols_offsets =
      phi::Empty<int64_t, Context>(dev_ctx, {out_dim0 * (out_n_rows + 1)});
  auto* out_cols_offsets_data = out_cols_offsets.data<int64_t>();
  phi::backends::gpu::GpuMemcpyAsync(
      out_cols_offsets_data,
      out_crows_data,
      out_dim0 * (out_n_rows + 1) * sizeof(int64_t),
      gpuMemcpyDeviceToDevice,
      dev_ctx.stream());
  dev_ctx.Wait();
  int64_t out_nnz =
#ifdef PADDLE_WITH_HIP
      thrust::reduce(thrust::hip::par.on(dev_ctx.stream()),
#else
      thrust::reduce(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                     out_crows_data,
                     out_crows_data + out_dim0 * (out_n_rows + 1));
  for (int64_t i = 0; i < out_dim0; ++i) {
    int64_t st = i * (out_n_rows + 1);
#ifdef PADDLE_WITH_HIP
    thrust::inclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
    thrust::inclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                           out_crows_data + st,
                           out_crows_data + st + out_n_rows + 1,
                           out_crows_data + st);
  }
#ifdef PADDLE_WITH_HIP
  thrust::inclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::inclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                         out_cols_offsets_data,
                         out_cols_offsets_data + out_dim0 * (out_n_rows + 1),
                         out_cols_offsets_data);

  DenseTensor out_cols = phi::Empty<int64_t, Context>(dev_ctx, {out_nnz});
  auto* out_cols_data = out_cols.data<int64_t>();
  DenseTensor out_values = phi::Empty<T, Context>(dev_ctx, {out_nnz});
  auto* out_values_data = out_values.data<T>();
  out->SetMember(out_crows, out_cols, out_values, out_dims);
  config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, x_dim0 * (x_n_rows + 1) + 1, 1);
  GetCsr3DCudaKernel<T><<<config.block_per_grid.x,
                          config.thread_per_block.x,
                          0,
                          dev_ctx.stream()>>>(x_crows_data,
                                              x_cols_data,
                                              x_values_data,
                                              x_cols_offsets_data,
                                              x_dim0,
                                              x_n_rows,
                                              d_starts,
                                              d_ends,
                                              out_n_rows,
                                              out_cols_offsets_data,
                                              out_crows_data,
                                              out_cols_data,
                                              out_values_data);
}

template <typename T, typename Context>
void SliceCsrCompute(const Context& dev_ctx,
                     const SparseCsrTensor& x,
                     const std::vector<int64_t>& axes,
                     const std::vector<int64_t>& starts,
                     const std::vector<int64_t>& ends,
                     SparseCsrTensor* out) {
  const phi::DDim& x_dims = x.dims();

  // Step1: Infer output dims
  auto out_dims = funcs::GetSliceDims<int64_t>(
      x_dims, axes, starts, ends, nullptr, nullptr);

  // Step2: Construct new axes, starts and ends.
  std::vector<int64_t> new_axes(3), new_starts(3), new_ends(3);
  funcs::ConstructNewSliceAttrs(
      x_dims, axes, starts, ends, &new_axes, &new_starts, &new_ends);

  // Setp3: Slice csr tensor according to its dimension
  if (x_dims.size() == 2) {
    SliceCsrTensor2D<T, Context>(
        dev_ctx, x, new_axes, new_starts, new_ends, out_dims, out);
  } else if (x_dims.size() == 3) {
    SliceCsrTensor3D<T, Context>(
        dev_ctx, x, new_axes, new_starts, new_ends, out_dims, out);
  } else {
    // throw exception
    common::errors::InvalidArgument(
        "Slice for Sparse CSR Tensor only support 2-D or 3-D, but got %d-D.",
        x_dims.size());
  }
}

template <typename T, typename Context>
void SliceCsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    const phi::IntArray& axes,
                    const phi::IntArray& starts,
                    const phi::IntArray& ends,
                    SparseCsrTensor* out) {
  const phi::DDim& x_dims = x.dims();

  std::vector<int64_t> axes_vec = axes.GetData();
  std::vector<int64_t> starts_vec = starts.GetData();
  std::vector<int64_t> ends_vec = ends.GetData();
  // Check and update attr
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(
      x_dims, &axes_vec, &starts_vec, &ends_vec);

  SliceCsrCompute<T, Context>(dev_ctx, x, axes_vec, starts_vec, ends_vec, out);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(slice_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SliceCooKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(slice_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SliceCsrKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
