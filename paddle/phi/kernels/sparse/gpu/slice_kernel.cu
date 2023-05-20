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

#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {
namespace sparse {

__global__ void GetCooNonZeroNumsCudaKernel(const int64_t* x_indices_data,
                                            const int64_t* axes,
                                            const int64_t* starts,
                                            const int64_t* ends,
                                            const int64_t axes_size,
                                            const int64_t x_nnz,
                                            int* out_nnz) {
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
    atomicAdd(out_nnz, 1);
  }
}

template <typename T>
__global__ void GetCooOutCudaKernel(const int64_t* x_indices_data,
                                    const T* x_values_data,
                                    const int64_t* axes,
                                    const int64_t* starts,
                                    const int64_t* ends,
                                    const int64_t axes_size,
                                    const int64_t sparse_dim,
                                    const int64_t x_nnz,
                                    const int out_nnz,
                                    int64_t* out_indices_data,
                                    T* out_values_data) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid == 0) {
    int64_t index = 0;
    for (int64_t j = 0; j < x_nnz && index < static_cast<int64_t>(out_nnz);
         ++j) {
      bool hit = true;
      for (size_t ii = 0; ii < axes_size; ++ii) {
        auto item = x_indices_data[axes[ii] * x_nnz + j];
        if (!(starts[ii] <= item && item < ends[ii])) {
          hit = false;
          break;
        }
      }
      if (!hit) continue;
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
      index++;
    }
  }
}

template <typename T, typename Context>
void SliceCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const phi::IntArray& axes_arr,
                    const phi::IntArray& starts_arr,
                    const phi::IntArray& ends_arr,
                    SparseCooTensor* out) {
  const phi::DDim& x_dims = x.dims();

  std::vector<int64_t> axes = axes_arr.GetData();
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();

  // Step1: Check and update attr
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(x_dims, &axes, &starts, &ends);

  // Step2: Infer output dims
  auto out_dims = funcs::GetSliceDims<int64_t>(
      x_dims, axes, starts, ends, nullptr, nullptr);

  // Step3: Get the number of non zero elements
  DenseTensor d_out_nnz = phi::Empty<int32_t>(dev_ctx, {1});
  int* d_out_nnz_ptr = d_out_nnz.data<int32_t>();
  phi::backends::gpu::GpuMemsetAsync(
      d_out_nnz_ptr, 0, sizeof(int32_t), dev_ctx.stream());

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

  const auto* x_indices_data = x.indices().data<int64_t>();

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x.nnz(), 1);
  GetCooNonZeroNumsCudaKernel<<<config.block_per_grid.x,
                                config.thread_per_block.x,
                                0,
                                dev_ctx.stream()>>>(x_indices_data,
                                                    d_axes,
                                                    d_starts,
                                                    d_ends,
                                                    axes.size(),
                                                    x.nnz(),
                                                    d_out_nnz_ptr);

  int32_t out_nnz = 0;
  phi::backends::gpu::GpuMemcpyAsync(&out_nnz,
                                     d_out_nnz_ptr,
                                     sizeof(int32_t),
                                     gpuMemcpyDeviceToHost,
                                     dev_ctx.stream());

  // Step4: Get the values and indices of output
  auto sparse_dim = static_cast<int64_t>(x.sparse_dim());
  DenseTensor out_indices =
      phi::Empty<int64_t, Context>(dev_ctx, {sparse_dim, out_nnz});
  DenseTensor out_values = phi::Empty<T, Context>(dev_ctx, {out_nnz});
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());

  auto* out_indices_data = out_indices.data<int64_t>();
  auto* out_values_data = out_values.data<T>();
  const auto* x_values_data = x.values().data<T>();

  GetCooOutCudaKernel<T><<<1, 1, 0, dev_ctx.stream()>>>(x_indices_data,
                                                        x_values_data,
                                                        d_axes,
                                                        d_starts,
                                                        d_ends,
                                                        axes.size(),
                                                        sparse_dim,
                                                        x.nnz(),
                                                        out_nnz,
                                                        out_indices_data,
                                                        out_values_data);
}

__global__ void GetCsrNonZeroNumsCudaKernel(const int64_t* x_crows_data,
                                            const int64_t* x_cols_data,
                                            const int64_t x_crows_start,
                                            const int64_t x_crows_end,
                                            const int64_t min_col,
                                            const int64_t max_col,
                                            int* out_nnz,
                                            const int64_t offset = 0) {
  CUDA_KERNEL_LOOP_TYPE(i, x_crows_end - x_crows_start, int64_t) {
    int64_t st = x_crows_data[x_crows_start + i] + offset;
    int64_t ed = x_crows_data[x_crows_start + i + 1] + offset;
    for (int64_t jj = st; jj < ed; ++jj) {
      if (x_cols_data[jj] >= min_col && x_cols_data[jj] < max_col) {
        atomicAdd(out_nnz, 1);
      }
    }
  }
}

template <typename T>
__global__ void GetCsrSubMatrixCudaKernel(const int64_t* x_crows_data,
                                          const int64_t* x_cols_data,
                                          const T* x_values_data,
                                          const int64_t x_crows_start,
                                          const int64_t x_crows_end,
                                          const int64_t min_col,
                                          const int64_t max_col,
                                          int64_t* out_crows_data,
                                          int64_t* out_cols_data,
                                          T* out_values_data,
                                          const int64_t out_crows_offset = 0,
                                          const int64_t x_cols_offset = 0,
                                          const int64_t out_cols_offset = 0) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid == 0) {
    out_crows_data[out_crows_offset] = 0;
    int64_t index = 0, new_n_rows = x_crows_end - x_crows_start;
    for (int i = 0; i < new_n_rows; ++i) {
      int64_t st = x_crows_data[x_crows_start + i] + x_cols_offset;
      int64_t ed = x_crows_data[x_crows_start + i + 1] + x_cols_offset;
      for (int64_t jj = st; jj < ed; ++jj) {
        if (x_cols_data[jj] >= min_col && x_cols_data[jj] < max_col) {
          out_cols_data[out_cols_offset + index] = x_cols_data[jj] - min_col;
          out_values_data[out_cols_offset + index] = x_values_data[jj];
          index++;
        }
      }
      out_crows_data[out_crows_offset + i + 1] = index;
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
  // Step1: Get the number of non zero elements for out
  DenseTensor d_out_nnz = phi::Empty<int32_t>(dev_ctx, {1});
  int* d_out_nnz_ptr = d_out_nnz.data<int32_t>();
  phi::backends::gpu::GpuMemsetAsync(
      d_out_nnz_ptr, 0, sizeof(int32_t), dev_ctx.stream());
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, ends[0] - starts[0], 1);
  GetCsrNonZeroNumsCudaKernel<<<config.block_per_grid.x,
                                config.thread_per_block.x,
                                0,
                                dev_ctx.stream()>>>(x_crows_data,
                                                    x_cols_data,
                                                    starts[0],
                                                    ends[0],
                                                    starts[1],
                                                    ends[1],
                                                    d_out_nnz_ptr,
                                                    0);
  int32_t out_nnz = 0;
  phi::backends::gpu::GpuMemcpyAsync(&out_nnz,
                                     d_out_nnz_ptr,
                                     sizeof(int32_t),
                                     gpuMemcpyDeviceToHost,
                                     dev_ctx.stream());
  // Step2: Set out
  int64_t out_n_rows = ends[0] - starts[0];
  DenseTensor out_crows =
      phi::Empty<int64_t, Context>(dev_ctx, {out_n_rows + 1});
  DenseTensor out_cols = phi::Empty<int64_t, Context>(dev_ctx, {out_nnz});
  DenseTensor out_values = phi::Empty<T, Context>(dev_ctx, {out_nnz});
  out->SetMember(out_crows, out_cols, out_values, out_dims);
  GetCsrSubMatrixCudaKernel<T>
      <<<1, 1, 0, dev_ctx.stream()>>>(x_crows_data,
                                      x_cols_data,
                                      x_values_data,
                                      starts[0],
                                      ends[0],
                                      starts[1],
                                      ends[1],
                                      out_crows.data<int64_t>(),
                                      out_cols.data<int64_t>(),
                                      out_values.data<T>(),
                                      0,
                                      0,
                                      0);
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
  int64_t offset = 0;
  int64_t out_nnz = 0;
  int64_t* temp_x_crows_data = new int64_t[x_dim0 * (x_n_rows + 1)];
  phi::backends::gpu::GpuMemcpyAsync(temp_x_crows_data,
                                     x_crows_data,
                                     sizeof(int64_t) * x_dim0 * (x_n_rows + 1),
                                     gpuMemcpyDeviceToHost,
                                     dev_ctx.stream());

  // Step1: Get the number of non zero elements for out
  std::vector<int64_t> all_nnzs(ends[0] - starts[0]);
  DenseTensor d_nnz = phi::Empty<int32_t>(dev_ctx, {1});
  int* d_nnz_ptr = d_nnz.data<int32_t>();

  for (int64_t i = 0; i < x_dim0; ++i) {
    if (i >= starts[0] && i < ends[0]) {  // slice dim 0
      int64_t crows_st = i * (x_n_rows + 1) + starts[1];
      int64_t crows_ed = i * (x_n_rows + 1) + ends[1];

      phi::backends::gpu::GpuMemsetAsync(
          d_nnz_ptr, 0, sizeof(int32_t), dev_ctx.stream());
      auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
          dev_ctx, crows_ed - crows_st, 1);
      GetCsrNonZeroNumsCudaKernel<<<config.block_per_grid.x,
                                    config.thread_per_block.x,
                                    0,
                                    dev_ctx.stream()>>>(x_crows_data,
                                                        x_cols_data,
                                                        crows_st,
                                                        crows_ed,
                                                        starts[2],
                                                        ends[2],
                                                        d_nnz_ptr,
                                                        offset);
      int32_t nnz = 0;
      phi::backends::gpu::GpuMemcpyAsync(&nnz,
                                         d_nnz_ptr,
                                         sizeof(int32_t),
                                         gpuMemcpyDeviceToHost,
                                         dev_ctx.stream());
      out_nnz += static_cast<int64_t>(nnz);
      all_nnzs[i - starts[0]] = static_cast<int64_t>(nnz);
    }
    // get the start index in non_zero_elements_ and non_zero_cols_
    offset += temp_x_crows_data[(i + 1) * (x_n_rows + 1) - 1];
  }

  // Set out
  const int64_t out_dim0 = out_dims[0], out_n_rows = out_dims[1];
  DenseTensor out_crows =
      phi::Empty<int64_t, Context>(dev_ctx, {out_dim0 * (out_n_rows + 1)});
  DenseTensor out_cols = phi::Empty<int64_t, Context>(dev_ctx, {out_nnz});
  DenseTensor out_values = phi::Empty<T, Context>(dev_ctx, {out_nnz});
  out->SetMember(out_crows, out_cols, out_values, out_dims);

  int64_t x_cols_offset = 0, out_crows_offset = 0, out_cols_offset = 0;
  for (int64_t i = 0; i < x_dim0; ++i) {
    if (i >= starts[0] && i < ends[0]) {  // slice dim 0
      int64_t x_crows_start = i * (x_n_rows + 1) + starts[1];
      int64_t x_crows_end = i * (x_n_rows + 1) + ends[1];

      GetCsrSubMatrixCudaKernel<T>
          <<<1, 1, 0, dev_ctx.stream()>>>(x_crows_data,
                                          x_cols_data,
                                          x_values_data,
                                          x_crows_start,
                                          x_crows_end,
                                          starts[2],
                                          ends[2],
                                          out_crows.data<int64_t>(),
                                          out_cols.data<int64_t>(),
                                          out_values.data<T>(),
                                          out_crows_offset,
                                          x_cols_offset,
                                          out_cols_offset);
      out_crows_offset += (out_n_rows + 1);
      out_cols_offset += all_nnzs[i - starts[0]];
    }
    x_cols_offset += temp_x_crows_data[(i + 1) * (x_n_rows + 1) - 1];
  }
}

template <typename T, typename Context>
void SliceCsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    const phi::IntArray& axes_arr,
                    const phi::IntArray& starts_arr,
                    const phi::IntArray& ends_arr,
                    SparseCsrTensor* out) {
  const phi::DDim& x_dims = x.dims();

  std::vector<int64_t> axes = axes_arr.GetData();
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();

  // Step1: Check and update attr
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(x_dims, &axes, &starts, &ends);

  // Step2: Infer output dims
  auto out_dims = funcs::GetSliceDims<int64_t>(
      x_dims, axes, starts, ends, nullptr, nullptr);

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

  // Step3: Construct new axes, starts and ends.
  std::vector<int64_t> new_axes(3), new_starts(3), new_ends(3);
  funcs::ConstructNewSliceAttrs(
      x_dims, axes, starts, ends, &new_axes, &new_starts, &new_ends);

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

  // Setp4: Slice csr tensor according to its dimension
  if (x_dims.size() == 2) {
    SliceCsrTensor2D<T, Context>(
        dev_ctx, x, new_axes, new_starts, new_ends, out_dims, out);
  } else if (x_dims.size() == 3) {
    SliceCsrTensor3D<T, Context>(
        dev_ctx, x, new_axes, new_starts, new_ends, out_dims, out);
  } else {
    // throw exception
    phi::errors::InvalidArgument(
        "Slice for Sparse CSR Tensor only support 2-D or 3-D, but got %d-D.",
        x_dims.size());
  }
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
