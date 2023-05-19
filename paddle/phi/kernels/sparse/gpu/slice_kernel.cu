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
