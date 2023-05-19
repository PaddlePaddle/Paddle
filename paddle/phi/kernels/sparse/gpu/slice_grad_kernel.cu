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
void SliceCooGradKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const SparseCooTensor& out_grad,
                        const phi::IntArray& axes_arr,
                        const phi::IntArray& starts_arr,
                        const phi::IntArray& ends_arr,
                        SparseCooTensor* x_grad) {
  const phi::DDim& x_dims = x.dims();

  std::vector<int64_t> axes = axes_arr.GetData();
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();

  // Step1: Check and update sparse slice attrs
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(x_dims, &axes, &starts, &ends);

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
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_grad_nnz, 1);
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
