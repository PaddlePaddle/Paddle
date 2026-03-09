// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_fill_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/index_fill_util.h"

namespace phi {

// GPU kernel for index_fill forward pass.
//
// Core idea — "Three-Segment Decomposition":
//   For a tensor with shape [d0, d1, ..., d_{dim}, ..., d_{n-1}], we split
//   the dimensions into three groups around the target `dim`:
//
//     outer_size = d0 * d1 * ... * d_{dim-1}    (all dims before `dim`)
//     dim_size   = d_{dim}                       (the target dim itself)
//     inner_size = d_{dim+1} * ... * d_{n-1}     (all dims after `dim`)
//
//   This lets us treat ANY N-dimensional tensor as a 3D logical block
//   [outer_size, dim_size, inner_size], and locate any element with:
//
//     offset = outer_idx * (dim_size * inner_size) + dim_idx * inner_size +
//     inner_idx
//
//   Total threads = outer_size * index_size * inner_size
//   (one thread per element that needs to be filled)
template <typename T>
__global__ void IndexFillCudaKernel(const T* x,
                                    const int64_t* index,
                                    const int64_t index_size,
                                    const int dim,
                                    const int64_t outer_size,
                                    const int64_t dim_size,
                                    const int64_t inner_size,
                                    const T fill_value,
                                    T* out) {
  int64_t idx =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
  int64_t total = index_size * outer_size * inner_size;
  if (idx >= total) return;

  // Decompose the flat thread index into the three logical coordinates.
  // The iteration order is: outer (slowest) → index → inner (fastest).
  int64_t inner_idx = idx % inner_size;
  int64_t temp = idx / inner_size;
  int64_t index_idx = temp % index_size;
  int64_t outer_idx = temp / index_size;

  // Look up the actual position along the target dimension from the index
  // tensor.
  int64_t dim_idx = index[index_idx];
  if (dim_idx < 0) dim_idx += dim_size;  // support negative indexing

  if (dim_idx < 0 || dim_idx >= dim_size) return;  // out-of-bounds guard

  // Convert the 3D logical coordinate back to a flat memory offset.
  int64_t offset =
      outer_idx * dim_size * inner_size + dim_idx * inner_size + inner_idx;

  out[offset] = fill_value;
}

// Host-side launch function: computes the three-segment sizes and launches
// the CUDA kernel.
template <typename T, typename Context>
void LaunchIndexFillCudaKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               int dim,
                               const DenseTensor& index,
                               const Scalar& value,
                               DenseTensor* out) {
  auto* x_data = x.data<T>();
  T fill_value = value.to<T>();

  // "Copy-then-modify" pattern: first copy x entirely into out, then
  // overwrite only the positions specified by the index.
  // Skip the copy if out already shares memory with x (inplace mode).
  bool is_initialized = out->initialized();
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (!is_initialized || (x.data<T>() != out->data<T>())) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  }

  auto* index_data = index.data<int64_t>();
  int64_t index_size = index.numel();

  if (index_size == 0) {
    return;
  }

  // --- Three-segment decomposition ---
  auto x_dims = x.dims();
  const int rank = x_dims.size();

  if (dim < 0) {
    dim += rank;
  }

  int64_t outer_size = 1;  // product of dims before `dim`
  int64_t inner_size = 1;  // product of dims after `dim`
  int64_t dim_size = x_dims[dim];

  for (int i = 0; i < dim; ++i) {
    outer_size *= x_dims[i];
  }
  for (int i = dim + 1; i < rank; ++i) {
    inner_size *= x_dims[i];
  }

  // Each thread handles one (outer, index_element, inner) triple.
  int64_t numel = outer_size * index_size * inner_size;

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  IndexFillCudaKernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          x_data,
          index_data,
          index_size,
          dim,
          outer_size,
          dim_size,
          inner_size,
          fill_value,
          out_data);
}

// Top-level kernel entry: validates inputs and dispatches to the launcher.
template <typename T, typename Context>
void IndexFillKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& index,
                     int dim,
                     const Scalar& value,
                     DenseTensor* out) {
  // Early return for zero-element output tensor.
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  auto x_dims = x.dims();
  const int rank = x_dims.size();

  // Normalize negative dim and validate range.
  int real_dim = dim;
  if (real_dim < 0) {
    real_dim += rank;
  }

  PADDLE_ENFORCE_GE(real_dim,
                    0,
                    common::errors::InvalidArgument(
                        "The dim must be >= -%d and < %d, but received %d.",
                        rank,
                        rank,
                        dim));
  PADDLE_ENFORCE_LT(real_dim,
                    rank,
                    common::errors::InvalidArgument(
                        "The dim must be >= -%d and < %d, but received %d.",
                        rank,
                        rank,
                        dim));

  // index_fill only supports 1-D index tensors (a list of positions along dim).
  PADDLE_ENFORCE_EQ(index.dims().size(),
                    1,
                    common::errors::InvalidArgument(
                        "The index tensor must be 1-D, but received %d-D.",
                        index.dims().size()));

  // Empty index means nothing to fill; just copy x to out.
  if (index.numel() == 0) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  // The kernel requires int64 indices. If the user passes int32 indices,
  // cast them to int64 via a lightweight GPU kernel (CastToInt64Kernel).
  DenseTensor index_int64;
  const DenseTensor* ptr_index = nullptr;

  if (index.dtype() == phi::DataType::INT32) {
    index_int64.Resize(index.dims());
    dev_ctx.template Alloc<int64_t>(&index_int64);

    int64_t index_numel = index.numel();
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, index_numel);

    phi::funcs::CastToInt64Kernel<int32_t><<<config.block_per_grid,
                                             config.thread_per_block,
                                             0,
                                             dev_ctx.stream()>>>(
        index.data<int32_t>(), index_int64.data<int64_t>(), index_numel);

    ptr_index = &index_int64;
  } else if (index.dtype() == phi::DataType::INT64) {
    ptr_index = &index;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        phi::DataTypeToString(index.dtype())));
  }

  LaunchIndexFillCudaKernel<T, Context>(
      dev_ctx, x, real_dim, *ptr_index, value, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(index_fill,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexFillKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
