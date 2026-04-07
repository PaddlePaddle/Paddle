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

#include <climits>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/index_fill_kernel.h"

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
//
// IndexT: int32_t when numel <= INT32_MAX (faster mod/div on GPU),
//         int64_t otherwise.
// IndT:   matches index tensor dtype (int32_t or int64_t), avoids a
//         CastToInt64 kernel launch when the user supplies int32 indices.
template <typename T, typename IndexT, typename IndT>
__global__ void IndexFillCudaKernel(const T* x,
                                    const IndT* index,
                                    const IndexT index_size,
                                    const int dim,
                                    const IndexT outer_size,
                                    const int64_t dim_size,
                                    const IndexT inner_size,
                                    const T fill_value,
                                    T* out) {
  IndexT idx =
      static_cast<IndexT>(threadIdx.x) +
      static_cast<IndexT>(blockIdx.x) * static_cast<IndexT>(blockDim.x);
  IndexT total = index_size * outer_size * inner_size;
  if (idx >= total) return;

  // Decompose the flat thread index into the three logical coordinates.
  // The iteration order is: outer (slowest) → index → inner (fastest).
  IndexT inner_idx = idx % inner_size;
  IndexT temp = idx / inner_size;
  IndexT index_idx = temp % index_size;
  IndexT outer_idx = temp / index_size;

  // Look up the actual position along the target dimension from the index
  // tensor. Widen to int64 for the bounds check against dim_size.
  int64_t dim_idx = static_cast<int64_t>(index[index_idx]);
  if (dim_idx < 0) dim_idx += dim_size;  // support negative indexing

  if (dim_idx < 0 || dim_idx >= dim_size) return;  // out-of-bounds guard

  // Convert the 3D logical coordinate back to a flat memory offset.
  IndexT offset = outer_idx * static_cast<IndexT>(dim_size) * inner_size +
                  static_cast<IndexT>(dim_idx) * inner_size + inner_idx;

  out[offset] = fill_value;
}

// Host-side launch function: computes the three-segment sizes and launches
// the CUDA kernel with the appropriate index types (IndexT, IndT).
template <typename T, typename Context, typename IndexT, typename IndT>
void LaunchIndexFillCudaKernelImpl(const Context& dev_ctx,
                                   const T* x_data,
                                   const IndT* index_data,
                                   IndexT index_size,
                                   int dim,
                                   IndexT outer_size,
                                   int64_t dim_size,
                                   IndexT inner_size,
                                   T fill_value,
                                   T* out_data) {
  IndexT numel = outer_size * index_size * inner_size;
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  IndexFillCudaKernel<T, IndexT, IndT>
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

template <typename T, typename Context, typename IndT>
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

  const IndT* index_data = index.data<IndT>();
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

  // Select int32 index arithmetic when the total work fits in 32 bits —
  // GPU mod/div on 32-bit integers is significantly faster than on 64-bit.
  const int64_t numel = x.numel();
  constexpr int64_t kInt32Max = static_cast<int64_t>(INT32_MAX);

  if (numel <= kInt32Max) {
    LaunchIndexFillCudaKernelImpl<T, Context, int32_t, IndT>(
        dev_ctx,
        x_data,
        index_data,
        static_cast<int32_t>(index_size),
        dim,
        static_cast<int32_t>(outer_size),
        dim_size,
        static_cast<int32_t>(inner_size),
        fill_value,
        out_data);
  } else {
    LaunchIndexFillCudaKernelImpl<T, Context, int64_t, IndT>(dev_ctx,
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

  // Dispatch directly on index dtype — no cast kernel needed.
  if (index.dtype() == phi::DataType::INT32) {
    LaunchIndexFillCudaKernel<T, Context, int32_t>(
        dev_ctx, x, real_dim, index, value, out);
  } else if (index.dtype() == phi::DataType::INT64) {
    LaunchIndexFillCudaKernel<T, Context, int64_t>(
        dev_ctx, x, real_dim, index, value, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        phi::DataTypeToString(index.dtype())));
  }
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
