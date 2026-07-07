// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/transpose_kernel.h"

#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/transpose_function.cuh"
#include "paddle/phi/kernels/impl/transpose_grad_kernel_impl.h"

namespace phi {
namespace funcs {

typedef struct alignas(8) fp8x8_t {
  union data_t {
    phi::float8_e4m3fn scalar[8];
    uint2 vector;
  };
  data_t data;

  __device__ __forceinline__ void load(const void* ptr) {
    data = *reinterpret_cast<const data_t*>(ptr);
  }

  __device__ __forceinline__ void store(void* ptr) const {
    *reinterpret_cast<data_t*>(ptr) = data;
  }
} fp8x8_t;

static constexpr int kFp8VecSize = 8;
static constexpr int kFp8BlockDim = 16;
static constexpr int kFp8BlockTileSize = 128;
static constexpr int kFp8BlockTileWidth = kFp8BlockTileSize;
static constexpr int kFp8BlockTileHeight = kFp8BlockTileSize;
static constexpr int kFp8ThreadTileDim = kFp8BlockTileSize / kFp8BlockDim;

__global__ void __launch_bounds__(kFp8BlockDim* kFp8BlockDim)
    fp8_fast_transpose_kernel(const phi::float8_e4m3fn* __restrict__ src,
                              phi::float8_e4m3fn* __restrict__ dst,
                              uint32_t B,
                              uint32_t M,
                              uint32_t N,
                              size_t batch_stride) {
  __shared__ __align__(1024)
      fp8x8_t smem[kFp8BlockTileHeight][kFp8BlockTileWidth / kFp8VecSize + 1];

  fp8x8_t local_tile[kFp8VecSize];
  fp8x8_t local_tile_transposed[kFp8VecSize];

  const uint32_t tid_x = threadIdx.x;
  const uint32_t tid_y = threadIdx.y;

  const uint32_t block_x = blockIdx.x;
  const uint32_t block_y = blockIdx.y;
  const uint32_t block_z = blockIdx.z;

  const uint32_t global_m_offset = block_y * kFp8BlockTileHeight;
  const uint32_t global_n_offset = block_x * kFp8BlockTileWidth;

  const size_t current_batch_offset =
      static_cast<size_t>(batch_stride) * block_z;

#pragma unroll
  for (uint32_t k = 0; k < kFp8ThreadTileDim; ++k) {
    const uint32_t src_global_row =
        global_m_offset + tid_y * kFp8ThreadTileDim + k;
    const uint32_t src_global_col_start =
        global_n_offset + tid_x * kFp8ThreadTileDim;

    const phi::float8_e4m3fn* src_ptr =
        src + current_batch_offset + static_cast<size_t>(src_global_row) * N +
        src_global_col_start;
    local_tile[k].load(src_ptr);
  }

#pragma unroll
  for (uint32_t k_row = 0; k_row < kFp8ThreadTileDim; ++k_row) {
#pragma unroll
    for (uint32_t k_col = 0; k_col < kFp8ThreadTileDim; ++k_col) {
      local_tile_transposed[k_col].data.scalar[k_row] =
          local_tile[k_row].data.scalar[k_col];
    }
  }

#pragma unroll
  for (uint32_t k = 0; k < kFp8ThreadTileDim; ++k) {
    const uint32_t smem_row = tid_x * kFp8ThreadTileDim + k;
    const uint32_t smem_col_start = tid_y * kFp8ThreadTileDim / 8;
    smem[smem_row][smem_col_start] = local_tile_transposed[k];
  }

  __syncthreads();

#pragma unroll
  for (uint32_t k = 0; k < kFp8ThreadTileDim; ++k) {
    const uint32_t dst_global_row =
        global_n_offset + tid_y * kFp8ThreadTileDim + k;
    const uint32_t dst_global_col_start =
        global_m_offset + tid_x * kFp8ThreadTileDim;

    size_t offset = current_batch_offset +
                    static_cast<size_t>(dst_global_row) * M +
                    dst_global_col_start;
    phi::float8_e4m3fn* dst_ptr = dst + offset;

    fp8x8_t output_block;
    const uint32_t smem_row = tid_y * kFp8ThreadTileDim + k;
    const uint32_t smem_col = tid_x * kFp8ThreadTileDim / kFp8VecSize;
    output_block = smem[smem_row][smem_col];
    output_block.store(dst_ptr);
  }
}

template <typename T, typename IndexType>
void dispatch_fp8_fast_transpose_kernel(const GPUContext& d,
                                        const T* input,
                                        const uint32_t B,
                                        const uint32_t M,
                                        const uint32_t N,
                                        T* output) {
  dim3 grid, block;
  block.x = kFp8BlockDim;
  block.y = kFp8BlockDim;

  grid.z = B;
  grid.y = M / kFp8BlockTileSize;
  grid.x = N / kFp8BlockTileSize;

  fp8_fast_transpose_kernel<<<grid, block, 0, d.stream()>>>(
      input, output, B, M, N, static_cast<size_t>(M) * static_cast<size_t>(N));
}

template void dispatch_fp8_fast_transpose_kernel<phi::float8_e4m3fn, int>(
    const GPUContext& d,
    const phi::float8_e4m3fn* input,
    const uint32_t B,
    const uint32_t M,
    const uint32_t N,
    phi::float8_e4m3fn* output);

}  // namespace funcs

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  size_t x_rank = x.dims().size();
  std::vector<int> formatted_axis = axis;
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      formatted_axis[i] = axis[i] + x_rank;
    }
  }

  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  if (formatted_axis.size() == 0) {
    Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }
  funcs::TransposeGPUKernelDriver<T>(dev_ctx, x, formatted_axis, out);
}
#ifdef _WIN32
INSTANTIATE_TRANSPOSE_KERNEL(float, GPUContext)
INSTANTIATE_TRANSPOSE_KERNEL(dtype::float16, GPUContext)
#endif
}  // namespace phi

PD_REGISTER_KERNEL(transpose,
                   GPU,
                   ALL_LAYOUT,
                   phi::TransposeKernel,
                   bool,
                   float,
                   double,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   uint8_t,
                   uint16_t,
                   uint32_t,
                   uint64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2) {}
