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

constexpr int kVecSize = 8;
constexpr int BLOCK_DIM = 16;
constexpr int BLOCK_TILE_SIZE = 128;
constexpr int BLOCK_TILE_WIDTH = BLOCK_TILE_SIZE;
constexpr int BLOCK_TILE_HEIGHT = BLOCK_TILE_SIZE;
constexpr int THREAD_TILE_DIM = BLOCK_TILE_SIZE / BLOCK_DIM;

__global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
    fp8_fast_transpose_kernel(
        const phi::float8_e4m3fn* __restrict__ src,  // Source matrix (M x N)
        phi::float8_e4m3fn* __restrict__ dst,  // Destination matrix (N x M)
        uint32_t B,
        uint32_t M,
        uint32_t N,             // Batch size, M-dimension, N-dimension
        size_t batch_stride) {  // Stride between batches in global memory (M*N
                                // elements)
  // Shared memory tile with padding to avoid bank conflicts, padding instead of
  // swizzle for better performance
  __shared__ __align__(1024)
      fp8x8_t smem[BLOCK_TILE_HEIGHT][BLOCK_TILE_WIDTH / kVecSize + 1];

  // Thread-local storage: 8 fp8x8_t units, effectively an 8x8 block of fp8_t
  // values.
  fp8x8_t local_tile[kVecSize];
  fp8x8_t local_tile_transposed[kVecSize];

  // Thread indices within the block (0-15 for x and y, since 16x16 = 256
  // threads)
  const uint32_t tid_x = threadIdx.x;  // Column-wise thread index (0-15)
  const uint32_t tid_y = threadIdx.y;  // Row-wise thread index (0-15)

  // Block indices within the grid
  const uint32_t block_x = blockIdx.x;  // Tile index along N-dimension
  const uint32_t block_y = blockIdx.y;  // Tile index along M-dimension
  const uint32_t block_z = blockIdx.z;  // Batch index

  // Calculate global offsets for the current block's tile in the M x N source
  // matrix
  const uint32_t global_m_offset =
      block_y * BLOCK_TILE_HEIGHT;  // Starting M index for this block
  const uint32_t global_n_offset =
      block_x * BLOCK_TILE_WIDTH;  // Starting N index for this block

  const size_t current_batch_offset =
      static_cast<size_t>(batch_stride) * block_z;

// 1. Load src into register in uint2 vectorized manner.
#pragma unroll
  for (uint32_t k = 0; k < THREAD_TILE_DIM;
       ++k) {  // Iterate 8 times for the 8 rows in the thread's block
    const uint32_t src_global_row =
        global_m_offset + tid_y * THREAD_TILE_DIM + k;
    const uint32_t src_global_col_start =
        global_n_offset + tid_x * THREAD_TILE_DIM;

    // Check bounds for source matrix before loading
    // THREAD_TILE_DIM (8) is the width of the fp8x8_t block.
    const phi::float8_e4m3fn* src_ptr =
        src + current_batch_offset + static_cast<size_t>(src_global_row) * N +
        src_global_col_start;
    local_tile[k].load(src_ptr);
  }

// 2. Transpose local_tile in register level.
#pragma unroll
  for (uint32_t k_row = 0; k_row < THREAD_TILE_DIM; ++k_row) {
#pragma unroll
    for (uint32_t k_col = 0; k_col < THREAD_TILE_DIM; ++k_col) {
      local_tile_transposed[k_col].data.scalar[k_row] =
          local_tile[k_row].data.scalar[k_col];
    }
  }

// 3. Store transposed data to shared memory
#pragma unroll
  for (uint32_t k = 0; k < THREAD_TILE_DIM; ++k) {
    const uint32_t smem_row = tid_x * THREAD_TILE_DIM + k;
    const uint32_t smem_col_start = tid_y * THREAD_TILE_DIM / 8;  // = tid_y
    smem[smem_row][smem_col_start] = local_tile_transposed[k];
  }

  __syncthreads();

// 4. Store from shared memory to dst in uint2 vectorized manner.
#pragma unroll
  for (uint32_t k = 0; k < THREAD_TILE_DIM; ++k) {
    const uint32_t dst_global_row =
        global_n_offset + tid_y * THREAD_TILE_DIM + k;
    const uint32_t dst_global_col_start =
        global_m_offset + tid_x * THREAD_TILE_DIM;

    size_t offset = current_batch_offset +
                    static_cast<size_t>(dst_global_row) * M +
                    dst_global_col_start;
    phi::float8_e4m3fn* dst_ptr = dst + offset;

    fp8x8_t output_block;
    const uint32_t smem_row = tid_y * THREAD_TILE_DIM + k;
    const uint32_t smem_col = tid_x * THREAD_TILE_DIM / kVecSize;  // = tid_x
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
  block.x = BLOCK_DIM;  // 256 threads per block
  block.y = BLOCK_DIM;

  grid.z = B;
  grid.y = M / BLOCK_TILE_SIZE;  // not for un-aligned
  grid.x = N / BLOCK_TILE_SIZE;  // not for un-aligned

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

template void dispatch_fp8_fast_transpose_kernel<phi::float8_e4m3fn, int64_t>(
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
