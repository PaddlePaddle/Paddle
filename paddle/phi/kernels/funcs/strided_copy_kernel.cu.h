/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <iostream>
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"

namespace phi {
bool VerifyStridedCopyThreadConfigurationParameters(const dim3& block,
                                                    const dim3& grid) {
  return block.x <= 1024 && block.y <= 1024 && block.z <= 64 &&
         block.x * block.y * block.z <= 1024 &&
         block.x * block.y * block.z >= 96 && grid.y < 65536 && grid.z < 65536;
}

__device__ bool is_aligned(const void* ptr, size_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

template <typename T, size_t N>
__global__ void Contiguous2StridedCaseOneFunc(
    const T* input_data,
    T* out_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    phi::Array<int64_t, 6> dims,
    const int64_t x_max) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < x_max) {
    int64_t input_offset = (blockIdx.z * gridDim.y + blockIdx.y) * x_max + x;
    int64_t output_offset = 0;

    int64_t reg_dims[6] = {
        dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]};
    int64_t coordinate[phi::DDim::kMaxRank + 1];

    switch (N) {
      case 1:
        coordinate[0] = x % reg_dims[0];
        break;
      case 2:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        break;
      case 3:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        break;
      case 4:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        break;
      case 5:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        coordinate[4] = blockIdx.y / reg_dims[2] % reg_dims[3];
        break;
      case 6:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        coordinate[4] = blockIdx.y / reg_dims[2] % reg_dims[3];
        coordinate[5] = blockIdx.y / (reg_dims[2] * reg_dims[3]);
        break;
      case 7:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        coordinate[4] = blockIdx.y / reg_dims[2] % reg_dims[3];
        coordinate[5] = blockIdx.y / (reg_dims[2] * reg_dims[3]);
        coordinate[6] = blockIdx.z % reg_dims[4];
        break;
      case 8:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        coordinate[4] = blockIdx.y / reg_dims[2] % reg_dims[3];
        coordinate[5] = blockIdx.y / (reg_dims[2] * reg_dims[3]);
        coordinate[6] = blockIdx.z % reg_dims[4];
        coordinate[7] = blockIdx.z / reg_dims[4] % reg_dims[5];
        break;
      case 9:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        coordinate[4] = blockIdx.y / reg_dims[2] % reg_dims[3];
        coordinate[5] = blockIdx.y / (reg_dims[2] * reg_dims[3]);
        coordinate[6] = blockIdx.z % reg_dims[4];
        coordinate[7] = blockIdx.z / reg_dims[4] % reg_dims[5];
        coordinate[8] = blockIdx.z / (reg_dims[4] * reg_dims[5]);
        break;
    }

#pragma unroll
    for (int dim = N - 1; dim >= 0; --dim) {
      output_offset += coordinate[N - 1 - dim] * output_stride[dim];
    }

    out_data[output_offset] = input_data[input_offset];
  }
}

template <typename T, size_t N>
__global__ void Contiguous2StridedCaseOneDiffDimFunc(
    const T* input_data,
    T* out_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    phi::Array<int64_t, 6> dims,
    const int64_t x_max) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < x_max) {
    int64_t output_offset = 0;

    int64_t reg_dims[6] = {
        dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]};
    int64_t coordinate[phi::DDim::kMaxRank + 1];

    switch (N) {
      case 1:
        coordinate[0] = x % reg_dims[0];
        break;
      case 2:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        break;
      case 3:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        break;
      case 4:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        break;
      case 5:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        coordinate[4] = blockIdx.y / reg_dims[2] % reg_dims[3];
        break;
      case 6:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        coordinate[4] = blockIdx.y / reg_dims[2] % reg_dims[3];
        coordinate[5] = blockIdx.y / (reg_dims[2] * reg_dims[3]);
        break;
      case 7:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        coordinate[4] = blockIdx.y / reg_dims[2] % reg_dims[3];
        coordinate[5] = blockIdx.y / (reg_dims[2] * reg_dims[3]);
        coordinate[6] = blockIdx.z % reg_dims[4];
        break;
      case 8:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        coordinate[4] = blockIdx.y / reg_dims[2] % reg_dims[3];
        coordinate[5] = blockIdx.y / (reg_dims[2] * reg_dims[3]);
        coordinate[6] = blockIdx.z % reg_dims[4];
        coordinate[7] = blockIdx.z / reg_dims[4] % reg_dims[5];
        break;
      case 9:
        coordinate[0] = x % reg_dims[0];
        coordinate[1] = x / reg_dims[0] % reg_dims[1];
        coordinate[2] = x / (reg_dims[0] * reg_dims[1]);
        coordinate[3] = blockIdx.y % reg_dims[2];
        coordinate[4] = blockIdx.y / reg_dims[2] % reg_dims[3];
        coordinate[5] = blockIdx.y / (reg_dims[2] * reg_dims[3]);
        coordinate[6] = blockIdx.z % reg_dims[4];
        coordinate[7] = blockIdx.z / reg_dims[4] % reg_dims[5];
        coordinate[8] = blockIdx.z / (reg_dims[4] * reg_dims[5]);
        break;
    }

#pragma unroll
    for (int dim = N - 1; dim >= 0; --dim) {
      output_offset += coordinate[N - 1 - dim] * output_stride[dim];
    }

    out_data[output_offset] = input_data[0];
  }
}

// Check whether "out" is the output of the stride slice.
bool CheckStride(
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_stride,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& dims,
    int rank,
    int64_t numel) {
  int64_t stride = numel;
  int64_t last_stride = 1;
  for (size_t i = 0; i < rank; i++) {
    if (output_stride[i] < last_stride) return true;
    last_stride = output_stride[i];
    stride = stride / dims[i];
    if (output_stride[i] > stride) return false;
  }
  return true;
}

template <typename T, typename Context>
bool LaunchContiguous2StridedCaseOneKernel(
    const Context& dev_ctx,
    const T* input_data,
    T* output_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_stride,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& dims,
    int rank,
    int64_t numel,
    bool diff_dims) {
  if (!CheckStride(output_stride, dims, rank, numel)) {
    return false;
  }
  dim3 grid(1, 1, 1), block(1, 1, 1);
  phi::Array<int64_t, 6> cur_dims;
  block.x = 512;

  if (rank >= 1) {
    grid.x = (numel + block.x - 1) / block.x;
    cur_dims[0] = dims[rank - 1];
  }

  if (rank >= 2) {
    cur_dims[1] = dims[rank - 2];
  }

  if (rank >= 4) {
    grid.x = (dims[rank - 1] * dims[rank - 2] * dims[rank - 3] + block.x - 1) /
             block.x;
    grid.y = dims[rank - 4];
    cur_dims[2] = dims[rank - 4];
  }

  if (rank >= 5) {
    grid.y = dims[rank - 4] * dims[rank - 5];
    cur_dims[2] = dims[rank - 4];
    cur_dims[3] = dims[rank - 5];
  }

  if (rank >= 6) {
    grid.y = dims[rank - 4] * dims[rank - 5] * dims[rank - 6];
  }

  if (rank >= 7) {
    grid.z = dims[rank - 7];
    cur_dims[4] = dims[rank - 7];
  }

  if (rank >= 8) {
    grid.z = dims[rank - 7] * dims[rank - 8];
    cur_dims[5] = dims[rank - 8];
  }

  if (rank >= 9) {
    grid.z = dims[rank - 7] * dims[rank - 8] * dims[rank - 9];
  }

  if (!VerifyStridedCopyThreadConfigurationParameters(block, grid)) {
    return false;
  }

  if (diff_dims) {
    switch (rank) {
      case 1:
        Contiguous2StridedCaseOneDiffDimFunc<T, 1>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   output_data,
                                                   output_stride,
                                                   cur_dims,
                                                   dims[rank - 1]);
        break;
      case 2:
        Contiguous2StridedCaseOneDiffDimFunc<T, 2>
            <<<grid, block, 0, dev_ctx.stream()>>>(
                input_data,
                output_data,
                output_stride,
                cur_dims,
                dims[rank - 1] * dims[rank - 2]);
        break;
#define CASE_RANK(__Rk)                                        \
  case __Rk:                                                   \
    Contiguous2StridedCaseOneDiffDimFunc<T, __Rk>              \
        <<<grid, block, 0, dev_ctx.stream()>>>(                \
            input_data,                                        \
            output_data,                                       \
            output_stride,                                     \
            cur_dims,                                          \
            dims[rank - 1] * dims[rank - 2] * dims[rank - 3]); \
    break;
        CASE_RANK(3);
        CASE_RANK(4);
        CASE_RANK(5);
        CASE_RANK(6);
        CASE_RANK(7);
        CASE_RANK(8);
        CASE_RANK(9);
#undef CASE_RANK
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "The rank of input should be less than 9, but received %d.", rank));
    }
  } else {
    switch (rank) {
      case 1:
        Contiguous2StridedCaseOneFunc<T, 1>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   output_data,
                                                   output_stride,
                                                   cur_dims,
                                                   dims[rank - 1]);
        break;
      case 2:
        Contiguous2StridedCaseOneFunc<T, 2>
            <<<grid, block, 0, dev_ctx.stream()>>>(
                input_data,
                output_data,
                output_stride,
                cur_dims,
                dims[rank - 1] * dims[rank - 2]);
        break;
#define CASE_RANK(__Rk)                                        \
  case __Rk:                                                   \
    Contiguous2StridedCaseOneFunc<T, __Rk>                     \
        <<<grid, block, 0, dev_ctx.stream()>>>(                \
            input_data,                                        \
            output_data,                                       \
            output_stride,                                     \
            cur_dims,                                          \
            dims[rank - 1] * dims[rank - 2] * dims[rank - 3]); \
    break;
        CASE_RANK(3);
        CASE_RANK(4);
        CASE_RANK(5);
        CASE_RANK(6);
        CASE_RANK(7);
        CASE_RANK(8);
        CASE_RANK(9);
#undef CASE_RANK
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "The rank of input should be less than 9, but received %d.", rank));
    }
  }

  return true;
}

template <typename T, size_t RANK>
__global__ void Contiguous2StridedCaseZeroFunc(
    const T* input_data,
    T* output_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride) {
  int64_t input_offset = (blockIdx.z * gridDim.y * gridDim.x +
                          blockIdx.y * gridDim.x + blockIdx.x) *
                             blockDim.z * blockDim.y * blockDim.x +
                         threadIdx.z * blockDim.y * blockDim.x +
                         threadIdx.y * blockDim.x + threadIdx.x;
  int64_t output_offset = 0;

  int64_t coordinate[6] = {threadIdx.x,
                           threadIdx.y,
                           threadIdx.z,
                           blockIdx.x,
                           blockIdx.y,
                           blockIdx.z};

#pragma unroll
  for (int dim = RANK - 1; dim >= 0; --dim) {
    output_offset += coordinate[RANK - 1 - dim] * output_stride[dim];
  }

  output_data[output_offset] = input_data[input_offset];
}

template <typename T, size_t RANK>
__global__ void Contiguous2StridedCaseZeroDiffDimFunc(
    const T* input_data,
    T* output_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride) {
  int64_t output_offset = 0;

  int64_t coordinate[6] = {threadIdx.x,
                           threadIdx.y,
                           threadIdx.z,
                           blockIdx.x,
                           blockIdx.y,
                           blockIdx.z};

#pragma unroll
  for (int dim = RANK - 1; dim >= 0; --dim) {
    output_offset += coordinate[RANK - 1 - dim] * output_stride[dim];
  }

  output_data[output_offset] = input_data[0];
}

template <typename T, typename Context>
bool LaunchContiguous2StridedCaseZeroKernel(
    const Context& dev_ctx,
    const T* input_data,
    T* output_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_stride,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& dims,
    int rank,
    bool diff_dims) {
  if (rank > 6) {
    return false;
  }

  dim3 grid(1, 1, 1), block(1, 1, 1);

  if (rank >= 1) {
    block.x = dims[rank - 1];
  }

  if (rank >= 2) {
    block.y = dims[rank - 2];
  }

  if (rank >= 3) {
    block.z = dims[rank - 3];
  }

  if (rank >= 4) {
    grid.x = dims[rank - 4];
  }

  if (rank >= 5) {
    grid.y = dims[rank - 5];
  }

  if (rank >= 6) {
    grid.z = dims[rank - 6];
  }

  if (!VerifyStridedCopyThreadConfigurationParameters(block, grid)) {
    return false;
  }

  if (diff_dims) {
    switch (rank) {
#define CASE_RANK(__Rk)                              \
  case __Rk:                                         \
    Contiguous2StridedCaseZeroDiffDimFunc<T, __Rk>   \
        <<<grid, block, 0, dev_ctx.stream()>>>(      \
            input_data, output_data, output_stride); \
    break;
      CASE_RANK(1);
      CASE_RANK(2);
      CASE_RANK(3);
      CASE_RANK(4);
      CASE_RANK(5);
      CASE_RANK(6);
#undef CASE_RANK
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "The rank of input should be less than 9, but received %d.", rank));
    }
  } else {
    switch (rank) {
#define CASE_RANK(__Rk)                              \
  case __Rk:                                         \
    Contiguous2StridedCaseZeroFunc<T, __Rk>          \
        <<<grid, block, 0, dev_ctx.stream()>>>(      \
            input_data, output_data, output_stride); \
    break;
      CASE_RANK(1);
      CASE_RANK(2);
      CASE_RANK(3);
      CASE_RANK(4);
      CASE_RANK(5);
      CASE_RANK(6);
#undef CASE_RANK
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "The rank of input should be less than 9, but received %d.", rank));
    }
  }

  return true;
}

template <typename T, int VecSize, size_t OUT_RANK>
__global__ void Contiguous2StridedDefaultDiffDimFunc(
    const T* input_data,
    T* output_data,
    Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    Array<int64_t, phi::DDim::kMaxRank + 1> dims,
    const int64_t numel) {
  int MAX_LOAD_BYTES = VecSize * sizeof(T);
  int64_t gid = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
#pragma unroll
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x * VecSize) {
    int64_t output_offset = 0;
    int64_t index_tmp = i;

    for (int dim = OUT_RANK - 1; dim >= 0; --dim) {
      output_offset += (index_tmp % dims[dim]) * output_stride[dim];
      index_tmp = index_tmp / dims[dim];
    }
    if (is_aligned(&output_data[output_offset], MAX_LOAD_BYTES)) {
      using VecType = kps::details::VectorType<T, VecSize>;
      const VecType* src = reinterpret_cast<const VecType*>(&input_data[0]);
      VecType* dst = reinterpret_cast<VecType*>(&output_data[output_offset]);
      *dst = *src;
    } else {
      for (int j = 0; j < VecSize; j++) {
        output_data[output_offset + j] = input_data[0];
      }
    }
  }
}

template <typename T, int VecSize, size_t OUT_RANK>
__global__ void Contiguous2StridedDefaultFunc(
    const T* input_data,
    T* output_data,
    Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    Array<int64_t, phi::DDim::kMaxRank + 1> dims,
    const int64_t numel) {
  int MAX_LOAD_BYTES = VecSize * sizeof(T);
  int64_t gid = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
#pragma unroll
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x * VecSize) {
    int64_t output_offset = 0;
    int64_t index_tmp = i;
    for (int dim = OUT_RANK - 1; dim >= 0; --dim) {
      output_offset += (index_tmp % dims[dim]) * output_stride[dim];
      index_tmp = index_tmp / dims[dim];
    }
    if (is_aligned(&output_data[output_offset], MAX_LOAD_BYTES)) {
      using VecType = kps::details::VectorType<T, VecSize>;
      const VecType* src = reinterpret_cast<const VecType*>(&input_data[i]);
      VecType* dst = reinterpret_cast<VecType*>(&output_data[output_offset]);
      *dst = *src;
    } else {
      for (int j = 0; j < VecSize; j++) {
        output_data[output_offset + j] = input_data[i + j];
      }
    }
  }
}

template <typename T, typename Context, int VecSize>
void LaunchContiguous2StridedDefaultKernel(
    const Context& dev_ctx,
    const T* input_data,
    T* output_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_stride,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& dims,
    int rank,
    int64_t numel,
    bool diff_dims) {
  constexpr int loop_count = 4;
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, numel, VecSize * loop_count);
  auto& grid = config.block_per_grid;
  auto& block = config.thread_per_block;

  if (diff_dims) {
    if (VecSize == 4) {
      switch (rank) {
#define CASE_RANK(__Rk)                                           \
  case __Rk:                                                      \
    Contiguous2StridedDefaultDiffDimFunc<T, 4, __Rk>              \
        <<<grid, block, 0, dev_ctx.stream()>>>(                   \
            input_data, output_data, output_stride, dims, numel); \
    break
        CASE_RANK(1);
        CASE_RANK(2);
        CASE_RANK(3);
        CASE_RANK(4);
        CASE_RANK(5);
        CASE_RANK(6);
        CASE_RANK(7);
        CASE_RANK(8);
        CASE_RANK(9);
#undef CASE_RANK

        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "The rank of input should be less than 9, but received %d.",
              rank));
      }

    } else if (VecSize == 2) {
      switch (rank) {
#define CASE_RANK(__Rk)                                           \
  case __Rk:                                                      \
    Contiguous2StridedDefaultDiffDimFunc<T, 2, __Rk>              \
        <<<grid, block, 0, dev_ctx.stream()>>>(                   \
            input_data, output_data, output_stride, dims, numel); \
    break
        CASE_RANK(1);
        CASE_RANK(2);
        CASE_RANK(3);
        CASE_RANK(4);
        CASE_RANK(5);
        CASE_RANK(6);
        CASE_RANK(7);
        CASE_RANK(8);
        CASE_RANK(9);
#undef CASE_RANK

        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "The rank of input should be less than 9, but received %d.",
              rank));
      }
    } else {
      switch (rank) {
#define CASE_RANK(__Rk)                                           \
  case __Rk:                                                      \
    Contiguous2StridedDefaultDiffDimFunc<T, 1, __Rk>              \
        <<<grid, block, 0, dev_ctx.stream()>>>(                   \
            input_data, output_data, output_stride, dims, numel); \
    break
        CASE_RANK(1);
        CASE_RANK(2);
        CASE_RANK(3);
        CASE_RANK(4);
        CASE_RANK(5);
        CASE_RANK(6);
        CASE_RANK(7);
        CASE_RANK(8);
        CASE_RANK(9);
#undef CASE_RANK

        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "The rank of input should be less than 9, but received %d.",
              rank));
      }
    }
  } else {
    if (VecSize == 4) {
      switch (rank) {
#define CASE_RANK(__Rk)                                           \
  case __Rk:                                                      \
    Contiguous2StridedDefaultFunc<T, 4, __Rk>                     \
        <<<grid, block, 0, dev_ctx.stream()>>>(                   \
            input_data, output_data, output_stride, dims, numel); \
    break

        CASE_RANK(1);
        CASE_RANK(2);
        CASE_RANK(3);
        CASE_RANK(4);
        CASE_RANK(5);
        CASE_RANK(6);
        CASE_RANK(7);
        CASE_RANK(8);
        CASE_RANK(9);
#undef CASE_RANK
        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "The rank of input should be less than 9, but received %d.",
              rank));
      }
    } else if (VecSize == 2) {
      switch (rank) {
#define CASE_RANK(__Rk)                                           \
  case __Rk:                                                      \
    Contiguous2StridedDefaultFunc<T, 2, __Rk>                     \
        <<<grid, block, 0, dev_ctx.stream()>>>(                   \
            input_data, output_data, output_stride, dims, numel); \
    break

        CASE_RANK(1);
        CASE_RANK(2);
        CASE_RANK(3);
        CASE_RANK(4);
        CASE_RANK(5);
        CASE_RANK(6);
        CASE_RANK(7);
        CASE_RANK(8);
        CASE_RANK(9);
#undef CASE_RANK
        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "The rank of input should be less than 9, but received %d.",
              rank));
      }
    } else {
      switch (rank) {
#define CASE_RANK(__Rk)                                           \
  case __Rk:                                                      \
    Contiguous2StridedDefaultFunc<T, 1, __Rk>                     \
        <<<grid, block, 0, dev_ctx.stream()>>>(                   \
            input_data, output_data, output_stride, dims, numel); \
    break

        CASE_RANK(1);
        CASE_RANK(2);
        CASE_RANK(3);
        CASE_RANK(4);
        CASE_RANK(5);
        CASE_RANK(6);
        CASE_RANK(7);
        CASE_RANK(8);
        CASE_RANK(9);
#undef CASE_RANK
        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "The rank of input should be less than 9, but received %d.",
              rank));
      }
    }
  }
}

template <typename T, typename Context, int VecSize>
void StrideCopyDiffDimKernel(
    const Context& dev_ctx,
    const T* input_data,
    T* output_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_stride,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_dims,
    int rank,
    int numel) {
  if (LaunchContiguous2StridedCaseZeroKernel<T, Context>(dev_ctx,
                                                         input_data,
                                                         output_data,
                                                         output_stride,
                                                         output_dims,
                                                         rank,
                                                         true)) {
  } else if (LaunchContiguous2StridedCaseOneKernel<T, Context>(dev_ctx,
                                                               input_data,
                                                               output_data,
                                                               output_stride,
                                                               output_dims,
                                                               rank,
                                                               numel,
                                                               true)) {
  } else {
    switch (VecSize) {
#define CASE_VECSIZE(__Sz)                                                 \
  case __Sz:                                                               \
    LaunchContiguous2StridedDefaultKernel<T, Context, __Sz>(dev_ctx,       \
                                                            input_data,    \
                                                            output_data,   \
                                                            output_stride, \
                                                            output_dims,   \
                                                            rank,          \
                                                            numel,         \
                                                            true);         \
    break;
      CASE_VECSIZE(1);
      CASE_VECSIZE(2);
      CASE_VECSIZE(4);
#undef CASE_VECSIZE
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "unsurport vecsize %d for StrideCopyDiffDimKernel", VecSize));
    }
  }
}

}  // namespace phi
