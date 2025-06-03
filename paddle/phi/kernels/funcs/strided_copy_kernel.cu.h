/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may nint64_t use this file except in compliance with the License.
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

__global__ void GetOutOffsetKernel(const int64_t* __restrict__ output_stride,
                                   const int64_t* __restrict__ dims,
                                   int64_t numel,
                                   size_t rank,
                                   int64_t* offset) {
  // 计算全局索引
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 每个线程处理多个任务（向量化）
  for (int64_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
    int64_t index_tmp = i;
    int64_t offset_val = 0;

    // 计算偏移量
    for (int dim = rank - 1; dim >= 0; --dim) {
      offset_val += (index_tmp % dims[dim]) * output_stride[dim];
      index_tmp /= dims[dim];
    }

    // 写入结果
    offset[i] = offset_val;
  }
}

template <typename Context>
void GetOutOffset(const Context& dev_ctx,
                  const Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
                  const Array<int64_t, phi::DDim::kMaxRank + 1> dims,
                  int64_t numel,
                  int rank,
                  DenseTensor* offset) {
  // 设置输出张量的维度
  offset->Resize(common::make_ddim({numel}));

  // 分配 GPU 内存
  dev_ctx.template Alloc<int64_t>(offset);
  auto offset_data = offset->data<int64_t>();
  // 定义 CUDA Kernel 的线程块和网格大小
  int blockSize = 256;  // 每个线程块的线程数
  int gridSize = (numel + blockSize - 1) / blockSize;  // 网格大小

  // 启动 CUDA Kernel
  GetOutOffsetKernel<<<gridSize, blockSize>>>(
      output_stride.Get(), dims.Get(), numel, rank, offset_data);

  // 检查 CUDA 错误
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
}

template <typename T, size_t RANK>
__global__ void OnedimContiguous2StridedCaseZeroFunc(
    const T* input_data,
    T* output_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride) {
  int64_t input_offset = 0;
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

template <typename T, typename Context>
bool LaunchContiguous2StridedCaseZerOnedimKernel(
    const Context& dev_ctx,
    const T* input_data,
    T* output_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_stride,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& dims,
    int rank) {
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

  switch (rank) {
    case 1:
      OnedimContiguous2StridedCaseZeroFunc<T, 1>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data, output_data, output_stride);
      break;
    case 2:
      OnedimContiguous2StridedCaseZeroFunc<T, 2>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data, output_data, output_stride);
      break;
    case 3:
      OnedimContiguous2StridedCaseZeroFunc<T, 3>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data, output_data, output_stride);
      break;
    case 4:
      OnedimContiguous2StridedCaseZeroFunc<T, 4>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data, output_data, output_stride);
      break;
    case 5:
      OnedimContiguous2StridedCaseZeroFunc<T, 5>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data, output_data, output_stride);
      break;
    case 6:
      OnedimContiguous2StridedCaseZeroFunc<T, 6>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data, output_data, output_stride);
      break;
  }

  return true;
}

template <typename T, size_t N>
__global__ void OnedimContiguous2StridedCaseOneFunc(
    const T* input_data,
    T* out_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    phi::Array<int64_t, 6> dims,
    const int64_t x_max) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < x_max) {
    int64_t input_offset = 0;
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

template <typename T, typename Context>
bool LaunchContiguous2StridedCaseOneOnedimKernel(
    const Context& dev_ctx,
    const T* input_data,
    T* output_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_stride,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& dims,
    int rank,
    int64_t numel) {
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

  switch (rank) {
    case 1:
      OnedimContiguous2StridedCaseOneFunc<T, 1>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data, output_data, output_stride, cur_dims, dims[rank - 1]);
      break;
    case 2:
      OnedimContiguous2StridedCaseOneFunc<T, 2>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              output_data,
              output_stride,
              cur_dims,
              dims[rank - 1] * dims[rank - 2]);
      break;
    case 3:
      OnedimContiguous2StridedCaseOneFunc<T, 3>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              output_data,
              output_stride,
              cur_dims,
              dims[rank - 1] * dims[rank - 2] * dims[rank - 3]);
      break;
    case 4:
      OnedimContiguous2StridedCaseOneFunc<T, 4>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              output_data,
              output_stride,
              cur_dims,
              dims[rank - 1] * dims[rank - 2] * dims[rank - 3]);
      break;
    case 5:
      OnedimContiguous2StridedCaseOneFunc<T, 5>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              output_data,
              output_stride,
              cur_dims,
              dims[rank - 1] * dims[rank - 2] * dims[rank - 3]);
      break;
    case 6:
      OnedimContiguous2StridedCaseOneFunc<T, 6>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              output_data,
              output_stride,
              cur_dims,
              dims[rank - 1] * dims[rank - 2] * dims[rank - 3]);
      break;
    case 7:
      OnedimContiguous2StridedCaseOneFunc<T, 7>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              output_data,
              output_stride,
              cur_dims,
              dims[rank - 1] * dims[rank - 2] * dims[rank - 3]);
      break;
    case 8:
      OnedimContiguous2StridedCaseOneFunc<T, 8>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              output_data,
              output_stride,
              cur_dims,
              dims[rank - 1] * dims[rank - 2] * dims[rank - 3]);
      break;
    case 9:
      OnedimContiguous2StridedCaseOneFunc<T, 9>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              output_data,
              output_stride,
              cur_dims,
              dims[rank - 1] * dims[rank - 2] * dims[rank - 3]);
      break;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "The rank of input should be less than 9, but received %d.", rank));
  }

  return true;
}

template <typename T, int VecSize, typename OFFSET_T>
__global__ void OnedimContiguous2StridedDefaultFunc(
    const T* input_data,
    T* output_data,
    OFFSET_T* offset_data,
    Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    Array<int64_t, phi::DDim::kMaxRank + 1> dims,
    const int64_t numel) {
  int64_t gid = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
#pragma unroll
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x * VecSize) {
    int64_t output_offset = offset_data[i];
    using VecType = kps::details::VectorType<T, VecSize>;
    const VecType* src = reinterpret_cast<const VecType*>(&input_data[0]);
    VecType* dst = reinterpret_cast<VecType*>(&output_data[output_offset]);
    *dst = *src;
  }
}

template <typename T, typename Context>
void LaunchContiguous2StridedDefaultOnedimKernel(
    const Context& dev_ctx,
    const T* input_data,
    T* output_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_stride,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& dims,
    int rank,
    int64_t numel) {
  // int64_t block = 512;
  // int64_t grid = (numel + block - 1) / block;

  int vec_size = 4;

  vec_size = std::min(phi::GetVectorizedSize<T>(input_data), vec_size);
  vec_size = std::min(phi::GetVectorizedSize<T>(output_data), vec_size);
  while (vec_size > 1 && numel % vec_size != 0) {
    vec_size /= 2;
  }

  constexpr int loop_count = 4;
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, numel, vec_size * loop_count);
  auto& grid = config.block_per_grid;
  auto& block = config.thread_per_block;

  DenseTensor offset;
  GetOutOffset<Context>(dev_ctx, output_stride, dims, numel, rank, &offset);
  int64_t* offset_data = offset.data<int64_t>();

  switch (vec_size) {
#define CASE_VEC_SIZE(__Sz)                                                    \
  case __Sz:                                                                   \
    OnedimContiguous2StridedDefaultFunc<T, __Sz, int64_t>                      \
        <<<grid, block, 0, dev_ctx.stream()>>>(                                \
            input_data, output_data, offset_data, output_stride, dims, numel); \
    break
    CASE_VEC_SIZE(4);
    CASE_VEC_SIZE(2);
    CASE_VEC_SIZE(1);
#undef CASE_VEC_SIZE
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "The rank of input should be less than 9, but received %d.", rank));
  }
}

template <typename T, size_t OUT_RANK, int VecSize>
__global__ void Contiguous2StridedDefaultFuncV2(
    const T* input_data,
    T* output_data,
    Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    Array<int64_t, phi::DDim::kMaxRank + 1> dims,
    const int64_t numel) {
  int64_t gid = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
#pragma unroll
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x * VecSize) {
    int64_t output_offset = 0;
    int64_t index_tmp = i;

    for (int dim = OUT_RANK - 1; dim >= 0; --dim) {
      output_offset += (index_tmp % dims[dim]) * output_stride[dim];
      index_tmp = index_tmp / dims[dim];
    }

    using VecType = kps::details::VectorType<T, VecSize>;
    const VecType* src = reinterpret_cast<const VecType*>(&input_data[0]);
    VecType* dst = reinterpret_cast<VecType*>(&output_data[output_offset]);
    *dst = *src;
    // output_data[output_offset] = input_data[0];
  }
}

template <typename T, typename Context, int VecSize>
void LaunchContiguous2StridedDefaultKernelV2(
    const Context& dev_ctx,
    const T* input_data,
    T* output_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_stride,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& dims,
    int rank,
    int64_t numel) {
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;

  if (VecSize == 4) {
    switch (rank) {
#define CASE_RANK(__Rk)                                           \
  case __Rk:                                                      \
    Contiguous2StridedDefaultFuncV2<T, __Rk, 4>                   \
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
            "The rank of input should be less than 9, but received %d.", rank));
    }

  } else if (VecSize == 2) {
    switch (rank) {
#define CASE_RANK(__Rk)                                           \
  case __Rk:                                                      \
    Contiguous2StridedDefaultFuncV2<T, __Rk, 2>                   \
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
            "The rank of input should be less than 9, but received %d.", rank));
    }
  } else {
    switch (rank) {
#define CASE_RANK(__Rk)                                           \
  case __Rk:                                                      \
    Contiguous2StridedDefaultFuncV2<T, __Rk, 1>                   \
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
            "The rank of input should be less than 9, but received %d.", rank));
    }
  }
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("222 CUDA Error: %s\n", cudaGetErrorString(err));
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
  // if (LaunchContiguous2StridedCaseZerOnedimKernel<T, Context>(
  //         dev_ctx, input_data, output_data, output_stride, output_dims,
  //         rank)) {
  // } else if (LaunchContiguous2StridedCaseOneOnedimKernel<T, Context>(
  //                dev_ctx,
  //                input_data,
  //                output_data,
  //                output_stride,
  //                output_dims,
  //                rank,
  //                numel)) {
  // } else {
  switch (VecSize) {
    case 1:
      LaunchContiguous2StridedDefaultKernelV2<T, Context, 1>(dev_ctx,
                                                             input_data,
                                                             output_data,
                                                             output_stride,
                                                             output_dims,
                                                             rank,
                                                             numel);

      break;
    case 2:
      LaunchContiguous2StridedDefaultKernelV2<T, Context, 2>(dev_ctx,
                                                             input_data,
                                                             output_data,
                                                             output_stride,
                                                             output_dims,
                                                             rank,
                                                             numel);
      break;
    case 4:
      LaunchContiguous2StridedDefaultKernelV2<T, Context, 4>(dev_ctx,
                                                             input_data,
                                                             output_data,
                                                             output_stride,
                                                             output_dims,
                                                             rank,
                                                             numel);
      break;

    default:
      PADDLE_THROW(common::errors::InvalidArgument("ttttttttt"));
  }
  return;
  // LaunchContiguous2StridedDefaultOnedimKernel<T, Context>(dev_ctx,
  //                                                         input_data,
  //                                                         output_data,
  //                                                         output_stride,
  //                                                         output_dims,
  //                                                         rank,
  //                                                         numel);
  //}
}

}  // namespace phi
