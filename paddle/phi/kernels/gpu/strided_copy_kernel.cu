/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/strided_copy_kernel.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"
#include "paddle/phi/kernels/funcs/strided_copy_kernel.cu.h"

namespace phi {

template <typename T, size_t RANK>
__global__ void StridedCopyCaseZeroFunc(
    const T* input_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
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
    input_offset += coordinate[RANK - 1 - dim] * input_stride[dim];
    output_offset += coordinate[RANK - 1 - dim] * output_stride[dim];
  }

  output_data[output_offset] = input_data[input_offset];
}

template <typename T, typename Context>
bool LaunchStridedCopyCaseZeroKernel(
    const Context& dev_ctx,
    const T* input_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& input_stride,
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
      StridedCopyCaseZeroFunc<T, 1><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, input_stride, output_data, output_stride);
      break;
    case 2:
      StridedCopyCaseZeroFunc<T, 2><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, input_stride, output_data, output_stride);
      break;
    case 3:
      StridedCopyCaseZeroFunc<T, 3><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, input_stride, output_data, output_stride);
      break;
    case 4:
      StridedCopyCaseZeroFunc<T, 4><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, input_stride, output_data, output_stride);
      break;
    case 5:
      StridedCopyCaseZeroFunc<T, 5><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, input_stride, output_data, output_stride);
      break;
    case 6:
      StridedCopyCaseZeroFunc<T, 6><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, input_stride, output_data, output_stride);
      break;
  }

  return true;
}

template <typename T, size_t N>
__global__ void StridedCopyCaseOneFunc(
    const T* input_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
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
      input_offset += coordinate[N - 1 - dim] * input_stride[dim];
      output_offset += coordinate[N - 1 - dim] * output_stride[dim];
    }

    out_data[output_offset] = input_data[input_offset];
  }
}

template <typename T, typename Context>
bool LaunchStridedCopyCaseOneKernel(
    const Context& dev_ctx,
    const T* input_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& input_stride,
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
      StridedCopyCaseOneFunc<T, 1>
          <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                 input_stride,
                                                 output_data,
                                                 output_stride,
                                                 cur_dims,
                                                 dims[rank - 1]);
      break;
    case 2:
      StridedCopyCaseOneFunc<T, 2><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data,
          input_stride,
          output_data,
          output_stride,
          cur_dims,
          dims[rank - 1] * dims[rank - 2]);
      break;
#define CASE_RANK(__Rk)                                                    \
  case __Rk:                                                               \
    StridedCopyCaseOneFunc<T, __Rk><<<grid, block, 0, dev_ctx.stream()>>>( \
        input_data,                                                        \
        input_stride,                                                      \
        output_data,                                                       \
        output_stride,                                                     \
        cur_dims,                                                          \
        dims[rank - 1] * dims[rank - 2] * dims[rank - 3]);                 \
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

  return true;
}

template <typename T, size_t RANK>
__global__ void StridedCopyDefaultFunc(
    const T* input_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
    T* output_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> dims,
    const int64_t numel) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
#pragma unroll
    for (int dim = RANK - 1; dim >= 0; --dim) {
      input_offset += (index_tmp % dims[dim]) * input_stride[dim];
      index_tmp = index_tmp / dims[dim];
    }
    int64_t output_offset = 0;
    index_tmp = i;
#pragma unroll
    for (int dim = RANK - 1; dim >= 0; --dim) {
      output_offset += (index_tmp % dims[dim]) * output_stride[dim];
      index_tmp = index_tmp / dims[dim];
    }
    output_data[output_offset] = input_data[input_offset];
  }
}

template <typename T, typename Context>
void LaunchStridedCopyDefaultKernel(
    const Context& dev_ctx,
    const T* input_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& input_stride,
    T* output_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& output_stride,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& dims,
    int rank,
    int64_t numel) {
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;

  switch (rank) {
#define CASE_RANK(__Rk)                                                     \
  case __Rk:                                                                \
    StridedCopyDefaultFunc<T, __Rk><<<grid, block, 0, dev_ctx.stream()>>>(  \
        input_data, input_stride, output_data, output_stride, dims, numel); \
    break;
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

template <typename T, size_t RANK>
__global__ void Strided2ContiguousCaseZeroFunc(
    const T* input_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
    T* output_data) {
  int64_t input_offset = 0;
  int64_t output_offset = (blockIdx.z * gridDim.y * gridDim.x +
                           blockIdx.y * gridDim.x + blockIdx.x) *
                              blockDim.z * blockDim.y * blockDim.x +
                          threadIdx.z * blockDim.y * blockDim.x +
                          threadIdx.y * blockDim.x + threadIdx.x;
  int64_t coordinate[6] = {threadIdx.x,
                           threadIdx.y,
                           threadIdx.z,
                           blockIdx.x,
                           blockIdx.y,
                           blockIdx.z};

#pragma unroll
  for (int dim = RANK - 1; dim >= 0; --dim) {
    input_offset += coordinate[RANK - 1 - dim] * input_stride[dim];
  }

  output_data[output_offset] = input_data[input_offset];
}

template <typename T, typename Context>
bool LaunchStrided2ContiguousCaseZeroKernel(
    const Context& dev_ctx,
    const T* input_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& input_stride,
    T* output_data,
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
#define CASE_RANK(__Rk)                             \
  case __Rk:                                        \
    Strided2ContiguousCaseZeroFunc<T, __Rk>         \
        <<<grid, block, 0, dev_ctx.stream()>>>(     \
            input_data, input_stride, output_data); \
    break
    CASE_RANK(1);
    CASE_RANK(2);
    CASE_RANK(3);
    CASE_RANK(4);
    CASE_RANK(5);
    CASE_RANK(6);
#undef CASE_RANK
  }

  return true;
}

template <typename T, size_t N>
__global__ void Strided2ContiguousCaseOneFunc(
    const T* input_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
    T* out_data,
    phi::Array<int64_t, 6> dims,
    const int64_t x_max) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < x_max) {
    int64_t input_offset = 0;
    int64_t output_offset = (blockIdx.z * gridDim.y + blockIdx.y) * x_max + x;

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
      input_offset += coordinate[N - 1 - dim] * input_stride[dim];
    }

    out_data[output_offset] = input_data[input_offset];
  }
}

template <typename T, typename Context>
bool LaunchStrided2ContiguousCaseOneKernel(
    const Context& dev_ctx,
    const T* input_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& input_stride,
    T* output_data,
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
      Strided2ContiguousCaseOneFunc<T, 1><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, input_stride, output_data, cur_dims, dims[rank - 1]);
      break;
    case 2:
      Strided2ContiguousCaseOneFunc<T, 2><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data,
          input_stride,
          output_data,
          cur_dims,
          dims[rank - 1] * dims[rank - 2]);
      break;
#define CASE_RANK(__Rk)                                        \
  case __Rk:                                                   \
    Strided2ContiguousCaseOneFunc<T, __Rk>                     \
        <<<grid, block, 0, dev_ctx.stream()>>>(                \
            input_data,                                        \
            input_stride,                                      \
            output_data,                                       \
            cur_dims,                                          \
            dims[rank - 1] * dims[rank - 2] * dims[rank - 3]); \
    break
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

  return true;
}

template <typename T, size_t IN_RANK>
__global__ void Strided2ContiguousDefaultFunc(
    const T* input_data,
    Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
    T* output_data,
    Array<int64_t, phi::DDim::kMaxRank + 1> dims,
    const int64_t numel) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
#pragma unroll
    for (int dim = IN_RANK - 1; dim >= 0; --dim) {
      input_offset += (index_tmp % dims[dim]) * input_stride[dim];
      index_tmp = index_tmp / dims[dim];
    }
    output_data[i] = input_data[input_offset];
  }
}

template <typename T, typename Context>
void LaunchStrided2ContiguousDefaultKernel(
    const Context& dev_ctx,
    const T* input_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& input_stride,
    T* output_data,
    const phi::Array<int64_t, phi::DDim::kMaxRank + 1>& dims,
    int rank,
    int64_t numel) {
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;

  switch (rank) {
#define CASE_RANK(__Rk)                                          \
  case __Rk:                                                     \
    Strided2ContiguousDefaultFunc<T, __Rk>                       \
        <<<grid, block, 0, dev_ctx.stream()>>>(                  \
            input_data, input_stride, output_data, dims, numel); \
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

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = common::make_ddim(out_stride);
  meta.dims = common::make_ddim(dims);
  meta.offset = offset;
  out->set_meta(meta);
  int rank = out->dims().size();
  auto input_numel = input.numel();
  auto output_numel = out->numel();
  T* output_data = out->data<T>();
  PADDLE_ENFORCE_NOT_NULL(output_data,
                          common::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));
  Array<int64_t, phi::DDim::kMaxRank + 1> output_dims;
  Array<int64_t, phi::DDim::kMaxRank + 1> output_stride;
  for (int i = 0; i < meta.dims.size(); i++) {
    output_dims[i] = meta.dims[i];
    output_stride[i] = meta.strides[i];
  }

  const T* input_data = input.data<T>();

  // count vecsize
  int VecSize = 8;
  VecSize = std::min(phi::GetVectorizedSize<T>(input_data), VecSize);
  VecSize = std::min(phi::GetVectorizedSize<T>(output_data), VecSize);
  while (VecSize > 1 && output_numel % VecSize != 0) {
    VecSize /= 2;
  }
  if (input_numel != 1 && input_numel != output_numel) {
    while (VecSize > 1 && input_numel % VecSize != 0) {
      VecSize /= 2;
    }
  }
  while (VecSize > 1 && output_dims[meta.dims.size() - 1] % VecSize != 0) {
    VecSize /= 2;
  }
  if (output_stride[meta.dims.size() - 1] != 1) {
    VecSize = 1;
  }
  if (input.dims() != out->dims()) {
    if (input_numel == 1) {
      DenseTensor vec_input = Empty<T>(dev_ctx, IntArray{VecSize});
      ExpandKernel<T, Context>(dev_ctx, input, IntArray{VecSize}, &vec_input);
      const T* vec_input_data = vec_input.data<T>();
      switch (VecSize) {
#define CASE_VECSIZE(__Sz)                                    \
  case __Sz:                                                  \
    StrideCopyDiffDimKernel<T, Context, __Sz>(dev_ctx,        \
                                              vec_input_data, \
                                              output_data,    \
                                              output_stride,  \
                                              output_dims,    \
                                              rank,           \
                                              input_numel,    \
                                              output_numel);  \
    break;
        CASE_VECSIZE(1);
        CASE_VECSIZE(2);
        CASE_VECSIZE(4);
        CASE_VECSIZE(8);
#undef CASE_VECSIZE
        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "unsurport vecsize %d for StrideCopyDiffDimKernel", VecSize));
      }
      return;
    } else {
      bool can_expand =
          phi::funcs::CheckIsLastDimsMatch(input.dims(), out->dims());
      if (can_expand && input.meta().is_contiguous()) {
        switch (VecSize) {
#define CASE_VECSIZE(__Sz)                                                 \
  case __Sz:                                                               \
    LaunchContiguous2StridedDefaultKernel<T, Context, __Sz>(dev_ctx,       \
                                                            input_data,    \
                                                            output_data,   \
                                                            output_stride, \
                                                            output_dims,   \
                                                            rank,          \
                                                            input_numel,   \
                                                            output_numel,  \
                                                            false);        \
    break;
          CASE_VECSIZE(1);
          CASE_VECSIZE(2);
          CASE_VECSIZE(4);
          CASE_VECSIZE(8);
#undef CASE_VECSIZE
          default:
            PADDLE_THROW(common::errors::InvalidArgument(
                "unsurport vecsize %d for "
                "LaunchContiguous2StridedDefaultKernel",
                VecSize));
        }
        return;
      }
    }
  }
  PADDLE_ENFORCE_EQ(input.dims(),
                    out->dims(),
                    common::errors::InvalidArgument(
                        "Input shape(%s) must be equal with out shape(%s).",
                        input.dims(),
                        out->dims()));

  PADDLE_ENFORCE_EQ(input_numel,
                    output_numel,
                    common::errors::InvalidArgument(
                        "Input numel(%d) must be equal with out numel(%d).",
                        input_numel,
                        output_numel));

  Array<int64_t, phi::DDim::kMaxRank + 1> input_dims;
  Array<int64_t, phi::DDim::kMaxRank + 1> input_stride;
  for (int i = 0; i < input.dims().size(); i++) {
    input_dims[i] = input.dims()[i];
    input_stride[i] = input.strides()[i];
  }

  if (output_numel == 1) {
#ifdef PADDLE_WITH_HIP
    hipMemcpy(output_data,
              input_data,
              phi::SizeOf(input.dtype()),
              hipMemcpyDeviceToDevice);
#else
    cudaMemcpy(output_data,
               input_data,
               phi::SizeOf(input.dtype()),
               cudaMemcpyDeviceToDevice);
#endif

    return;
  }

  if (input.meta().is_contiguous()) {
    if (LaunchContiguous2StridedCaseZeroKernel<T, Context>(dev_ctx,
                                                           input_data,
                                                           output_data,
                                                           output_stride,
                                                           output_dims,
                                                           rank,
                                                           false)) {
    } else if (LaunchContiguous2StridedCaseOneKernel<T, Context>(dev_ctx,
                                                                 input_data,
                                                                 output_data,
                                                                 output_stride,
                                                                 output_dims,
                                                                 rank,
                                                                 output_numel,
                                                                 false)) {
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
                                                            input_numel,   \
                                                            output_numel,  \
                                                            false);        \
    break;
        CASE_VECSIZE(1);
        CASE_VECSIZE(2);
        CASE_VECSIZE(4);
        CASE_VECSIZE(8);
#undef CASE_VECSIZE
        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "unsurport vecsize %d for StrideCopyKernel", VecSize));
      }
    }
  } else if (out->meta().is_contiguous()) {
    if (LaunchStrided2ContiguousCaseZeroKernel<T, Context>(
            dev_ctx, input_data, input_stride, output_data, input_dims, rank)) {
    } else if (LaunchStrided2ContiguousCaseOneKernel<T, Context>(
                   dev_ctx,
                   input_data,
                   input_stride,
                   output_data,
                   input_dims,
                   rank,
                   output_numel)) {
    } else {
      LaunchStrided2ContiguousDefaultKernel<T, Context>(dev_ctx,
                                                        input_data,
                                                        input_stride,
                                                        output_data,
                                                        input_dims,
                                                        rank,
                                                        output_numel);
    }
  } else {
    if (LaunchStridedCopyCaseZeroKernel<T, Context>(dev_ctx,
                                                    input_data,
                                                    input_stride,
                                                    output_data,
                                                    output_stride,
                                                    input_dims,
                                                    rank)) {
    } else if (LaunchStridedCopyCaseOneKernel<T, Context>(dev_ctx,
                                                          input_data,
                                                          input_stride,
                                                          output_data,
                                                          output_stride,
                                                          input_dims,
                                                          rank,
                                                          output_numel)) {
    } else {
      LaunchStridedCopyDefaultKernel<T, Context>(dev_ctx,
                                                 input_data,
                                                 input_stride,
                                                 output_data,
                                                 output_stride,
                                                 input_dims,
                                                 rank,
                                                 output_numel);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(strided_copy,
                   GPU,
                   ALL_LAYOUT,
                   phi::StridedCopyKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   ::phi::dtype::float16,
                   ::phi::dtype::bfloat16,
                   ::phi::dtype::complex<float>,
                   ::phi::dtype::complex<double>,
                   ::phi::dtype::float8_e4m3fn,
                   ::phi::dtype::float8_e5m2) {}
