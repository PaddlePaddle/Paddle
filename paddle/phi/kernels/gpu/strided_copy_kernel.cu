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
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, size_t IN_RANK, size_t OUT_RANK>
__global__ void StridedCopyFunc(
    const T* input_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_dims,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
    T* output_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_dims,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    const int64_t numel) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
#pragma unroll
    for (int dim = IN_RANK - 1; dim >= 0; --dim) {
      input_offset += (index_tmp % input_dims[dim]) * input_stride[dim];
      index_tmp = index_tmp / input_dims[dim];
    }
    int64_t output_offset = 0;
    index_tmp = i;
#pragma unroll
    for (int dim = OUT_RANK - 1; dim >= 0; --dim) {
      output_offset += (index_tmp % output_dims[dim]) * output_stride[dim];
      index_tmp = index_tmp / output_dims[dim];
    }
    output_data[output_offset] = input_data[input_offset];
  }
}

template <typename T, size_t RANK>
__global__ void StridedCopyCaseZeroFunc(
    const T* input_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
    T* output_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride) {
  int64_t input_offset = (blockIdx.z * gridDim.y * gridDim.x +
                          blockIdx.y * gridDim.x + blockIdx.x) *
                             blockDim.z * blockDim.y * blockDim.x +
                         threadIdx.z * blockDim.y * blockDim.x +
                         threadIdx.y * blockDim.x + threadIdx.x;
  int64_t output_offset = input_offset;
  float coordinate[6] = {threadIdx.x,
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
    int64_t input_offset = (blockIdx.z * gridDim.y + blockIdx.y) * x_max + x;
    int64_t output_offset = input_offset;

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

template <typename T, size_t IN_RANK>
__global__ void Strided2ContiguousFunc(
    const T* input_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_dims,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
    T* output_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_dims,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    const int64_t numel) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
#pragma unroll
    for (int dim = IN_RANK - 1; dim >= 0; --dim) {
      input_offset += (index_tmp % input_dims[dim]) * input_stride[dim];
      index_tmp = index_tmp / input_dims[dim];
    }
    output_data[i] = input_data[input_offset];
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
  float coordinate[6] = {threadIdx.x,
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

template <typename T, size_t OUT_RANK>
__global__ void Contiguous2StridedFunc(
    const T* input_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_dims,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
    T* output_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_dims,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride,
    const int64_t numel) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x) {
    int64_t output_offset = 0;
    int64_t index_tmp = i;
#pragma unroll
    for (int dim = OUT_RANK - 1; dim >= 0; --dim) {
      output_offset += (index_tmp % output_dims[dim]) * output_stride[dim];
      index_tmp = index_tmp / output_dims[dim];
    }
    output_data[output_offset] = input_data[i];
  }
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
  float coordinate[6] = {threadIdx.x,
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

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = phi::make_ddim(out_stride);
  meta.dims = phi::make_ddim(dims);
  meta.offset = offset;
  out->set_meta(meta);

  PADDLE_ENFORCE_EQ(input.dims(),
                    out->dims(),
                    phi::errors::InvalidArgument(
                        "Input shape(%s) must be equal with out shape(%s).",
                        input.dims(),
                        out->dims()));

  PADDLE_ENFORCE_EQ(input.numel(),
                    out->numel(),
                    phi::errors::InvalidArgument(
                        "Input numel(%d) must be equal with out numel(%d).",
                        input.numel(),
                        out->numel()));

  const T* input_data = input.data<T>();
  int input_rank = input.dims().size();
  phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride;
  phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_dims;
  for (int i = 0; i < input.dims().size(); i++) {
    input_dims[i] = input.dims()[i];
    input_stride[i] = input.strides()[i];
  }

  T* output_data = out->data<T>();
  PADDLE_ENFORCE_NOT_NULL(output_data,
                          phi::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));

  int output_rank = meta.dims.size();
  phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_stride;
  phi::Array<int64_t, phi::DDim::kMaxRank + 1> output_dims;
  for (int i = 0; i < meta.dims.size(); i++) {
    output_dims[i] = meta.dims[i];
    output_stride[i] = meta.strides[i];
  }

  auto numel = input.numel();

  if (numel == 1) {
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

  dim3 grid(1, 1, 1), block(1, 1, 1);
  int rank = input_rank;
  int tmp = 1;

  for (int i = 0; i < 3 && i < rank; i++) {
    tmp *= input_dims[rank - 1 - i];
  }

  if (rank <= 6 && tmp <= 1024 &&
      (input_dims.size() < 3 || input_dims[rank - 3] <= 64)) {
    if (rank >= 1) {
      block.x = input_dims[rank - 1];
    }

    if (rank >= 2) {
      block.y = input_dims[rank - 2];
    }

    if (rank >= 3) {
      block.z = input_dims[rank - 3];
    }

    if (input.meta().is_contiguous()) {
      switch (rank) {
        case 1:
          Contiguous2StridedCaseZeroFunc<T, 1>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, output_data, output_stride);
          break;
        case 2:
          Contiguous2StridedCaseZeroFunc<T, 2>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, output_data, output_stride);
          break;
        case 3:
          Contiguous2StridedCaseZeroFunc<T, 3>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, output_data, output_stride);
          break;
        case 4:
          grid.x = input_dims[rank - 4];
          Contiguous2StridedCaseZeroFunc<T, 4>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, output_data, output_stride);
          break;
        case 5:
          grid.x = input_dims[rank - 4];
          grid.y = input_dims[rank - 5];
          Contiguous2StridedCaseZeroFunc<T, 5>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, output_data, output_stride);
          break;
        case 6:
          grid.x = input_dims[rank - 4];
          grid.y = input_dims[rank - 5];
          grid.z = input_dims[rank - 6];
          Contiguous2StridedCaseZeroFunc<T, 6>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, output_data, output_stride);
          break;
      }
    } else if (out->meta().is_contiguous()) {
      switch (rank) {
        case 1:
          Strided2ContiguousCaseZeroFunc<T, 1>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, input_stride, output_data);
          break;
        case 2:
          Strided2ContiguousCaseZeroFunc<T, 2>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, input_stride, output_data);
          break;
        case 3:
          Strided2ContiguousCaseZeroFunc<T, 3>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, input_stride, output_data);
          break;
        case 4:
          grid.x = input_dims[rank - 4];
          Strided2ContiguousCaseZeroFunc<T, 4>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, input_stride, output_data);
          break;
        case 5:
          grid.x = input_dims[rank - 4];
          grid.y = input_dims[rank - 5];
          Strided2ContiguousCaseZeroFunc<T, 5>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, input_stride, output_data);
          break;
        case 6:
          grid.x = input_dims[rank - 4];
          grid.y = input_dims[rank - 5];
          grid.z = input_dims[rank - 6];
          Strided2ContiguousCaseZeroFunc<T, 6>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data, input_stride, output_data);
          break;
      }
    } else {
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
          grid.x = input_dims[rank - 4];
          StridedCopyCaseZeroFunc<T, 4><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data, input_stride, output_data, output_stride);
          break;
        case 5:
          grid.x = input_dims[rank - 4];
          grid.y = input_dims[rank - 5];
          StridedCopyCaseZeroFunc<T, 5><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data, input_stride, output_data, output_stride);
          break;
        case 6:
          grid.x = input_dims[rank - 4];
          grid.y = input_dims[rank - 5];
          grid.z = input_dims[rank - 6];
          StridedCopyCaseZeroFunc<T, 6><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data, input_stride, output_data, output_stride);
          break;
      }
    }
  } else {
    phi::Array<int64_t, 6> cur_input_dims;
    block.x = 512;

    if (input.meta().is_contiguous()) {
      switch (rank) {
        case 1:
          grid.x = (numel + block.x - 1) / block.x;
          cur_input_dims[0] = input_dims[rank - 1];
          Contiguous2StridedCaseOneFunc<T, 1>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     output_data,
                                                     output_stride,
                                                     cur_input_dims,
                                                     input_dims[rank - 1]);
          break;
        case 2:
          grid.x = (numel + block.x - 1) / block.x;
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          Contiguous2StridedCaseOneFunc<T, 2>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data,
                  output_data,
                  output_stride,
                  cur_input_dims,
                  input_dims[rank - 1] * input_dims[rank - 2]);
          break;
        case 3:
          grid.x = (numel + block.x - 1) / block.x;
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          Contiguous2StridedCaseOneFunc<T, 3>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     output_data,
                                                     output_stride,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 4:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          Contiguous2StridedCaseOneFunc<T, 4>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     output_data,
                                                     output_stride,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 5:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          Contiguous2StridedCaseOneFunc<T, 5>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     output_data,
                                                     output_stride,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 6:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          Contiguous2StridedCaseOneFunc<T, 6>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     output_data,
                                                     output_stride,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 7:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          grid.z = input_dims[rank - 7];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          cur_input_dims[4] = input_dims[rank - 7];
          Contiguous2StridedCaseOneFunc<T, 7>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     output_data,
                                                     output_stride,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 8:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          grid.z = input_dims[rank - 7] * input_dims[rank - 8];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          cur_input_dims[4] = input_dims[rank - 7];
          cur_input_dims[5] = input_dims[rank - 8];
          Contiguous2StridedCaseOneFunc<T, 8>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     output_data,
                                                     output_stride,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 9:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          grid.z = input_dims[rank - 7] * input_dims[rank - 8] *
                   input_dims[rank - 9];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          cur_input_dims[4] = input_dims[rank - 7];
          cur_input_dims[5] = input_dims[rank - 8];
          Contiguous2StridedCaseOneFunc<T, 9>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     output_data,
                                                     output_stride,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        default:
          PADDLE_THROW(phi::errors::InvalidArgument(
              "The rank of input should be less than 9, but received %d.",
              rank));
      }
    } else if (out->meta().is_contiguous()) {
      switch (rank) {
        case 1:
          grid.x = (numel + block.x - 1) / block.x;
          cur_input_dims[0] = input_dims[rank - 1];
          Strided2ContiguousCaseOneFunc<T, 1>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     input_stride,
                                                     output_data,
                                                     cur_input_dims,
                                                     input_dims[rank - 1]);
          break;
        case 2:
          grid.x = (numel + block.x - 1) / block.x;
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          Strided2ContiguousCaseOneFunc<T, 2>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  input_data,
                  input_stride,
                  output_data,
                  cur_input_dims,
                  input_dims[rank - 1] * input_dims[rank - 2]);
          break;
        case 3:
          grid.x = (numel + block.x - 1) / block.x;
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          Strided2ContiguousCaseOneFunc<T, 3>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     input_stride,
                                                     output_data,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 4:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          Strided2ContiguousCaseOneFunc<T, 4>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     input_stride,
                                                     output_data,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 5:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          Strided2ContiguousCaseOneFunc<T, 5>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     input_stride,
                                                     output_data,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 6:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          Strided2ContiguousCaseOneFunc<T, 6>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     input_stride,
                                                     output_data,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 7:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          grid.z = input_dims[rank - 7];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          cur_input_dims[4] = input_dims[rank - 7];
          Strided2ContiguousCaseOneFunc<T, 7>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     input_stride,
                                                     output_data,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 8:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          grid.z = input_dims[rank - 7] * input_dims[rank - 8];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          cur_input_dims[4] = input_dims[rank - 7];
          cur_input_dims[5] = input_dims[rank - 8];
          Strided2ContiguousCaseOneFunc<T, 8>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     input_stride,
                                                     output_data,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        case 9:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          grid.z = input_dims[rank - 7] * input_dims[rank - 8] *
                   input_dims[rank - 9];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          cur_input_dims[4] = input_dims[rank - 7];
          cur_input_dims[5] = input_dims[rank - 8];
          Strided2ContiguousCaseOneFunc<T, 9>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     input_stride,
                                                     output_data,
                                                     cur_input_dims,
                                                     input_dims[rank - 1] *
                                                         input_dims[rank - 2] *
                                                         input_dims[rank - 3]);
          break;
        default:
          PADDLE_THROW(phi::errors::InvalidArgument(
              "The rank of input should be less than 9, but received %d.",
              rank));
      }
    } else {
      switch (rank) {
        case 1:
          grid.x = (numel + block.x - 1) / block.x;
          cur_input_dims[0] = input_dims[rank - 1];
          StridedCopyCaseOneFunc<T, 1>
              <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                     input_stride,
                                                     output_data,
                                                     output_stride,
                                                     cur_input_dims,
                                                     input_dims[rank - 1]);
          break;
        case 2:
          grid.x = (numel + block.x - 1) / block.x;
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          StridedCopyCaseOneFunc<T, 2><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              input_stride,
              output_data,
              output_stride,
              cur_input_dims,
              input_dims[rank - 1] * input_dims[rank - 2]);
          break;
        case 3:
          grid.x = (numel + block.x - 1) / block.x;
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          StridedCopyCaseOneFunc<T, 3><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              input_stride,
              output_data,
              output_stride,
              cur_input_dims,
              input_dims[rank - 1] * input_dims[rank - 2] *
                  input_dims[rank - 3]);
          break;
        case 4:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          StridedCopyCaseOneFunc<T, 4><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              input_stride,
              output_data,
              output_stride,
              cur_input_dims,
              input_dims[rank - 1] * input_dims[rank - 2] *
                  input_dims[rank - 3]);
          break;
        case 5:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          StridedCopyCaseOneFunc<T, 5><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              input_stride,
              output_data,
              output_stride,
              cur_input_dims,
              input_dims[rank - 1] * input_dims[rank - 2] *
                  input_dims[rank - 3]);
          break;
        case 6:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          StridedCopyCaseOneFunc<T, 6><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              input_stride,
              output_data,
              output_stride,
              cur_input_dims,
              input_dims[rank - 1] * input_dims[rank - 2] *
                  input_dims[rank - 3]);
          break;
        case 7:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          grid.z = input_dims[rank - 7];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          cur_input_dims[4] = input_dims[rank - 7];
          StridedCopyCaseOneFunc<T, 7><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              input_stride,
              output_data,
              output_stride,
              cur_input_dims,
              input_dims[rank - 1] * input_dims[rank - 2] *
                  input_dims[rank - 3]);
          break;
        case 8:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          grid.z = input_dims[rank - 7] * input_dims[rank - 8];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          cur_input_dims[4] = input_dims[rank - 7];
          cur_input_dims[5] = input_dims[rank - 8];
          StridedCopyCaseOneFunc<T, 8><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              input_stride,
              output_data,
              output_stride,
              cur_input_dims,
              input_dims[rank - 1] * input_dims[rank - 2] *
                  input_dims[rank - 3]);
          break;
        case 9:
          grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                        input_dims[rank - 3] +
                    block.x - 1) /
                   block.x;
          grid.y = input_dims[rank - 4] * input_dims[rank - 5] *
                   input_dims[rank - 6];
          grid.z = input_dims[rank - 7] * input_dims[rank - 8] *
                   input_dims[rank - 9];
          cur_input_dims[0] = input_dims[rank - 1];
          cur_input_dims[1] = input_dims[rank - 2];
          cur_input_dims[2] = input_dims[rank - 4];
          cur_input_dims[3] = input_dims[rank - 5];
          cur_input_dims[4] = input_dims[rank - 7];
          cur_input_dims[5] = input_dims[rank - 8];
          StridedCopyCaseOneFunc<T, 9><<<grid, block, 0, dev_ctx.stream()>>>(
              input_data,
              input_stride,
              output_data,
              output_stride,
              cur_input_dims,
              input_dims[rank - 1] * input_dims[rank - 2] *
                  input_dims[rank - 3]);
          break;
        default:
          PADDLE_THROW(phi::errors::InvalidArgument(
              "The rank of input should be less than 9, but received %d.",
              rank));
      }
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
                   ::phi::dtype::complex<double>) {}
