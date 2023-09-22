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
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;

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

  if (input.meta().is_contiguous()) {
    switch (input_rank) {
      case 1:
        Contiguous2StridedFunc<T, 1>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 2:
        Contiguous2StridedFunc<T, 2>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 3:
        Contiguous2StridedFunc<T, 3>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 4:
        Contiguous2StridedFunc<T, 4>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 5:
        Contiguous2StridedFunc<T, 5>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 6:
        Contiguous2StridedFunc<T, 6>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 7:
        Contiguous2StridedFunc<T, 7>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 8:
        Contiguous2StridedFunc<T, 8>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 9:
        Contiguous2StridedFunc<T, 9>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "The rank of input should be less than 9, but received %d.",
            input_rank));
    }
  } else if (out->meta().is_contiguous()) {
    switch (output_rank) {
      case 1:
        Strided2ContiguousFunc<T, 1>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 2:
        Strided2ContiguousFunc<T, 2>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 3:
        Strided2ContiguousFunc<T, 3>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 4:
        Strided2ContiguousFunc<T, 4>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 5:
        Strided2ContiguousFunc<T, 5>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 6:
        Strided2ContiguousFunc<T, 6>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 7:
        Strided2ContiguousFunc<T, 7>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 8:
        Strided2ContiguousFunc<T, 8>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      case 9:
        Strided2ContiguousFunc<T, 9>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   input_dims,
                                                   input_stride,
                                                   output_data,
                                                   output_dims,
                                                   output_stride,
                                                   numel);
        break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "The rank of output should be less than 9, but received %d.",
            output_rank));
    }
  } else {
    switch (input_rank) {
      case 1: {
        switch (output_rank) {
          case 1:
            StridedCopyFunc<T, 1, 1>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 2:
            StridedCopyFunc<T, 1, 2>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 3:
            StridedCopyFunc<T, 1, 3>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 4:
            StridedCopyFunc<T, 1, 4>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 5:
            StridedCopyFunc<T, 1, 5>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 6:
            StridedCopyFunc<T, 1, 6>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 7:
            StridedCopyFunc<T, 1, 7>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 8:
            StridedCopyFunc<T, 1, 8>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 9:
            StridedCopyFunc<T, 1, 9>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          default:
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The rank of output should be less than 9, but received %d.",
                output_rank));
        }
      } break;
      case 2: {
        switch (output_rank) {
          case 1:
            StridedCopyFunc<T, 2, 1>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 2:
            StridedCopyFunc<T, 2, 2>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 3:
            StridedCopyFunc<T, 2, 3>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 4:
            StridedCopyFunc<T, 2, 4>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 5:
            StridedCopyFunc<T, 2, 5>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 6:
            StridedCopyFunc<T, 2, 6>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 7:
            StridedCopyFunc<T, 2, 7>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 8:
            StridedCopyFunc<T, 2, 8>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 9:
            StridedCopyFunc<T, 2, 9>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          default:
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The rank of output should be less than 9, but received %d.",
                output_rank));
        }
      } break;
      case 3: {
        switch (output_rank) {
          case 1:
            StridedCopyFunc<T, 3, 1>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 2:
            StridedCopyFunc<T, 3, 2>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 3:
            StridedCopyFunc<T, 3, 3>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 4:
            StridedCopyFunc<T, 3, 4>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 5:
            StridedCopyFunc<T, 3, 5>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 6:
            StridedCopyFunc<T, 3, 6>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 7:
            StridedCopyFunc<T, 3, 7>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 8:
            StridedCopyFunc<T, 3, 8>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 9:
            StridedCopyFunc<T, 3, 9>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          default:
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The rank of output should be less than 9, but received %d.",
                output_rank));
        }
      } break;
      case 4: {
        switch (output_rank) {
          case 1:
            StridedCopyFunc<T, 4, 1>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 2:
            StridedCopyFunc<T, 4, 2>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 3:
            StridedCopyFunc<T, 4, 3>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 4:
            StridedCopyFunc<T, 4, 4>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 5:
            StridedCopyFunc<T, 4, 5>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 6:
            StridedCopyFunc<T, 4, 6>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 7:
            StridedCopyFunc<T, 4, 7>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 8:
            StridedCopyFunc<T, 4, 8>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 9:
            StridedCopyFunc<T, 4, 9>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          default:
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The rank of output should be less than 9, but received %d.",
                output_rank));
        }
      } break;
      case 5: {
        switch (output_rank) {
          case 1:
            StridedCopyFunc<T, 5, 1>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 2:
            StridedCopyFunc<T, 5, 2>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 3:
            StridedCopyFunc<T, 5, 3>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 4:
            StridedCopyFunc<T, 5, 4>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 5:
            StridedCopyFunc<T, 5, 5>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 6:
            StridedCopyFunc<T, 5, 6>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 7:
            StridedCopyFunc<T, 5, 7>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 8:
            StridedCopyFunc<T, 5, 8>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 9:
            StridedCopyFunc<T, 5, 9>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          default:
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The rank of output should be less than 9, but received %d.",
                output_rank));
        }
      } break;
      case 6: {
        switch (output_rank) {
          case 1:
            StridedCopyFunc<T, 6, 1>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 2:
            StridedCopyFunc<T, 6, 2>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 3:
            StridedCopyFunc<T, 6, 3>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 4:
            StridedCopyFunc<T, 6, 4>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 5:
            StridedCopyFunc<T, 6, 5>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 6:
            StridedCopyFunc<T, 6, 6>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 7:
            StridedCopyFunc<T, 6, 7>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 8:
            StridedCopyFunc<T, 6, 8>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 9:
            StridedCopyFunc<T, 6, 9>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          default:
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The rank of output should be less than 9, but received %d.",
                output_rank));
        }
      } break;
      case 7: {
        switch (output_rank) {
          case 1:
            StridedCopyFunc<T, 7, 1>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 2:
            StridedCopyFunc<T, 7, 2>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 3:
            StridedCopyFunc<T, 7, 3>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 4:
            StridedCopyFunc<T, 7, 4>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 5:
            StridedCopyFunc<T, 7, 5>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 6:
            StridedCopyFunc<T, 7, 6>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 7:
            StridedCopyFunc<T, 7, 7>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 8:
            StridedCopyFunc<T, 7, 8>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 9:
            StridedCopyFunc<T, 7, 9>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          default:
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The rank of output should be less than 9, but received %d.",
                output_rank));
        }
      } break;
      case 8: {
        switch (output_rank) {
          case 1:
            StridedCopyFunc<T, 8, 1>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 2:
            StridedCopyFunc<T, 8, 2>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 3:
            StridedCopyFunc<T, 8, 3>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 4:
            StridedCopyFunc<T, 8, 4>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 5:
            StridedCopyFunc<T, 8, 5>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 6:
            StridedCopyFunc<T, 8, 6>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 7:
            StridedCopyFunc<T, 8, 7>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 8:
            StridedCopyFunc<T, 8, 8>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 9:
            StridedCopyFunc<T, 8, 9>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          default:
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The rank of output should be less than 9, but received %d.",
                output_rank));
        }
      } break;
      case 9: {
        switch (output_rank) {
          case 1:
            StridedCopyFunc<T, 9, 1>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 2:
            StridedCopyFunc<T, 9, 2>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 3:
            StridedCopyFunc<T, 9, 3>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 4:
            StridedCopyFunc<T, 9, 4>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 5:
            StridedCopyFunc<T, 9, 5>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 6:
            StridedCopyFunc<T, 9, 6>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 7:
            StridedCopyFunc<T, 9, 7>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 8:
            StridedCopyFunc<T, 9, 8>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          case 9:
            StridedCopyFunc<T, 9, 9>
                <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       input_dims,
                                                       input_stride,
                                                       output_data,
                                                       output_dims,
                                                       output_stride,
                                                       numel);
            break;
          default:
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The rank of output should be less than 9, but received %d.",
                output_rank));
        }
      } break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "The rank of input should be less than 9, but received %d.",
            input_rank));
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
