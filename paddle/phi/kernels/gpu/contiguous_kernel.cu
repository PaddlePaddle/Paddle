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

#include "paddle/phi/kernels/contiguous_kernel.h"

#include <set>

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
template <typename T, size_t N>
__global__ void ContiguousCaseZeroFunc(
    const T* input_data,
    T* out_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride) {
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
  for (int dim = N - 1; dim >= 0; --dim) {
    input_offset += coordinate[N - 1 - dim] * input_stride[dim];
  }

  out_data[output_offset] = input_data[input_offset];
}

template <typename T, size_t N>
__global__ void ContiguousCaseOneFunc(
    const T* input_data,
    T* out_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
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

bool is_only_transposed(const DDim& shape,
                        const DDim& stride,
                        uint64_t offset,
                        DDim& src_shape,           // NOLINT
                        DDim& src_stride,          // NOLINT
                        std::vector<int>& axis) {  // NOLINT
  if (offset != 0) {
    return false;
  }
  std::set<int> visited_idx;
  axis.resize(stride.size());
  for (int i = 0; i < stride.size(); i++) {
    int64_t max_num = 0;
    int max_idx = -1;
    for (int j = 0; j < stride.size(); j++) {
      if (visited_idx.count(j)) {
        continue;
      }
      if (stride[j] < 1) {
        return false;
      }
      if (stride[j] > max_num) {
        max_num = stride[j];
        max_idx = j;
      }
    }
    if (max_idx == -1) {
      return false;
    }
    if (i != 0 && src_stride[i - 1] == max_num) {
      return false;
    }
    visited_idx.insert(max_idx);
    src_stride[i] = max_num;
    src_shape[i] = shape[max_idx];
    axis[max_idx] = i;
  }

  if (DenseTensorMeta::calc_strides(src_shape) == src_stride) {
    return true;
  } else {
    return false;
  }
}

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  std::vector<int> axis;
  DDim src_stride = meta.strides;
  DDim src_shape = meta.dims;
  if (is_only_transposed(
          meta.dims, meta.strides, meta.offset, src_shape, src_stride, axis)) {
    meta.strides = meta.calc_strides(meta.dims);
    out->set_meta(meta);
    DenseTensor tmp_tensor = input;
    phi::DenseTensorMeta tmp_meta = meta;
    tmp_meta.strides = src_stride;
    tmp_meta.dims = src_shape;
    tmp_tensor.set_meta(tmp_meta);
    TransposeKernel<T, Context>(dev_ctx, tmp_tensor, axis, out);
    return;
  }

  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);

  const T* input_data = input.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  int rank = input.dims().size();
  auto numel = input.numel();

  if (numel <= 0) {
    return;
  }

  phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride;
  phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_dims;
  for (int i = 0; i < input.dims().size(); i++) {
    input_dims[i] = input.dims()[i];
    input_stride[i] = input.strides()[i];
  }

  if (rank == 0) {
    rank = 1;
    input_dims[0] = numel;
    input_stride[0] = 1;
  }

  dim3 grid(1, 1, 1), block(1, 1, 1);

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

    switch (rank) {
      case 1:
        ContiguousCaseZeroFunc<T, 1><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data, output_data, input_stride);
        break;
      case 2:
        ContiguousCaseZeroFunc<T, 2><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data, output_data, input_stride);
        break;
      case 3:
        ContiguousCaseZeroFunc<T, 3><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data, output_data, input_stride);
        break;
      case 4:
        grid.x = input_dims[rank - 4];
        ContiguousCaseZeroFunc<T, 4><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data, output_data, input_stride);
        break;
      case 5:
        grid.x = input_dims[rank - 4];
        grid.y = input_dims[rank - 5];
        ContiguousCaseZeroFunc<T, 5><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data, output_data, input_stride);
        break;
      case 6:
        grid.x = input_dims[rank - 4];
        grid.y = input_dims[rank - 5];
        grid.z = input_dims[rank - 6];
        ContiguousCaseZeroFunc<T, 6><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data, output_data, input_stride);
        break;
    }
  } else {
    phi::Array<int64_t, 6> cur_input_dims;
    block.x = 512;
    switch (rank) {
      case 1:
        grid.x = (numel + block.x - 1) / block.x;
        cur_input_dims[0] = input_dims[rank - 1];
        ContiguousCaseOneFunc<T, 1>
            <<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                   output_data,
                                                   input_stride,
                                                   cur_input_dims,
                                                   input_dims[rank - 1]);
        break;
      case 2:
        grid.x = (numel + block.x - 1) / block.x;
        cur_input_dims[0] = input_dims[rank - 1];
        cur_input_dims[1] = input_dims[rank - 2];
        ContiguousCaseOneFunc<T, 2><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data,
            output_data,
            input_stride,
            cur_input_dims,
            input_dims[rank - 1] * input_dims[rank - 2]);
        break;
      case 3:
        grid.x = (numel + block.x - 1) / block.x;
        cur_input_dims[0] = input_dims[rank - 1];
        cur_input_dims[1] = input_dims[rank - 2];
        ContiguousCaseOneFunc<T, 3><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data,
            output_data,
            input_stride,
            cur_input_dims,
            input_dims[rank - 1] * input_dims[rank - 2] * input_dims[rank - 3]);
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
        ContiguousCaseOneFunc<T, 4><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data,
            output_data,
            input_stride,
            cur_input_dims,
            input_dims[rank - 1] * input_dims[rank - 2] * input_dims[rank - 3]);
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
        ContiguousCaseOneFunc<T, 5><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data,
            output_data,
            input_stride,
            cur_input_dims,
            input_dims[rank - 1] * input_dims[rank - 2] * input_dims[rank - 3]);
        break;
      case 6:
        grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                      input_dims[rank - 3] +
                  block.x - 1) /
                 block.x;
        grid.y =
            input_dims[rank - 4] * input_dims[rank - 5] * input_dims[rank - 6];
        cur_input_dims[0] = input_dims[rank - 1];
        cur_input_dims[1] = input_dims[rank - 2];
        cur_input_dims[2] = input_dims[rank - 4];
        cur_input_dims[3] = input_dims[rank - 5];
        ContiguousCaseOneFunc<T, 6><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data,
            output_data,
            input_stride,
            cur_input_dims,
            input_dims[rank - 1] * input_dims[rank - 2] * input_dims[rank - 3]);
        break;
      case 7:
        grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                      input_dims[rank - 3] +
                  block.x - 1) /
                 block.x;
        grid.y =
            input_dims[rank - 4] * input_dims[rank - 5] * input_dims[rank - 6];
        grid.z = input_dims[rank - 7];
        cur_input_dims[0] = input_dims[rank - 1];
        cur_input_dims[1] = input_dims[rank - 2];
        cur_input_dims[2] = input_dims[rank - 4];
        cur_input_dims[3] = input_dims[rank - 5];
        cur_input_dims[4] = input_dims[rank - 7];
        ContiguousCaseOneFunc<T, 7><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data,
            output_data,
            input_stride,
            cur_input_dims,
            input_dims[rank - 1] * input_dims[rank - 2] * input_dims[rank - 3]);
        break;
      case 8:
        grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                      input_dims[rank - 3] +
                  block.x - 1) /
                 block.x;
        grid.y =
            input_dims[rank - 4] * input_dims[rank - 5] * input_dims[rank - 6];
        grid.z = input_dims[rank - 7] * input_dims[rank - 8];
        cur_input_dims[0] = input_dims[rank - 1];
        cur_input_dims[1] = input_dims[rank - 2];
        cur_input_dims[2] = input_dims[rank - 4];
        cur_input_dims[3] = input_dims[rank - 5];
        cur_input_dims[4] = input_dims[rank - 7];
        cur_input_dims[5] = input_dims[rank - 8];
        ContiguousCaseOneFunc<T, 8><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data,
            output_data,
            input_stride,
            cur_input_dims,
            input_dims[rank - 1] * input_dims[rank - 2] * input_dims[rank - 3]);
        break;
      case 9:
        grid.x = (input_dims[rank - 1] * input_dims[rank - 2] *
                      input_dims[rank - 3] +
                  block.x - 1) /
                 block.x;
        grid.y =
            input_dims[rank - 4] * input_dims[rank - 5] * input_dims[rank - 6];
        grid.z =
            input_dims[rank - 7] * input_dims[rank - 8] * input_dims[rank - 9];
        cur_input_dims[0] = input_dims[rank - 1];
        cur_input_dims[1] = input_dims[rank - 2];
        cur_input_dims[2] = input_dims[rank - 4];
        cur_input_dims[3] = input_dims[rank - 5];
        cur_input_dims[4] = input_dims[rank - 7];
        cur_input_dims[5] = input_dims[rank - 8];
        ContiguousCaseOneFunc<T, 9><<<grid, block, 0, dev_ctx.stream()>>>(
            input_data,
            output_data,
            input_stride,
            cur_input_dims,
            input_dims[rank - 1] * input_dims[rank - 2] * input_dims[rank - 3]);
        break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "The rank of input should be less than 9, but received %d.", rank));
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(contiguous,
                   GPU,
                   ALL_LAYOUT,
                   phi::ContiguousKernel,
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
