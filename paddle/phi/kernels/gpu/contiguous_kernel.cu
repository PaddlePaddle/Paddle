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

template <typename T>
__global__ void ContiguousFunc(
    const T* input_data,
    T* out_data,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride,
    phi::Array<int64_t, phi::DDim::kMaxRank + 1> dims,
    const int rank,
    const int64_t numel) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
#pragma unroll
    for (int dim = rank - 1; dim >= 0; --dim) {
      input_offset += index_tmp % dims[dim] * input_stride[dim];
      index_tmp = index_tmp / dims[dim];
    }

    out_data[i] = input_data[input_offset];
  }
}

bool is_only_transposed(const DDim& shape,
                        const DDim& strides,
                        DDim& src_shape,           // NOLINT
                        DDim& src_strides,         // NOLINT
                        std::vector<int>& axis) {  // NOLINT
  std::set<int> visited_idx;
  axis.resize(strides.size());
  for (int i = 0; i < strides.size(); i++) {
    int64_t max_num = 0;
    int max_idx = -1;
    for (int j = 0; j < strides.size(); j++) {
      if (visited_idx.count(j)) {
        continue;
      }
      if (strides[j] < 1) {
        return false;
      }
      if (strides[j] > max_num) {
        max_num = strides[j];
        max_idx = j;
      }
    }
    if (max_idx == -1) {
      return false;
    }
    if (i != 0 && src_strides[i - 1] == max_num) {
      return false;
    }
    visited_idx.insert(max_idx);
    src_strides[i] = max_num;
    src_shape[i] = shape[max_idx];
    axis[max_idx] = i;
  }

  if (DenseTensorMeta::calc_strides(src_shape) == src_strides) {
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
  DDim src_strides = meta.strides;
  DDim src_shape = meta.dims;
  if (is_only_transposed(
          meta.dims, meta.strides, src_shape, src_strides, axis)) {
    meta.strides = meta.calc_strides(meta.dims, meta.layout);
    out->set_meta(meta);
    DenseTensor tmp_tensor = input;
    phi::DenseTensorMeta tmp_meta = meta;
    tmp_meta.strides = src_strides;
    tmp_meta.dims = src_shape;
    tmp_tensor.set_meta(tmp_meta);
    TransposeKernel<T, Context>(dev_ctx, tmp_tensor, axis, out);
    return;
  }

  meta.strides = meta.calc_strides(meta.dims, meta.layout);
  out->set_meta(meta);

  const T* input_data = input.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  int rank = input.dims().size();
  auto numel = input.numel();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;

  phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_stride;
  phi::Array<int64_t, phi::DDim::kMaxRank + 1> input_dims;
  for (int i = 0; i < input.dims().size(); i++) {
    input_dims[i] = input.dims()[i];
    input_stride[i] = input.strides()[i];
  }

  ContiguousFunc<<<grid, block, 0, dev_ctx.stream()>>>(
      input_data, output_data, input_stride, input_dims, rank, numel);
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
