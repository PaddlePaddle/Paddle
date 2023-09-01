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
#include "paddle/phi/kernels/funcs/segmented_array.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, size_t N>
__global__ void ContiguousFunc(
    const T* input_data,
    T* out_data,
    funcs::ValueArray<int64_t, funcs::SegmentedArraySize::kFixed16>
        input_stride,
    funcs::ValueArray<int64_t, funcs::SegmentedArraySize::kFixed16> input_dims,
    const int64_t numel) {
  int64_t element_num_each_thread = (numel / (blockDim.x * gridDim.x)) + 1;
  int64_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t start_offset = thread_index * element_num_each_thread;
  int64_t end_offset = start_offset + element_num_each_thread;
  int64_t start_dims[9];
  int64_t end_dims[9];
  int64_t strides[9];
  int64_t dims[9];
  int64_t index_tmp_start = start_offset;
  int64_t index_tmp_end = end_offset;
#pragma unroll
  for (int dim = 8; dim >= 0; --dim) {
    if (dim < N - 1) {
      start_dims[dim] = 0;
      end_dims[dim] = 0;
      strides[dim] = 0;
      dims[dim] = 0;
    } else {
      strides[dim] = input_stride.data[dim];
      dims[dim] = input_dims.data[dim];
      start_dims[dim] = index_tmp_start % input_dims.data[dim];
      end_dims[dim] = index_tmp_end % input_dims.data[dim];
      index_tmp_start = index_tmp_start / input_dims.data[dim];
      index_tmp_end = index_tmp_end / input_dims.data[dim];
    }
  }

  for (int64_t dim8 = start_dims[8]; dim8 >= end_dims[8]; ++dim8) {
    int64_t offset8 = strides[8] * dim8;
    int64_t end_dims7 = dim8 == end_dims[8] ? end_dims[7] : dims[7] - 1;
    for (int64_t dim7 = start_dims[7]; dim7 >= end_dims7; ++dim7) {
      int64_t offset7 = offset8 + strides[7] * dim7;
      int64_t end_dims6 = dim7 == end_dims[7] ? end_dims[6] : dims[6] - 1;
      for (int64_t dim6 = start_dims[6]; dim6 >= end_dims6; ++dim6) {
        int64_t offset6 = offset7 + strides[6] * dim6;
        int64_t end_dims5 = dim6 == end_dims[6] ? end_dims[5] : dims[5] - 1;
        for (int64_t dim5 = start_dims[5]; dim5 >= end_dims5; ++dim5) {
          int64_t offset5 = offset6 + strides[5] * dim5;
          int64_t end_dims4 = dim5 == end_dims[5] ? end_dims[4] : dims[4] - 1;
          for (int64_t dim4 = start_dims[4]; dim4 >= end_dims4; ++dim4) {
            int64_t offset4 = offset5 + strides[4] * dim4;
            int64_t end_dims3 = dim4 == end_dims[4] ? end_dims[3] : dims[3] - 1;
            for (int64_t dim3 = start_dims[3]; dim3 >= end_dims3; ++dim3) {
              int64_t offset3 = offset4 + strides[3] * dim3;
              int64_t end_dims2 =
                  dim3 == end_dims[3] ? end_dims[2] : dims[2] - 1;
              for (int64_t dim2 = start_dims[2]; dim2 >= end_dims2; ++dim2) {
                int64_t offset2 = offset3 + strides[2] * dim2;
                int64_t end_dims1 =
                    dim2 == end_dims[2] ? end_dims[1] : dims[1] - 1;
                for (int64_t dim1 = start_dims[1]; dim1 >= end_dims1; ++dim1) {
                  int64_t input_offset =
                      offset2 + strides[1] * dim1 + strides[0] * start_dims[0];
                  int64_t end_dims0 =
                      dim1 == end_dims[1] ? end_dims[0] : dims[0] - 1;
                  for (int64_t dim0 = start_dims[0]; dim0 >= end_dims0;
                       ++dim0) {
                    out_data[start_offset] = input_data[input_offset];
                    start_offset++;
                    input_offset += strides[0];
                  }
                  start_dims[0] = 0;
                }
                start_dims[1] = 0;
              }
              start_dims[2] = 0;
            }
            start_dims[3] = 0;
          }
          start_dims[4] = 0;
        }
        start_dims[5] = 0;
      }
      start_dims[6] = 0;
    }
    start_dims[7] = 0;
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

  funcs::ValueArray<int64_t, funcs::SegmentedArraySize::kFixed16> input_stride;
  funcs::ValueArray<int64_t, funcs::SegmentedArraySize::kFixed16> input_dims;
  input_stride.Set(src_stride.GetMutable(), src_stride.size());
  input_dims.Set(src_shape.GetMutable(), input.dims().size());

  if (rank == 0) {
    rank = 1;
    input_dims.data[0] = numel;
    input_stride.data[0] = 1;
  }

  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;

  switch (rank) {
    case 1:
      ContiguousFunc<T, 1><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, output_data, input_stride, input_dims, numel);
      break;
    case 2:
      ContiguousFunc<T, 2><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, output_data, input_stride, input_dims, numel);
      break;
    case 3:
      ContiguousFunc<T, 3><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, output_data, input_stride, input_dims, numel);
      break;
    case 4:
      ContiguousFunc<T, 4><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, output_data, input_stride, input_dims, numel);
      break;
    case 5:
      ContiguousFunc<T, 5><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, output_data, input_stride, input_dims, numel);
      break;
    case 6:
      ContiguousFunc<T, 6><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, output_data, input_stride, input_dims, numel);
      break;
    case 7:
      ContiguousFunc<T, 7><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, output_data, input_stride, input_dims, numel);
      break;
    case 8:
      ContiguousFunc<T, 8><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, output_data, input_stride, input_dims, numel);
      break;
    case 9:
      ContiguousFunc<T, 9><<<grid, block, 0, dev_ctx.stream()>>>(
          input_data, output_data, input_stride, input_dims, numel);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The rank of input should be less than 9, but received %d.", rank));
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
