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

#include "paddle/phi/kernels/index_sample_kernel.h"

#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

namespace {
template <typename Context>
void LimitGridDim(const Context& ctx, dim3* grid_dim) {
  auto max_grid_dim =
      reinterpret_cast<const phi::GPUContext&>(ctx).GetCUDAMaxGridDimSize();
  grid_dim->x = grid_dim->x < max_grid_dim[0] ? grid_dim->x : max_grid_dim[0];
  grid_dim->y = grid_dim->y < max_grid_dim[1] ? grid_dim->y : max_grid_dim[1];
}
#define PREDEFINED_BLOCK_SIZE_X 512
#define PREDEFINED_BLOCK_SIZE 1024
#define MIN(a, b) ((a) < (b) ? (a) : (b))
}

template <typename T, typename IndexT = int>
__global__ void IndexSampleForward(const IndexT* index,
                                   const T* in_data,
                                   T* out_data,
                                   size_t index_length,
                                   size_t input_length,
                                   size_t batch_size) {
  unsigned int index_i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int index_j = blockDim.y * blockIdx.y + threadIdx.y;
  for (; index_j < batch_size; index_j += blockDim.y * gridDim.y) {
    index_i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; index_i < index_length; index_i += blockDim.x * gridDim.x) {
      unsigned int index_idx = index_j * index_length + index_i;
      unsigned int in_idx = index_j * input_length + index_i;
      IndexT sample_idx = index[index_idx];
      out_data[index_idx] = in_data[in_idx - index_i + sample_idx];
    }
  }
}

template <typename T, typename Context>
void IndexSampleKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& index,
                       DenseTensor* out) {
  auto index_type = index.dtype();
  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      errors::InvalidArgument(
          "Input(Index) holds the wrong type, it holds %s, but "
          "desires to be %s or %s",
          paddle::framework::DataTypeToString(
              paddle::framework::TransToProtoVarType(index_type)),
          paddle::framework::DataTypeToString(
              paddle::framework::TransToProtoVarType(DataType::INT32)),
          paddle::framework::DataTypeToString(
              paddle::framework::TransToProtoVarType((DataType::INT64)))));
  const T* in_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
  auto input_dim = x.dims();
  auto index_dim = index.dims();
  size_t batch_size = input_dim[0];
  size_t input_length = input_dim[1];
  size_t index_length = index_dim[1];

  auto block_width = paddle::platform::RoundToPowerOfTwo(index_length);
  block_width = MIN(block_width, PREDEFINED_BLOCK_SIZE_X);
  int block_height =
      paddle::platform::RoundToPowerOfTwo(index_length * batch_size) /
      block_width;
  block_height = MIN(block_height, PREDEFINED_BLOCK_SIZE / block_width);
  dim3 block_dim(block_width, block_height);
  dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                (batch_size + block_dim.y - 1) / block_dim.y);
  LimitGridDim(ctx, &grid_dim);

  if (index_type == DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    IndexSampleForward<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
        index_data, in_data, out_data, index_length, input_length, batch_size);
  } else if (index_type == DataType::INT32) {
    const int* index_data = index.data<int>();
    IndexSampleForward<T, int><<<grid_dim, block_dim, 0, stream>>>(
        index_data, in_data, out_data, index_length, input_length, batch_size);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_sample,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexSampleKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
