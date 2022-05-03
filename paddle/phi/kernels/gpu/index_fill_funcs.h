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

#pragma once
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/index_fill_kernel.h"

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename IndexT>
__global__ void index_fill_cuda_kernel(T* output,
                                       const IndexT* index,
                                       int64_t N,
                                       int64_t stride,
                                       int64_t size,
                                       int64_t delta,
                                       T fill_val) {
  CUDA_KERNEL_LOOP_TYPE(idx, N, int64_t) {
    int64_t pre_idx = idx / (stride * size);
    int64_t dim_idx = idx % (stride * size) / stride;
    IndexT src_dim_idx = index[dim_idx];
    int64_t output_idx =
        idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
    output[output_idx] = fill_val;
  }
}

template <typename T, typename Context>
void index_fill_cuda_impl(const Context& dev_ctx,
                          const DenseTensor& index,
                          int axis,
                          float fill_value,
                          DenseTensor* output) {
  auto output_dim = output->dims();
  axis = axis >= 0 ? axis : axis + output_dim.size();
  auto stride_dim = phi::stride(output_dim);
  int64_t stride = stride_dim[axis];
  int64_t size = index.dims()[0];
  int64_t delta = output_dim[axis] - size;
  const auto& index_type = index.dtype();

  bool index_type_match =
      index_type == phi::DataType::INT64 || index_type == phi::DataType::INT32;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  auto* out_data = output->data<T>();

  output_dim[axis] = size;
  int64_t numel = phi::product(output_dim);
  if (numel == 0) {
    return;
  }
  auto stream = dev_ctx.stream();

  unsigned int block_dim = PADDLE_CUDA_NUM_THREADS;
  dim3 grid_dim = dim3((numel + block_dim - 1) / block_dim);
  paddle::platform::LimitGridDim(dev_ctx, &grid_dim);

  T fill_val = static_cast<T>(fill_value);
  if (index_type == phi::DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    index_fill_cuda_kernel<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
        out_data, index_data, numel, stride, size, delta, fill_val);
  } else {
    const int* index_data = index.data<int>();
    index_fill_cuda_kernel<T, int><<<grid_dim, block_dim, 0, stream>>>(
        out_data, index_data, numel, stride, size, delta, fill_val);
  }
}

}  // namespace phi
