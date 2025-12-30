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

#include "paddle/phi/kernels/index_add_kernel.h"

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

COMMON_DECLARE_bool(cudnn_deterministic);

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename IndexT>
__global__ void index_add_cuda_kernel(const T* input,
                                      const IndexT* index,
                                      const T* add_value,
                                      int64_t N,
                                      int64_t stride,
                                      int64_t size,
                                      int64_t delta,
                                      T* output,
                                      int64_t index_dim_size) {
  CUDA_KERNEL_LOOP_TYPE(idx, N, int64_t) {
    int64_t pre_idx = idx / (stride * size);
    int64_t dim_idx = idx % (stride * size) / stride;
    IndexT src_dim_idx =
        (index[dim_idx] < 0 ? index[dim_idx] + index_dim_size : index[dim_idx]);
    int64_t input_idx =
        idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
    phi::CudaAtomicAdd(&output[input_idx], add_value[idx]);
  }
}

template <typename T, typename IndexT>
__global__ void index_add_deterministic_cuda_kernel(const T* input,
                                                    const IndexT* index,
                                                    const T* add_value,
                                                    int64_t index_size,
                                                    int64_t stride,
                                                    int64_t pre_size,
                                                    int64_t output_dim_size,
                                                    T* output) {
  int64_t num_columns = pre_size * stride;
  CUDA_KERNEL_LOOP_TYPE(col_idx, num_columns, int64_t) {
    int64_t pre_idx = col_idx / stride;
    int64_t post_idx = col_idx % stride;

    for (int64_t k = 0; k < index_size; ++k) {
      IndexT src_dim_idx = index[k];
      IndexT actual_dim_idx =
          (src_dim_idx < 0 ? src_dim_idx + output_dim_size : src_dim_idx);

      int64_t val_idx = (pre_idx * index_size + k) * stride + post_idx;
      int64_t out_idx =
          (pre_idx * output_dim_size + actual_dim_idx) * stride + post_idx;
      output[out_idx] += add_value[val_idx];
    }
  }
}

template <typename T, typename Context>
void IndexAddKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& index,
                    const DenseTensor& add_value,
                    int axis,
                    DenseTensor* output) {
  if (x.numel() == 0) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);
    return;
  }
  if (index.numel() == 0) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);
    return;
  }
  if (add_value.numel() == 0) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);
    return;
  }
  auto input_dim = x.dims();
  auto output_dim = output->dims();
  auto add_value_dim = add_value.dims();
  const auto& index_type = index.dtype();
  int dim = axis;
  dim = dim >= 0 ? dim : dim + input_dim.size();
  auto stride_dim = common::stride(input_dim);
  int64_t stride = stride_dim[dim];
  int64_t size = add_value_dim[dim];
  int64_t delta = input_dim[dim] - size;

  auto* in_data = x.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(output);
  auto* add_value_data = add_value.data<T>();

  int64_t numel = add_value.numel();
  auto stream = dev_ctx.stream();

  // copy input to output.
  // todo(@limin29): inplace do not need copy.
  Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);

  auto index_dim_size = input_dim[dim];

  if (FLAGS_cudnn_deterministic) {
    int64_t pre_size = numel / (size * stride);
    int64_t num_columns = pre_size * stride;

    unsigned int block_dim = PADDLE_CUDA_NUM_THREADS;
    dim3 grid_dim = dim3((num_columns + block_dim - 1) / block_dim);
    phi::backends::gpu::LimitGridDim(dev_ctx, &grid_dim);

    if (index_type == phi::DataType::INT64) {
      const int64_t* index_data = index.data<int64_t>();
      index_add_deterministic_cuda_kernel<T, int64_t>
          <<<grid_dim, block_dim, 0, stream>>>(in_data,
                                               index_data,
                                               add_value_data,
                                               size,
                                               stride,
                                               pre_size,
                                               index_dim_size,
                                               out_data);
    } else {
      const int* index_data = index.data<int>();
      index_add_deterministic_cuda_kernel<T, int>
          <<<grid_dim, block_dim, 0, stream>>>(in_data,
                                               index_data,
                                               add_value_data,
                                               size,
                                               stride,
                                               pre_size,
                                               index_dim_size,
                                               out_data);
    }
  } else {
    unsigned int block_dim = PADDLE_CUDA_NUM_THREADS;
    dim3 grid_dim = dim3((numel + block_dim - 1) / block_dim);
    phi::backends::gpu::LimitGridDim(dev_ctx, &grid_dim);

    if (index_type == phi::DataType::INT64) {
      const int64_t* index_data = index.data<int64_t>();
      index_add_cuda_kernel<T, int64_t>
          <<<grid_dim, block_dim, 0, stream>>>(in_data,
                                               index_data,
                                               add_value_data,
                                               numel,
                                               stride,
                                               size,
                                               delta,
                                               out_data,
                                               index_dim_size);
    } else {
      const int* index_data = index.data<int>();
      index_add_cuda_kernel<T, int>
          <<<grid_dim, block_dim, 0, stream>>>(in_data,
                                               index_data,
                                               add_value_data,
                                               numel,
                                               stride,
                                               size,
                                               delta,
                                               out_data,
                                               index_dim_size);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_add,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexAddKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   int64_t) {}
