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

#include "paddle/phi/kernels/index_select_kernel.h"

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename IndexT>
__global__ void index_select_cuda_kernel(const T* input,
                                         T* output,
                                         const IndexT* index,
                                         int64_t N,
                                         int64_t stride,
                                         int64_t size,
                                         int64_t delta) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int64_t pre_idx = idx / (stride * size);
  int64_t dim_idx = idx % (stride * size) / stride;
  IndexT src_dim_idx = index[dim_idx];
  int64_t input_idx = idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
  output[idx] = input[input_idx];
}

template <typename T, typename Context>
void IndexSelectKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& index,
                       int dim,
                       DenseTensor* output) {
  auto input_dim = x.dims();
  auto output_dim = output->dims();
  dim = dim >= 0 ? dim : dim + input_dim.size();
  auto stride_dim = phi::stride(input_dim);
  int64_t stride = stride_dim[dim];
  int64_t size = output_dim[dim];
  int64_t delta = input_dim[dim] - size;
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

  auto* in_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(output);

  int64_t numel = output->numel();
  auto stream = ctx.stream();

  if (index_type == phi::DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    index_select_cuda_kernel<T, int64_t><<<
        (numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
        PADDLE_CUDA_NUM_THREADS,
        0,
        stream>>>(in_data, out_data, index_data, numel, stride, size, delta);
    phi::backends::gpu::GpuStreamSync(stream);
  } else {
    const int* index_data = index.data<int>();
    index_select_cuda_kernel<
        T,
        int><<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
               PADDLE_CUDA_NUM_THREADS,
               0,
               stream>>>(
        in_data, out_data, index_data, numel, stride, size, delta);
    phi::backends::gpu::GpuStreamSync(stream);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_select,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexSelectKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {}
