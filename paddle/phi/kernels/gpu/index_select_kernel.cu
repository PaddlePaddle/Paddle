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

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/gpu/index_select_impl.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

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
  if (numel == 0) {
    return;
  }
  auto stream = ctx.stream();

  unsigned int block_dim = PADDLE_CUDA_NUM_THREADS;
  dim3 grid_dim = dim3((numel + block_dim - 1) / block_dim);
  phi::backends::gpu::LimitGridDim(ctx, &grid_dim);

  if (index_type == phi::DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    index_select_cuda_kernel<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
        in_data, out_data, index_data, numel, stride, size, delta);
  } else {
    const int* index_data = index.data<int>();
    index_select_cuda_kernel<T, int><<<grid_dim, block_dim, 0, stream>>>(
        in_data, out_data, index_data, numel, stride, size, delta);
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
