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

#include "paddle/phi/kernels/index_select_grad_kernel.h"

#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/math_function.h"

DECLARE_bool(cudnn_deterministic);

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename IndexT>
__global__ void index_select_grad_cuda_kernel(const T* output_grad,
                                              T* input_grad,
                                              const IndexT* index,
                                              int64_t nums,
                                              int64_t N,
                                              int64_t stride,
                                              int64_t size,
                                              int64_t delta) {
  CUDA_KERNEL_LOOP_TYPE(idx, N, int64_t) {
    int64_t pre_idx = idx / (stride * size);
    int64_t dim_idx = idx % (stride * size) / stride;
    IndexT src_dim_idx = index[dim_idx];
    int64_t input_idx =
        idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
    paddle::platform::CudaAtomicAdd(&input_grad[input_idx], output_grad[idx]);
  }
}

template <typename T, typename Context>
void IndexSelectGradKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& index,
                           const DenseTensor& out_grad,
                           int dim,
                           DenseTensor* x_grad) {
  auto* output_grad_data = out_grad.data<T>();
  auto* in_grad_data = ctx.template Alloc<T>(x_grad);

  auto input_dim = x_grad->dims();
  auto output_dim = out_grad.dims();
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

  int64_t numel = x_grad->numel();
  if (numel == 0) {
    return;
  }
  int64_t index_nums = index.numel();
  int64_t out_nums = out_grad.numel();

  auto stream = ctx.stream();

  unsigned int block_dim = PADDLE_CUDA_NUM_THREADS;
  dim3 grid_dim = dim3((numel + block_dim - 1) / block_dim);
  paddle::platform::LimitGridDim(ctx, &grid_dim);

  phi::funcs::SetConstant<phi::GPUContext, T> index_select_grad_init;
  index_select_grad_init(ctx, x_grad, static_cast<T>(0));

  if (FLAGS_cudnn_deterministic) {
    VLOG(2) << "Run grad kernel of index_select with single thread.";
    block_dim = 1;
    grid_dim.x = 1;
  }

  if (index_type == phi::DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    index_select_grad_cuda_kernel<T, int64_t>
        <<<grid_dim, block_dim, 0, stream>>>(output_grad_data,
                                             in_grad_data,
                                             index_data,
                                             index_nums,
                                             out_nums,
                                             stride,
                                             size,
                                             delta);
  } else {
    const int* index_data = index.data<int>();
    index_select_grad_cuda_kernel<T, int>
        <<<grid_dim, block_dim, 0, stream>>>(output_grad_data,
                                             in_grad_data,
                                             index_data,
                                             index_nums,
                                             out_nums,
                                             stride,
                                             size,
                                             delta);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_select_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexSelectGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {}
