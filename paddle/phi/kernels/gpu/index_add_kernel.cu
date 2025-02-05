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

template <typename T, typename Context>
void IndexAddKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& index,
                    const DenseTensor& add_value,
                    int axis,
                    DenseTensor* output) {
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
  T* out_data = ctx.template Alloc<T>(output);
  auto* add_value_data = add_value.data<T>();

  int64_t numel = add_value.numel();
  if (numel == 0) {
    return;
  }
  auto stream = ctx.stream();

  unsigned int block_dim = PADDLE_CUDA_NUM_THREADS;
  dim3 grid_dim = dim3((numel + block_dim - 1) / block_dim);
  phi::backends::gpu::LimitGridDim(ctx, &grid_dim);

  // copy input to output.
  // todo(@limin29): inplace do not need copy.
  phi::Copy(ctx, x, ctx.GetPlace(), false, output);

  if (FLAGS_cudnn_deterministic) {
    VLOG(2) << "Run grad kernel of index_add with single thread.";
    block_dim = 1;
    grid_dim.x = 1;
  }
  auto index_dim_size = input_dim[dim];
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

}  // namespace phi

PD_REGISTER_KERNEL(index_add,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexAddKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t) {}
