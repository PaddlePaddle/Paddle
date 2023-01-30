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

<<<<<<< HEAD
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

DECLARE_bool(cudnn_deterministic);

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;
=======
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

template <typename T, typename IndexT>
__global__ void index_add_cuda_kernel(const T* input,
                                      const IndexT* index,
                                      const T* add_value,
                                      int64_t N,
                                      int64_t stride,
                                      int64_t size,
                                      int64_t delta,
                                      T* output) {
  CUDA_KERNEL_LOOP_TYPE(idx, N, int64_t) {
    int64_t pre_idx = idx / (stride * size);
    int64_t dim_idx = idx % (stride * size) / stride;
    IndexT src_dim_idx = index[dim_idx];
    int64_t input_idx =
        idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
<<<<<<< HEAD
    phi::CudaAtomicAdd(&output[input_idx], add_value[idx]);
=======
    paddle::platform::CudaAtomicAdd(&output[input_idx], add_value[idx]);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }
}

template <typename T, typename Context>
void IndexAddKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& index,
                    const DenseTensor& add_value,
                    int axis,
                    DenseTensor* output) {
<<<<<<< HEAD
  auto input_dim = x.dims();
  auto output_dim = output->dims();
  auto add_value_dim = add_value.dims();
  const auto& index_type = index.dtype();
  int dim = axis;
=======
  int dim = axis;
  auto input_dim = x.dims();
  auto output_dim = output->dims();
  auto add_value_dim = add_value.dims();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  dim = dim >= 0 ? dim : dim + input_dim.size();
  auto stride_dim = phi::stride(input_dim);
  int64_t stride = stride_dim[dim];
  int64_t size = add_value_dim[dim];
  int64_t delta = input_dim[dim] - size;
<<<<<<< HEAD
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
  phi::backends::gpu::LimitGridDim(ctx, &grid_dim);
=======
  paddle::platform::LimitGridDim(ctx, &grid_dim);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

  // copy input to output.
  // todo(@limin29): inplace do not need copy.
  phi::Copy(ctx, x, ctx.GetPlace(), false, output);

<<<<<<< HEAD
  if (FLAGS_cudnn_deterministic) {
    VLOG(2) << "Run grad kernel of index_add with single thread.";
    block_dim = 1;
    grid_dim.x = 1;
  }

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
                                             out_data);
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
                                             out_data);
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
                   int,
                   int64_t) {}
