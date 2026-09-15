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

#include "paddle/phi/kernels/take_along_axis_grad_kernel.h"

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

COMMON_DECLARE_bool(cudnn_deterministic);

namespace phi {

// Deterministic scatter-add for take_along_axis_grad:
// Each thread owns an exclusive slice of x_grad along the axis dimension,
// thus no atomic operations are needed and the reduction order is fixed,
// so the result is bit-identical across runs.
template <typename T, typename IndexT>
__global__ void TakeAlongAxisGradDeterministicKernel(
    const T* __restrict__ out_grad,
    T* __restrict__ x_grad,
    const IndexT* __restrict__ index,
    int64_t index_axis_size,  // index.dims()[axis]
    int64_t x_axis_size,      // x_grad.dims()[axis]
    int64_t inner_size,       // prod(dims[axis+1..n-1])
    int64_t outer_size) {     // prod(dims[0..axis-1])
  int64_t num_cols = outer_size * inner_size;
  CUDA_KERNEL_LOOP_TYPE(col_idx, num_cols, int64_t) {
    int64_t outer_idx = col_idx / inner_size;
    int64_t inner_idx = col_idx % inner_size;

    for (int64_t k = 0; k < index_axis_size; ++k) {
      int64_t index_pos =
          (outer_idx * index_axis_size + k) * inner_size + inner_idx;
      int64_t val = static_cast<int64_t>(index[index_pos]);
      if (val < 0) val += x_axis_size;
      int64_t out_grad_pos = index_pos;  // out_grad has same shape as index
      int64_t x_grad_pos =
          (outer_idx * x_axis_size + val) * inner_size + inner_idx;
      x_grad[x_grad_pos] += out_grad[out_grad_pos];
    }
  }
}

template <typename T, typename Context>
void TakeAlongAxisGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& index,
                             const DenseTensor& out_grad,
                             int axis,
                             DenseTensor* x_grad) {
  // We need to know the shape of input matrix to determine the shape of grad
  // matrix of input.
  x_grad->Resize(x.dims());
  dev_ctx.template Alloc<T>(x_grad);

  if (x_grad->numel() == 0) {
    return;
  }

  // Set to zero tensor.
  funcs::SetConstant<Context, T> functor;
  functor(dev_ctx, x_grad, static_cast<T>(0));
  const auto& index_type = index.dtype();

  if (FLAGS_cudnn_deterministic) {
    const auto& x_dims = x_grad->dims();
    int ndim = x_dims.size();
    int64_t outer_size = 1, inner_size = 1;
    for (int d = 0; d < axis; ++d) outer_size *= x_dims[d];
    for (int d = axis + 1; d < ndim; ++d) inner_size *= x_dims[d];
    int64_t index_axis_size = index.dims()[axis];
    int64_t x_axis_size = x_dims[axis];
    int64_t num_cols = outer_size * inner_size;

    auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, num_cols);
    auto stream = dev_ctx.stream();

    if (index_type == DataType::INT32) {
      TakeAlongAxisGradDeterministicKernel<T, int32_t>
          <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
              out_grad.data<T>(),
              x_grad->data<T>(),
              index.data<int32_t>(),
              index_axis_size,
              x_axis_size,
              inner_size,
              outer_size);
    } else if (index_type == DataType::INT64) {
      TakeAlongAxisGradDeterministicKernel<T, int64_t>
          <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
              out_grad.data<T>(),
              x_grad->data<T>(),
              index.data<int64_t>(),
              index_axis_size,
              x_axis_size,
              inner_size,
              outer_size);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The data type of input index is expected "
          "to be int32 or int64, but received %s.",
          DataTypeToString(index_type)));
    }
    return;
  }

  if (index_type == DataType::INT32) {
    funcs::gpu_scatter_add_kernel<T, int32_t>(
        *x_grad,
        axis,
        index,
        out_grad,
        true,
        dev_ctx);  // the gradient of gather is scatter
  } else if (index_type == DataType::INT64) {
    funcs::gpu_scatter_add_kernel<T, int64_t>(
        *x_grad, axis, index, out_grad, true, dev_ctx);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The data type of input index is expected "
        "to be int32 or int64, but received %s.",
        DataTypeToString(index_type)));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(take_along_axis_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::TakeAlongAxisGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {}
