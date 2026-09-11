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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#include "paddle/phi/kernels/gather_kernel.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {

template <typename T, typename IndexT>
__global__ void gather_grad_deterministic_cuda_kernel(const T* out_grad,
                                                      T* x_grad,
                                                      const IndexT* index,
                                                      int64_t index_size,
                                                      int64_t inner_dim_size,
                                                      int64_t outer_dim_size,
                                                      int64_t x_axis_dim_size) {
  int64_t num_columns = inner_dim_size * outer_dim_size;
  CUDA_KERNEL_LOOP_TYPE(col_idx, num_columns, int64_t) {
    int64_t inner_dim_index = col_idx / outer_dim_size;
    int64_t outer_dim_index = col_idx % outer_dim_size;
    for (int64_t k = index_size - 1; k >= 0; k--) {
      int64_t out_grad_idx =
          (inner_dim_index * index_size + k) * outer_dim_size + outer_dim_index;
      int64_t x_grad_idx =
          (inner_dim_index * x_axis_dim_size + static_cast<int64_t>(index[k])) *
              outer_dim_size +
          outer_dim_index;
      x_grad[x_grad_idx] += out_grad[out_grad_idx];
    }
  }
}

template <typename T, typename Context, typename IndexT>
void GatherV2GradDeterministicCUDAFunction(const Context& dev_ctx,
                                           const DenseTensor& out_grad,
                                           const DenseTensor& index,
                                           int axis,
                                           DenseTensor* x_grad) {
  const auto& out_grad_dims = out_grad.dims();
  int64_t index_size = index.dims().size() == 0 ? 1 : out_grad_dims[axis];
  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  for (int i = 0; i < axis; ++i) {
    inner_dim_size *= out_grad_dims[i];
  }
  for (int i = axis + 1; i < out_grad_dims.size(); ++i) {
    outer_dim_size *= out_grad_dims[i];
  }
  int64_t x_axis_dim_size = x_grad->dims()[axis];

  const auto* out_grad_data = out_grad.data<T>();
  auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  const auto* index_data = index.data<IndexT>();

  funcs::set_constant(dev_ctx, x_grad, static_cast<float>(0.0));

  int64_t num_columns = inner_dim_size * outer_dim_size;
  unsigned int block_dim = PADDLE_CUDA_NUM_THREADS;
  const uint64_t grid_x = (num_columns + block_dim - 1) / block_dim;
  PADDLE_ENFORCE_LE_UINT32_MAX(grid_x, "grid.x");
  dim3 grid_dim = dim3(static_cast<uint32_t>(grid_x));
  backends::gpu::LimitGridDim(dev_ctx, &grid_dim);

  gather_grad_deterministic_cuda_kernel<T, IndexT>
      <<<grid_dim, block_dim, 0, dev_ctx.stream()>>>(out_grad_data,
                                                     x_grad_data,
                                                     index_data,
                                                     index_size,
                                                     inner_dim_size,
                                                     outer_dim_size,
                                                     x_axis_dim_size);
}

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& index,
                      const DenseTensor& out_grad,
                      const Scalar& axis,
                      DenseTensor* x_grad) {
  // x [4, 2], index [2, 0], out [2, 0], x_grad [4, 2]
  if (out_grad.numel() == 0 || (x_grad && x_grad->numel() == 0)) {
    if (x_grad) {
      Full<T, Context>(dev_ctx, x_grad->dims(), 0, x_grad);
    }
    return;
  }
  const auto& index_type = index.dtype();
  auto axis_v = axis.to<int>();
  if (axis_v < 0) {
    axis_v += static_cast<int>(x.dims().size());
  }

  if (axis_v != 0) {
    if (FLAGS_use_accuracy_compatible_kernel) {
      if (index_type == DataType::INT32) {
        GatherV2GradDeterministicCUDAFunction<T, Context, int32_t>(
            dev_ctx, out_grad, index, axis_v, x_grad);
      } else if (index_type == DataType::INT64) {
        GatherV2GradDeterministicCUDAFunction<T, Context, int64_t>(
            dev_ctx, out_grad, index, axis_v, x_grad);
      }
      return;
    }
    if (index_type == DataType::INT32) {
      funcs::GatherV2GradCUDAFunction<T, int32_t>(
          &out_grad, &index, axis_v, x_grad, dev_ctx);
    } else if (index_type == DataType::INT64) {
      funcs::GatherV2GradCUDAFunction<T, int64_t>(
          &out_grad, &index, axis_v, x_grad, dev_ctx);
    }
    return;
  }

  dev_ctx.template Alloc<T>(x_grad);
  funcs::set_constant(dev_ctx, x_grad, static_cast<float>(0));
  if (out_grad.numel() == 0) {
    return;
  }

  if (index.dims().size() != 0) {
    if (index_type == DataType::INT32) {
      DenseTensor index_int64 =
          Cast<int32_t, Context>(dev_ctx, index, DataType::INT64);
      funcs::GPUScatterAdd<T, int64_t>(
          dev_ctx, out_grad, index_int64, x_grad, axis_v);
    } else if (index_type == DataType::INT64) {
      funcs::GPUScatterAdd<T, int64_t>(
          dev_ctx, out_grad, index, x_grad, axis_v);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The data type of Input(Index) of gather_grad must be int32 or int64 "
          "on GPU."));
    }
  } else {
    if (index_type == DataType::INT32) {
      funcs::GPUScatterAssign<T, int>(dev_ctx, out_grad, index, x_grad, false);
    } else if (index_type == DataType::INT64) {
      funcs::GPUScatterAssign<T, int64_t>(
          dev_ctx, out_grad, index, x_grad, false);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The data type of Input(Index) of gather_grad must be int32 or int64 "
          "on GPU."));
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gather_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::GatherGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
