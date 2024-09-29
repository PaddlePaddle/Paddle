// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/lerp_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/broadcast_tensors_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T>
__global__ void LerpGradKernelImpl(const T* weight,
                                   const T* dout,
                                   T* dx,
                                   T* dy,
                                   const int out_size,
                                   const int x_size,
                                   const int y_size) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  CUDA_KERNEL_LOOP_TYPE(idx, out_size, int64_t) {
    MPType temp_dx =
        static_cast<MPType>(weight[idx]) * static_cast<MPType>(dout[idx]);
    if (dx) {
      if (idx < x_size) {
        dx[idx] = static_cast<T>(static_cast<MPType>(dout[idx]) - temp_dx);
      }
    }
    if (dy) {
      if (idx < y_size) {
        dy[idx] = static_cast<T>(temp_dx);
      }
    }
  }
}

template <typename T>
__global__ void LerpGradScalarKernelImpl(const T* weight,
                                         const T* dout,
                                         T* dx,
                                         T* dy,
                                         const int out_size,
                                         const int x_size,
                                         const int y_size) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType weight_scalar = static_cast<MPType>(weight[0]);
  CUDA_KERNEL_LOOP_TYPE(idx, out_size, int64_t) {
    MPType temp_dx = weight_scalar * static_cast<MPType>(dout[idx]);
    if (dx) {
      if (idx < x_size) {
        dx[idx] = static_cast<T>(static_cast<MPType>(dout[idx]) - temp_dx);
      }
    }
    if (dy) {
      if (idx < y_size) {
        dy[idx] = static_cast<T>(temp_dx);
      }
    }
  }
}

bool XYNeedReduce(const DenseTensor& x,
                  const DenseTensor& y,
                  const DenseTensor& out) {
  auto x_dims = x.dims().size() ? x.dims()
                                : common::make_ddim(std::vector<int64_t>(1, 1));
  auto y_dims = y.dims().size() ? y.dims()
                                : common::make_ddim(std::vector<int64_t>(1, 1));

  auto out_dims = out.dims();
  if (out_dims.size() == 0) {
    return false;
  }
  int x_rank = x_dims.size();
  int y_rank = y_dims.size();
  int out_rank = out_dims.size();
  int smaller_rank = std::min(x_rank, y_rank);
  if (std::max(x_rank, y_rank) < out_rank) {
    return true;
  }
  for (int i = 1; i <= smaller_rank; ++i) {
    int x_idx = x_rank - i;
    int y_idx = y_rank - i;
    int out_idx = out_rank - i;
    if (x_dims[x_idx] != y_dims[y_idx]) {
      return true;
    }
    if (x_dims[x_idx] == 1 && y_dims[y_idx] == 1 && out_dims[out_idx] != 1) {
      return true;
    }
  }
  return false;
}

template <typename T, typename Context>
void SwitchKernel(const Context& ctx,
                  const DenseTensor& weight,
                  const DenseTensor& out_grad,
                  const int x_grad_size,
                  const int y_grad_size,
                  T* x_grad_data,
                  T* y_grad_data) {
  if (weight.numel() == 1) {
    //    condition when weight is a scalar
    const T* weight_data = weight.data<T>();
    const T* out_grad_data = out_grad.data<T>();
    const int64_t out_size = out_grad.numel();
    const int64_t weight_size = weight.numel();
    auto gpu_config = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, out_size);
    LerpGradScalarKernelImpl<T><<<gpu_config.GetGridSize(),
                                  gpu_config.GetBlockSize(),
                                  0,
                                  ctx.stream()>>>(weight_data,
                                                  out_grad_data,
                                                  x_grad_data,
                                                  y_grad_data,
                                                  out_size,
                                                  x_grad_size,
                                                  y_grad_size);
  } else {
    //    broadcast weight with out_grad's dimensions
    const std::vector<const DenseTensor*> in_tensors = {&weight, &out_grad};
    DenseTensor b_weight = phi::EmptyLike<T>(ctx, out_grad);
    DenseTensor b_out = phi::EmptyLike<T>(ctx, out_grad);
    std::vector<DenseTensor*> out_tensors = {&b_weight, &b_out};

    phi::BroadcastTensorsKernel<T, Context>(ctx, in_tensors, out_tensors);

    const T* weight_data = b_weight.data<T>();
    const T* out_grad_data = b_out.data<T>();
    const int out_size = out_grad.numel();
    const int weight_size = weight.numel();

    auto gpu_config = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, out_size);
    LerpGradKernelImpl<T><<<gpu_config.GetGridSize(),
                            gpu_config.GetBlockSize(),
                            0,
                            ctx.stream()>>>(weight_data,
                                            out_grad_data,
                                            x_grad_data,
                                            y_grad_data,
                                            out_size,
                                            x_grad_size,
                                            y_grad_size);
  }
}

template <typename T, typename Context>
void LerpGradKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    const DenseTensor& weight,
                    const DenseTensor& out,
                    const DenseTensor& out_grad,
                    DenseTensor* x_grad,
                    DenseTensor* y_grad) {
  const int rank = out.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      0,
      common::errors::InvalidArgument(
          "The number of dimensions for LerpGradOp must be "
          "greater than or equal to 0, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      common::errors::InvalidArgument(
          "The number of dimensions for LerpGradOp must be "
          "less than or equal to 6, but the value received is %d.",
          rank));

  //  check if x_grad and y_grad need to be reduced
  //  if x has a different dimension with y or weight in the middle axis, then
  //  they need to be broadcast and then reduced.
  bool reduce_flag = XYNeedReduce(x, y, out);
  if (!reduce_flag) {
    int x_grad_size = 0, y_grad_size = 0;
    T* x_grad_data = NULL;
    T* y_grad_data = NULL;

    if (x_grad) {
      x_grad_data = ctx.template Alloc<T>(x_grad);
      x_grad_size = x.numel();
    }

    if (y_grad) {
      y_grad_data = ctx.template Alloc<T>(y_grad);
      y_grad_size = y.numel();
    }

    SwitchKernel<T, Context>(ctx,
                             weight,
                             out_grad,
                             x_grad_size,
                             y_grad_size,
                             x_grad_data,
                             y_grad_data);

  } else {
    int x_grad_size = 0, y_grad_size = 0;
    DenseTensor b_xgrad = phi::EmptyLike<T, Context>(ctx, out_grad);
    DenseTensor b_ygrad = phi::EmptyLike<T, Context>(ctx, out_grad);
    T* x_grad_data = NULL;
    T* y_grad_data = NULL;

    if (x_grad) {
      x_grad_data = ctx.template Alloc<T>(&b_xgrad);
      x_grad_size = out.numel();
    }

    if (y_grad) {
      y_grad_data = ctx.template Alloc<T>(&b_ygrad);
      y_grad_size = out.numel();
    }

    SwitchKernel<T, Context>(ctx,
                             weight,
                             out_grad,
                             x_grad_size,
                             y_grad_size,
                             x_grad_data,
                             y_grad_data);

    auto zero_dim = common::make_ddim(std::vector<int64_t>(1, 1));
    if (x_grad) {
      std::vector<int> reduce_axis_x =
          funcs::GetReduceDim(x_grad->dims().size() ? x_grad->dims() : zero_dim,
                              b_xgrad.dims(),
                              -1);
      if (!reduce_axis_x.empty()) {
        phi::SumKernel<T, Context>(
            ctx, b_xgrad, reduce_axis_x, b_xgrad.dtype(), false, x_grad);
      } else {
        x_grad->ShareDataWith(b_xgrad);
      }
    }

    if (y_grad) {
      std::vector<int> reduce_axis_y =
          funcs::GetReduceDim(y_grad->dims().size() ? y_grad->dims() : zero_dim,
                              b_ygrad.dims(),
                              -1);
      if (!reduce_axis_y.empty()) {
        phi::SumKernel<T, Context>(
            ctx, b_ygrad, reduce_axis_y, b_ygrad.dtype(), false, y_grad);
      } else {
        y_grad->ShareDataWith(b_ygrad);
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(lerp_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LerpGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double) {}
