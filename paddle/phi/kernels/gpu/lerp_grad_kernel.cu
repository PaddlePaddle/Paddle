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

#include "paddle/phi/kernels/broadcast_tensors_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/gpu/reduce.h"

namespace phi {

template <typename T>
__global__ void LerpGradKernelImpl(const T* weight,
                                   const T* dout,
                                   T* dx,
                                   T* dy,
                                   const int out_size,
                                   const int x_size,
                                   const int y_size) {
  CUDA_KERNEL_LOOP_TYPE(idx, out_size, int64_t) {
    T temp_dx = weight[idx] * dout[idx];
    if (idx < x_size) {
      dx[idx] = dout[idx] - temp_dx;
    }
    if (idx < y_size) {
      dy[idx] = temp_dx;
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
  T weight_scalar = weight[0];
  CUDA_KERNEL_LOOP_TYPE(idx, out_size, int64_t) {
    T temp_dx = weight_scalar * dout[idx];
    if (idx < x_size) {
      dx[idx] = dout[idx] - temp_dx;
    }
    if (idx < y_size) {
      dy[idx] = temp_dx;
    }
  }
}

bool XYNeedReduce(const DenseTensor& x,
                  const DenseTensor& y,
                  const DenseTensor& out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto out_dims = out.dims();
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
      1,
      phi::errors::InvalidArgument(
          "The number of dimensions for LerpGradOp must be "
          "greater than or equal to 1, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      phi::errors::InvalidArgument(
          "The number of dimensions for LerpGradOp must be "
          "less than or equal to 6, but the value received is %d.",
          rank));

  //  check if x_grad and y_grad need to be reduced
  //  if x has a different dimension with y or weight in the middle axis, then
  //  they need to be broadcast and then reduced.
  bool reduce_flag = XYNeedReduce(x, y, out);
  if (!reduce_flag) {
    T* x_grad_data = ctx.template Alloc<T>(x_grad);
    T* y_grad_data = ctx.template Alloc<T>(y_grad);
    int x_grad_size = x.numel();
    int y_grad_size = y.numel();

    SwitchKernel<T, Context>(ctx,
                             weight,
                             out_grad,
                             x_grad_size,
                             y_grad_size,
                             x_grad_data,
                             y_grad_data);

  } else {
    DenseTensor b_xgrad = phi::EmptyLike<T, Context>(ctx, out_grad);
    DenseTensor b_ygrad = phi::EmptyLike<T, Context>(ctx, out_grad);
    T* x_grad_data = ctx.template Alloc<T>(&b_xgrad);
    T* y_grad_data = ctx.template Alloc<T>(&b_ygrad);
    int x_grad_size = out.numel();
    int y_grad_size = out.numel();

    SwitchKernel<T, Context>(ctx,
                             weight,
                             out_grad,
                             x_grad_size,
                             y_grad_size,
                             x_grad_data,
                             y_grad_data);

    std::vector<int> reduce_axis_x =
        funcs::GetReduceDim(x_grad->dims(), b_xgrad.dims(), -1);

    std::vector<int> reduce_axis_y =
        funcs::GetReduceDim(y_grad->dims(), b_ygrad.dims(), -1);

    if (!reduce_axis_x.empty()) {
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          ctx, b_xgrad, x_grad, kps::IdentityFunctor<T>(), reduce_axis_x);
    } else {
      x_grad->ShareDataWith(b_xgrad);
    }

    if (!reduce_axis_y.empty()) {
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          ctx, b_ygrad, y_grad, kps::IdentityFunctor<T>(), reduce_axis_y);
    } else {
      y_grad->ShareDataWith(b_ygrad);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    lerp_grad, GPU, ALL_LAYOUT, phi::LerpGradKernel, float, double) {}
