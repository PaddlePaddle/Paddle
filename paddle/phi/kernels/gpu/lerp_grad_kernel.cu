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

#include "paddle/phi/kernels/lerp_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/broadcast_tensors_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
__global__ void GetLerpGrad(const T* weight,
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
__global__ void GetLerpGradRankZero(const T* weight,
                                    const T* dout,
                                    T* dx,
                                    T* dy,
                                    const int out_size,
                                    const int x_size,
                                    const int y_size) {
  CUDA_KERNEL_LOOP_TYPE(idx, out_size, int64_t) {
    T temp_dx = weight[0] * dout[idx];
    if (idx < x_size) {
      dx[idx] = dout[idx] - temp_dx;
    }
    if (idx < y_size) {
      dy[idx] = temp_dx;
    }
  }
}

template <typename T, typename Context, size_t D>
void GetRduceResult(const Context& ctx,
                    const DenseTensor& out_grad,
                    const DenseTensor& b_xgrad,
                    const DenseTensor& b_ygrad,
                    DenseTensor* x_grad,
                    DenseTensor* y_grad) {
  auto& dout = out_grad;
  auto dout_dims = dout.dims();
  auto* dx = x_grad;
  auto* dy = y_grad;
  DDim dx_dims;
  DDim dy_dims;
  Eigen::DSizes<int, D * 2> dx_reshape_dims;
  Eigen::DSizes<int, D * 2> dy_reshape_dims;
  Eigen::DSizes<int, D> reduce_dims;
  Eigen::DSizes<int, D> dx_bcast_dims;
  Eigen::DSizes<int, D> dy_bcast_dims;

  dx_dims = phi::funcs::ExtendDims2Rank(dx->dims(), D);
  phi::funcs::GetBroadcastDims<D>(dx_dims, dout_dims, &dx_bcast_dims);
  dy_dims = phi::funcs::ExtendDims2Rank(dy->dims(), D);
  phi::funcs::GetBroadcastDims<D>(dy_dims, dout_dims, &dy_bcast_dims);
  for (int i = 0; i < dout_dims.size(); ++i) {
    dx_reshape_dims[2 * i] = dx_bcast_dims[i];
    dx_reshape_dims[2 * i + 1] = dx_dims[i];

    dy_reshape_dims[2 * i] = dy_bcast_dims[i];
    dy_reshape_dims[2 * i + 1] = dy_dims[i];
    reduce_dims[i] = 2 * i;
  }

  ctx.template Alloc<T>(dx);
  ctx.template Alloc<T>(dy);
  auto eigen_dx = phi::EigenTensor<T, D>::From(*dx, dx_dims);
  auto eigen_dy = phi::EigenTensor<T, D>::From(*dy, dy_dims);
  dx_dims = phi::funcs::ExtendDims2Rank(x_grad->dims(), D);
  auto broad_dx = phi::EigenTensor<T, D>::From(b_xgrad);
  auto broad_dy = phi::EigenTensor<T, D>::From(b_ygrad);

  auto& place = *ctx.eigen_device();
  eigen_dx.device(place) = broad_dx.reshape(dx_reshape_dims)
                               .sum(reduce_dims)
                               .reshape(eigen_dx.dimensions());
  eigen_dy.device(place) = broad_dy.reshape(dy_reshape_dims)
                               .sum(reduce_dims)
                               .reshape(eigen_dy.dimensions());
}

int XYNeedReduce(const DenseTensor& x,
                 const DenseTensor& y,
                 const DenseTensor& out) {
  // 不考虑不可broadcast的情况，在算子调用时已排除
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto out_dims = out.dims();
  int x_rank = x_dims.size();
  int y_rank = y_dims.size();
  int out_rank = out_dims.size();
  int smaller_rank = std::min(x_rank, y_rank);
  if (std::max(x_rank, y_rank) < out_rank) {
    return 1;
  }
  for (int i = 1; i <= smaller_rank; ++i) {
    int x_idx = x_rank - i;
    int y_idx = y_rank - i;
    int out_idx = out_rank - i;
    if (x_dims[x_idx] != y_dims[y_idx]) {
      return 1;
    }
    if (x_dims[x_idx] == 1 && y_dims[y_idx] == 1 && out_dims[out_idx] != 1) {
      return 1;
    }
  }
  return 0;
}

template <typename T, typename Context>
void SwitchKernel(const Context& ctx,
                  const DenseTensor& weight,
                  const DenseTensor& out_grad,
                  const int x_grad_size,
                  const int y_grad_size,
                  T* x_grad_data,
                  T* y_grad_data) {
  if (weight.dims().size() == 1) {
    const T* weight_data = weight.data<T>();
    const T* out_grad_data = out_grad.data<T>();
    const int out_size = out_grad.numel();
    const int weight_size = weight.numel();
    auto gpu_config = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, out_size);
    GetLerpGradRankZero<T><<<gpu_config.GetGridSize(),
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
    // 首先对weight进行braodcast，使用
    // phi::BroadcastTensorsKernel，使其维度和out_grad一致
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
    GetLerpGrad<T><<<gpu_config.GetGridSize(),
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

  // 判断x_grad, y_grad
  // 是否需要reduce，需要reduce的话就先进行broadcast用b_xgrad,
  // b_ygrad。不需要的话就用x_grad, y_grad。
  // 如果x,y在中间有某个维度不一致，或者和weight的中间某个维度不一致，就需要先broadcast再reduce。
  //  例如 case1:  x:2*1*3, y:2*2*3  w:2*2*3 => out: 2*2*3
  //      case2:  x:2*1:3, y: 2*1*3 w:2*2*3 => out: 2*2*3
  // 如果x,y在初始维度不一致，就无所谓，只要控制kernel写入的idx大小不越界访问内存就可以。例如
  // x:1*2*3, y:2*2*3，无需reduce。

  int reduce_flag = XYNeedReduce(x, y, out);
  if (reduce_flag == 0) {
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

    switch (rank) {
      case 1:
        GetRduceResult<T, Context, 1>(
            ctx, out_grad, b_xgrad, b_ygrad, x_grad, y_grad);
        break;
      case 2:
        GetRduceResult<T, Context, 2>(
            ctx, out_grad, b_xgrad, b_ygrad, x_grad, y_grad);
        break;
      case 3:
        GetRduceResult<T, Context, 3>(
            ctx, out_grad, b_xgrad, b_ygrad, x_grad, y_grad);
        break;
      case 4:
        GetRduceResult<T, Context, 4>(
            ctx, out_grad, b_xgrad, b_ygrad, x_grad, y_grad);
        break;
      case 5:
        GetRduceResult<T, Context, 5>(
            ctx, out_grad, b_xgrad, b_ygrad, x_grad, y_grad);
        break;
      case 6:
        GetRduceResult<T, Context, 6>(
            ctx, out_grad, b_xgrad, b_ygrad, x_grad, y_grad);
        break;
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    lerp_grad, GPU, ALL_LAYOUT, phi::LerpGradKernel, float, double) {}
