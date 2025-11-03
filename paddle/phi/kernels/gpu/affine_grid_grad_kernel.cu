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

#pragma once

#include "paddle/phi/kernels/affine_grid_grad_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/affine_grid_utils.h"

#include "paddle/phi/kernels/bmm_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/linspace_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T>
__global__ void LinspaceKernel(T start, T step, int64_t size, T* out) {
  CUDA_KERNEL_LOOP_TYPE(index, size, int64_t) {
    out[index] = start + step * index;
  }
}

template <typename T>
struct Linspace<phi::GPUContext, T> {
  void operator()(T start,
                  T end,
                  int count,
                  bool align_corners,
                  DenseTensor* numbers,
                  const phi::GPUContext& dev_ctx) {
    numbers->Resize(common::make_ddim({count}));
    T* number_data = dev_ctx.template Alloc<T>(numbers);
    T slice = (end - start) / (T)(count - 1);
    if (!align_corners) {
      slice = (end - start) / (T)count;
      start *= (T)(count - 1) / (T)count;
    }
    auto stream = dev_ctx.stream();
    int block = 512;
    int grid = (count + block - 1) / block;
    LinspaceKernel<T>
        <<<grid, block, 0, stream>>>(start, slice, count, number_data);
  }
};

template <typename T>
__global__ void affine_grid_grad_kernel_4d(const int64_t count,
                                           int n,
                                           int out_h,
                                           int out_w,
                                           T h_start,
                                           T w_start,
                                           T h_step,
                                           T w_step,
                                           const T* out_grad,  // N, H, W, 2
                                           T* theta_grad) {    // N, 2, 3
  CUDA_KERNEL_LOOP_TYPE(index, count, int64_t) {
    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int n = index / (out_w * out_h);
    T h_coor = h_step * static_cast<T>(h) + static_cast<T>(h_start);
    T w_coor = w_step * static_cast<T>(w) + static_cast<T>(w_start);

    int theta_offset = n * 6;  // 2 * 3;
    T out_grad_x = out_grad[index * 2];
    phi::CudaAtomicAdd(theta_grad + theta_offset, out_grad_x * w_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 1, out_grad_x * h_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 2, out_grad_x);

    T out_grad_y = out_grad[index * 2 + 1];
    phi::CudaAtomicAdd(theta_grad + theta_offset + 3, out_grad_y * w_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 4, out_grad_y * h_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 5, out_grad_y);
  }
}

template <typename T>
__global__ void affine_grid_grad_kernel_5d(const int64_t count,
                                           int n,
                                           int out_d,
                                           int out_h,
                                           int out_w,
                                           T d_start,
                                           T h_start,
                                           T w_start,
                                           T d_step,
                                           T h_step,
                                           T w_step,
                                           const T* out_grad,  // N, D, H, W, 3
                                           T* theta_grad) {    // N, 3, 4
  CUDA_KERNEL_LOOP_TYPE(index, count, int64_t) {
    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int d = (index / (out_w * out_h)) % out_d;
    int n = index / (out_w * out_h * out_d);

    T d_coor = d_step * static_cast<T>(d) + static_cast<T>(d_start);
    T h_coor = h_step * static_cast<T>(h) + static_cast<T>(h_start);
    T w_coor = w_step * static_cast<T>(w) + static_cast<T>(w_start);

    int theta_offset = n * 12;  // 3 * 4;
    T out_grad_x = out_grad[index * 3];
    phi::CudaAtomicAdd(theta_grad + theta_offset, out_grad_x * w_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 1, out_grad_x * h_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 2, out_grad_x * d_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 3, out_grad_x);

    T out_grad_y = out_grad[index * 3 + 1];
    phi::CudaAtomicAdd(theta_grad + theta_offset + 4, out_grad_y * w_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 5, out_grad_y * h_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 6, out_grad_y * d_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 7, out_grad_y);

    T out_grad_z = out_grad[index * 3 + 2];
    phi::CudaAtomicAdd(theta_grad + theta_offset + 8, out_grad_z * w_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 9, out_grad_z * h_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 10, out_grad_z * d_coor);
    phi::CudaAtomicAdd(theta_grad + theta_offset + 11, out_grad_z);
  }
}

template <typename T>
__global__ void CombineBaseGridKernel(const T* w_data,
                                      const T* h_data,
                                      const T* ones_data,
                                      T* base_grid_data,
                                      int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    base_grid_data[idx * 3 + 0] = w_data[idx];
    base_grid_data[idx * 3 + 1] = h_data[idx];
    base_grid_data[idx * 3 + 2] = ones_data[idx];
  }
}

template <typename T, typename Context>
void AffineGridGrad4DCUDAKernel(const Context& dev_ctx,
                                const DenseTensor& output_grad,
                                const IntArray& outputShape,
                                bool align_corners,
                                DenseTensor* input_grad) {
  // output_grad 的形状是 [N, H, W, 2]
  auto grad_grid_dims = output_grad.dims();
  int n = grad_grid_dims[0];
  int h = grad_grid_dims[1];
  int w = grad_grid_dims[2];

  // input_grad (theta的梯度) 的形状应该是 [N, 2, 3]
  input_grad->Resize(common::make_ddim({n, 2, 3}));
  T* grad_theta_data = dev_ctx.template Alloc<T>(input_grad);

  if (output_grad.numel() == 0) {
    phi::Full<T, Context>(dev_ctx,
                          phi::IntArray(common::vectorize(input_grad->dims())),
                          0,
                          input_grad);
    return;
  }

  // 1. 创建基础网格 base_grid，形状为 [N, H, W, 3]
  // 这部分和前向实现完全相同
  DenseTensor base_grid;
  base_grid.Resize(common::make_ddim({n, h, w, 3}));
  dev_ctx.template Alloc<T>(&base_grid);

  // 2. 生成 W 方向的坐标
  DenseTensor w_range;
  if (w <= 1) {
    w_range.Resize(common::make_ddim({1}));
    dev_ctx.template Alloc<T>(&w_range);
    phi::Full<T, Context>(dev_ctx, phi::IntArray({1}), 0, &w_range);
  } else {
    DenseTensor start_w, stop_w, num_w;
    start_w.Resize(common::make_ddim({1}));
    stop_w.Resize(common::make_ddim({1}));
    num_w.Resize(common::make_ddim({1}));
    dev_ctx.template Alloc<T>(&start_w);
    dev_ctx.template Alloc<T>(&stop_w);
    dev_ctx.template Alloc<int32_t>(&num_w);

    phi::Full<T, Context>(dev_ctx, phi::IntArray({1}), -1, &start_w);
    phi::Full<T, Context>(dev_ctx, phi::IntArray({1}), 1, &stop_w);
    phi::Full<int32_t, Context>(dev_ctx, phi::IntArray({1}), w, &num_w);

    phi::LinspaceKernel<T, Context>(
        dev_ctx, start_w, stop_w, num_w, output_grad.dtype(), &w_range);

    if (!align_corners) {
      phi::ScaleKernel<T, Context>(
          dev_ctx, w_range, static_cast<T>(w - 1), 0.0, false, &w_range);
      phi::ScaleKernel<T, Context>(dev_ctx,
                                   w_range,
                                   static_cast<T>(1) / static_cast<T>(w),
                                   0.0,
                                   false,
                                   &w_range);
    }
  }

  // 3. 生成 H 方向的坐标
  DenseTensor h_range;
  if (h <= 1) {
    h_range.Resize(common::make_ddim({1}));
    dev_ctx.template Alloc<T>(&h_range);
    phi::Full<T, Context>(dev_ctx, phi::IntArray({1}), 0, &h_range);
  } else {
    DenseTensor start_h, stop_h, num_h;
    start_h.Resize(common::make_ddim({1}));
    stop_h.Resize(common::make_ddim({1}));
    num_h.Resize(common::make_ddim({1}));
    dev_ctx.template Alloc<T>(&start_h);
    dev_ctx.template Alloc<T>(&stop_h);
    dev_ctx.template Alloc<int32_t>(&num_h);

    phi::Full<T, Context>(dev_ctx, phi::IntArray({1}), -1, &start_h);
    phi::Full<T, Context>(dev_ctx, phi::IntArray({1}), 1, &stop_h);
    phi::Full<int32_t, Context>(dev_ctx, phi::IntArray({1}), h, &num_h);

    phi::LinspaceKernel<T, Context>(
        dev_ctx, start_h, stop_h, num_h, output_grad.dtype(), &h_range);

    if (!align_corners) {
      phi::ScaleKernel<T, Context>(
          dev_ctx, h_range, static_cast<T>(h - 1), 0.0, false, &h_range);
      phi::ScaleKernel<T, Context>(dev_ctx,
                                   h_range,
                                   static_cast<T>(1) / static_cast<T>(h),
                                   0.0,
                                   false,
                                   &h_range);
    }
  }

  // 4. 扩展 w_range 到 [N, H, W]
  DenseTensor w_range_reshaped;
  w_range_reshaped.ShareDataWith(w_range);
  w_range_reshaped.Resize(common::make_ddim({1, 1, w}));

  DenseTensor w_range_expanded;
  phi::ExpandKernel<T, Context>(
      dev_ctx, w_range_reshaped, phi::IntArray({n, h, w}), &w_range_expanded);

  // 5. 扩展 h_range 到 [N, H, W]
  DenseTensor h_range_reshaped;
  h_range_reshaped.ShareDataWith(h_range);
  h_range_reshaped.Resize(common::make_ddim({1, h, 1}));

  DenseTensor h_range_expanded;
  phi::ExpandKernel<T, Context>(
      dev_ctx, h_range_reshaped, phi::IntArray({n, h, w}), &h_range_expanded);

  // 6. 创建全1的张量
  DenseTensor ones;
  ones.Resize(common::make_ddim({n, h, w}));
  dev_ctx.template Alloc<T>(&ones);
  phi::Full<T, Context>(dev_ctx, phi::IntArray({n, h, w}), 1, &ones);

  // 7. 将三个分量组合成 base_grid
  const T* w_data = w_range_expanded.data<T>();
  const T* h_data = h_range_expanded.data<T>();
  const T* ones_data = ones.data<T>();
  T* base_grid_data = base_grid.data<T>();

  int total_elements = n * h * w;
  auto stream = dev_ctx.stream();

  int block_size = 512;
  int grid_size = (total_elements + block_size - 1) / block_size;

  CombineBaseGridKernel<T><<<grid_size, block_size, 0, stream>>>(
      w_data, h_data, ones_data, base_grid_data, total_elements);

  // 8. 重塑 base_grid 为 [N, H*W, 3]
  DenseTensor base_grid_reshaped;
  base_grid_reshaped.ShareDataWith(base_grid);
  base_grid_reshaped.Resize(common::make_ddim({n, h * w, 3}));

  // 9. 转置 base_grid: [N, H*W, 3] -> [N, 3, H*W]
  DenseTensor base_grid_transposed;
  base_grid_transposed.Resize(common::make_ddim({n, 3, h * w}));
  phi::TransposeKernel<T, Context>(
      dev_ctx, base_grid_reshaped, {0, 2, 1}, &base_grid_transposed);

  // 10. 重塑 output_grad 为 [N, H*W, 2]
  DenseTensor grad_grid_reshaped;
  grad_grid_reshaped.ShareDataWith(output_grad);
  grad_grid_reshaped.Resize(common::make_ddim({n, h * w, 2}));

  // 11. 批量矩阵乘法: [N, 3, H*W] x [N, H*W, 2] = [N, 3, 2]
  DenseTensor grad_theta_temp;
  grad_theta_temp.Resize(common::make_ddim({n, 3, 2}));

  phi::BmmKernel<T, Context>(
      dev_ctx, base_grid_transposed, grad_grid_reshaped, &grad_theta_temp);

  // 12. 转置得到最终结果: [N, 3, 2] -> [N, 2, 3]
  phi::TransposeKernel<T, Context>(
      dev_ctx, grad_theta_temp, {0, 2, 1}, input_grad);
}

template <typename T, typename Context>
void AffineGridGrad5DCUDAKernel(const Context& dev_ctx,
                                const DenseTensor& output_grad,
                                const IntArray& outputShape,
                                bool align_corners,
                                DenseTensor* input_grad) {
  // VLOG(0) << "in affine grid backward 5D";
  auto& theta_grad = input_grad;
  int n = output_grad.dims()[0];
  auto& size_attr = outputShape.GetData();
  int d = 0;
  int h = 0;
  int w = 0;
  d = size_attr[2];
  h = size_attr[3];
  w = size_attr[4];
  theta_grad->Resize(common::make_ddim({n, 3, 4}));
  T* theta_grad_data = dev_ctx.template Alloc<T>(theta_grad);
  phi::funcs::SetConstant<phi::GPUContext, T>()(
      dev_ctx, theta_grad, static_cast<T>(0));

  T d_step;
  T h_step;
  T w_step;
  T d_start = -1;
  T h_start = -1;
  T w_start = -1;
  if (align_corners) {
    d_step = static_cast<T>(2) / static_cast<T>(d - 1);
    h_step = static_cast<T>(2) / static_cast<T>(h - 1);
    w_step = static_cast<T>(2) / static_cast<T>(w - 1);
  } else {
    d_step = static_cast<T>(2) / static_cast<T>(d);
    h_step = static_cast<T>(2) / static_cast<T>(h);
    w_step = static_cast<T>(2) / static_cast<T>(w);

    d_start *= static_cast<T>(d - 1) / static_cast<T>(d);
    h_start *= static_cast<T>(h - 1) / static_cast<T>(h);
    w_start *= static_cast<T>(w - 1) / static_cast<T>(w);
  }
  const int count = n * d * h * w;
  int block = 512;
  int grid = (count + block - 1) / block;
  auto cu_stream = dev_ctx.stream();
  affine_grid_grad_kernel_5d<<<grid, block, 0, cu_stream>>>(
      count,
      n,
      d,
      h,
      w,
      d_start,
      h_start,
      w_start,
      d_step,
      h_step,
      w_step,
      output_grad.data<T>(),
      theta_grad_data);
}

template <typename T, typename Context>
void AffineGridGradCUDAKernel(const Context& dev_ctx,
                              const DenseTensor& input,
                              const IntArray& outputShape,
                              bool align_corners,
                              DenseTensor* output) {
  auto* theta = &input;
  auto theta_size = theta->dims().size();
  if (output->numel() == 0 || input.numel() == 0) {
    dev_ctx.template Alloc<T>(output);
    phi::funcs::SetConstant<phi::GPUContext, T>()(
        dev_ctx, output, static_cast<T>(0));
    return;
  }
  if (theta_size == 4) {
    AffineGridGrad4DCUDAKernel<T, Context>(
        dev_ctx, input, outputShape, align_corners, output);
  } else {
    AffineGridGrad5DCUDAKernel<T, Context>(
        dev_ctx, input, outputShape, align_corners, output);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(affine_grid_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AffineGridGradCUDAKernel,
                   float,
                   double){};
