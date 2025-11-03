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

#include "paddle/phi/kernels/affine_grid_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/affine_grid_utils.h"

#include "paddle/phi/kernels/bmm_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/linspace_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

// template <typename T>
// __global__ void LinspaceKernel(T start, T step, int64_t size, T* out) {
//   CUDA_KERNEL_LOOP(index, size) { out[index] = start + step * index; }
// }

// template <typename T>
// struct Linspace<phi::GPUContext, T> {
//   void operator()(T start,
//                   T end,
//                   int count,
//                   bool align_corners,
//                   DenseTensor* numbers,
//                   const phi::GPUContext& dev_ctx) {
//     numbers->Resize(common::make_ddim({count}));
//     T* number_data = dev_ctx.template Alloc<T>(numbers);
//     T slice = (end - start) / (T)(count - 1);
//     if (!align_corners) {
//       slice = (end - start) / (T)count;
//       start *= (T)(count - 1) / (T)count;
//     }
//     auto stream = dev_ctx.stream();
//     int block = 512;
//     int grid = (count + block - 1) / block;
//     LinspaceKernel<T>
//         <<<grid, block, 0, stream>>>(start, slice, count, number_data);
//   }
// };

template <typename T>
__global__ void affine_grid_kernel_4d(const int count,
                                      int n,
                                      int out_h,
                                      int out_w,
                                      T h_start,
                                      T w_start,
                                      T h_step,
                                      T w_step,
                                      const T* theta,  // N, 2, 3
                                      T* output) {
  CUDA_KERNEL_LOOP(index, count) {
    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int n = index / (out_w * out_h);

    T h_coor = h_step * static_cast<T>(h) + static_cast<T>(h_start);
    T w_coor = w_step * static_cast<T>(w) + static_cast<T>(w_start);

    int theta_offset = n * 6;  // 2 * 3;
    // affine from (h_coor, w_coor) to (x, y)
    output[index * 2] = theta[theta_offset] * w_coor +
                        theta[theta_offset + 1] * h_coor +
                        theta[theta_offset + 2];
    output[index * 2 + 1] = theta[theta_offset + 3] * w_coor +
                            theta[theta_offset + 4] * h_coor +
                            theta[theta_offset + 5];
  }
}

template <typename T>
__global__ void affine_grid_kernel_5d(const int count,
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
                                      const T* theta,  // N, 3, 4
                                      T* output) {
  CUDA_KERNEL_LOOP(index, count) {
    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int d = (index / (out_w * out_h)) % out_d;
    int n = index / (out_w * out_h * out_d);

    T d_coor = d_step * static_cast<T>(d) + static_cast<T>(d_start);
    T h_coor = h_step * static_cast<T>(h) + static_cast<T>(h_start);
    T w_coor = w_step * static_cast<T>(w) + static_cast<T>(w_start);

    int theta_offset = n * 12;  // 3 * 4
    // affine from (h_coor, w_coor) to (x, y)
    output[index * 3] =
        theta[theta_offset] * w_coor + theta[theta_offset + 1] * h_coor +
        theta[theta_offset + 2] * d_coor + theta[theta_offset + 3];
    output[index * 3 + 1] =
        theta[theta_offset + 4] * w_coor + theta[theta_offset + 5] * h_coor +
        theta[theta_offset + 6] * d_coor + theta[theta_offset + 7];
    output[index * 3 + 2] =
        theta[theta_offset + 8] * w_coor + theta[theta_offset + 9] * h_coor +
        theta[theta_offset + 10] * d_coor + theta[theta_offset + 11];
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
void AffineGrid4DCUDAKernel(const Context& dev_ctx,
                            const DenseTensor& input,
                            const IntArray& outputShape,
                            bool align_corners,
                            DenseTensor* output) {
  auto* theta = &input;
  int n = theta->dims()[0];
  auto& size_attr = outputShape.GetData();
  int h = 0;
  int w = 0;
  h = size_attr[2];
  w = size_attr[3];
  output->Resize(common::make_ddim({n, h, w, 2}));
  T* out_data = dev_ctx.template Alloc<T>(output);
  if (input.numel() == 0) {
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(output->dims())), 0, output);
    return;
  }

  // 1. 创建基础网格 base_grid，形状为 [N, H, W, 3]
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
        dev_ctx, start_w, stop_w, num_w, theta->dtype(), &w_range);

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
        dev_ctx, start_h, stop_h, num_h, theta->dtype(), &h_range);

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
  // 首先将 w_range 从 [w] 重塑为 [1, 1, w]
  DenseTensor w_range_reshaped;
  w_range_reshaped.ShareDataWith(w_range);
  w_range_reshaped.Resize(common::make_ddim({1, 1, w}));

  DenseTensor w_range_expanded;
  phi::ExpandKernel<T, Context>(
      dev_ctx, w_range_reshaped, phi::IntArray({n, h, w}), &w_range_expanded);

  // 5. 扩展 h_range 到 [N, H, W]
  // 首先将 h_range 从 [h] 重塑为 [1, h, 1]
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

  // 使用标准的CUDA kernel启动方式
  int block_size = 512;
  int grid_size = (total_elements + block_size - 1) / block_size;
  CombineBaseGridKernel<T><<<grid_size, block_size, 0, stream>>>(
      w_data, h_data, ones_data, base_grid_data, total_elements);

  // 8. 重塑 base_grid 为 [N, H*W, 3]
  DenseTensor base_grid_reshaped;
  base_grid_reshaped.ShareDataWith(base_grid);  //--
  base_grid_reshaped.Resize(common::make_ddim({n, h * w, 3}));
  // phi::Full<T, Context>(dev_ctx, phi::IntArray({n, h*w, 3}), 1,
  // &base_grid_reshaped);//---

  // 9. 转置 theta: [N, 2, 3] -> [N, 3, 2]
  DenseTensor theta_transposed;
  theta_transposed.Resize(common::make_ddim({n, 3, 2}));
  phi::TransposeKernel<T, Context>(
      dev_ctx, *theta, {0, 2, 1}, &theta_transposed);

  // 10. 批量矩阵乘法: [N, H*W, 3] x [N, 3, 2] = [N, H*W, 2]
  DenseTensor grid_flat;
  grid_flat.Resize(common::make_ddim({n, h * w, 2}));

  // 使用 paddle 的 bmm 操作
  phi::BmmKernel<T, Context>(
      dev_ctx, base_grid_reshaped, theta_transposed, &grid_flat);

  // 11. 重塑输出为 [N, H, W, 2]
  output->ShareDataWith(grid_flat);
  output->Resize(common::make_ddim({n, h, w, 2}));
}

template <typename T, typename Context>
void AffineGrid5DCUDAKernel(const Context& dev_ctx,
                            const DenseTensor& input,
                            const IntArray& outputShape,
                            bool align_corners,
                            DenseTensor* output) {
  auto* theta = &input;
  int n = theta->dims()[0];
  auto& size_attr = outputShape.GetData();
  int d = 0;
  int h = 0;
  int w = 0;
  d = size_attr[2];
  h = size_attr[3];
  w = size_attr[4];
  output->Resize(common::make_ddim({n, d, h, w, 3}));
  T* out_data = dev_ctx.template Alloc<T>(output);
  if (input.numel() == 0) {
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(output->dims())), 0, output);
    return;
  }

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
  affine_grid_kernel_5d<<<grid, block, 0, cu_stream>>>(
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
      theta->data<T>(),  // N, 3, 4
      out_data);
}

template <typename T, typename Context>
void AffineGridCUDAKernel(const Context& dev_ctx,
                          const DenseTensor& input,
                          const IntArray& outputShape,
                          bool align_corners,
                          DenseTensor* output) {
  auto* theta = &input;
  int theta_h = theta->dims()[1];
  if (theta_h == 2) {
    AffineGrid4DCUDAKernel<T, Context>(
        dev_ctx, input, outputShape, align_corners, output);
  } else {
    AffineGrid5DCUDAKernel<T, Context>(
        dev_ctx, input, outputShape, align_corners, output);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    affine_grid, GPU, ALL_LAYOUT, phi::AffineGridCUDAKernel, float, double){};
