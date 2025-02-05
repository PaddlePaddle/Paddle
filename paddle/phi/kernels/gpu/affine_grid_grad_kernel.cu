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

namespace phi {

template <typename T>
__global__ void LinspaceKernel(T start, T step, int64_t size, T* out) {
  CUDA_KERNEL_LOOP(index, size) { out[index] = start + step * index; }
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
__global__ void affine_grid_grad_kernel_4d(const int count,
                                           int n,
                                           int out_h,
                                           int out_w,
                                           T h_start,
                                           T w_start,
                                           T h_step,
                                           T w_step,
                                           const T* out_grad,  // N, H, W, 2
                                           T* theta_grad) {    // N, 2, 3
  CUDA_KERNEL_LOOP(index, count) {
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
__global__ void affine_grid_grad_kernel_5d(const int count,
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
  CUDA_KERNEL_LOOP(index, count) {
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

template <typename T, typename Context>
void AffineGridGrad4DCUDAKernel(const Context& dev_ctx,
                                const DenseTensor& output_grad,
                                const IntArray& outputShape,
                                bool align_corners,
                                DenseTensor* input_grad) {
  auto& theta_grad = input_grad;
  int n = output_grad.dims()[0];
  auto& size_attr = outputShape.GetData();
  int h = 0;
  int w = 0;
  h = size_attr[2];
  w = size_attr[3];
  theta_grad->Resize(common::make_ddim({n, 2, 3}));
  T* theta_grad_data = dev_ctx.template Alloc<T>(theta_grad);
  phi::funcs::SetConstant<phi::GPUContext, T>()(
      dev_ctx, theta_grad, static_cast<T>(0));

  T h_step;
  T w_step;
  T h_start = -1;
  T w_start = -1;
  if (align_corners) {
    h_step = static_cast<T>(2) / static_cast<T>(h - 1);
    w_step = static_cast<T>(2) / static_cast<T>(w - 1);
  } else {
    h_step = static_cast<T>(2) / static_cast<T>(h);
    w_step = static_cast<T>(2) / static_cast<T>(w);

    h_start *= static_cast<T>(h - 1) / static_cast<T>(h);
    w_start *= static_cast<T>(w - 1) / static_cast<T>(w);
  }
  const int count = n * h * w;
  VLOG(3) << "count: " << count << "; h_step: " << h_step
          << "; w_step: " << w_step << "; h_start: " << h_start
          << "; w_start: " << w_start;
  int block = 512;
  int grid = (count + block - 1) / block;
  auto cu_stream = dev_ctx.stream();
  affine_grid_grad_kernel_4d<<<grid, block, 0, cu_stream>>>(
      count,
      n,
      h,
      w,
      h_start,
      w_start,
      h_step,
      w_step,
      output_grad.data<T>(),
      theta_grad_data);
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
