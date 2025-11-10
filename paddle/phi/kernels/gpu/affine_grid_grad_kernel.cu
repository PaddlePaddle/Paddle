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
#include "paddle/phi/kernels/bmm_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/affine_grid_utils.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void AffineGridGrad4DCUDAKernel(const Context& dev_ctx,
                                const DenseTensor& output_grad,
                                const IntArray& outputShape,
                                bool align_corners,
                                DenseTensor* input_grad) {
  // The shape of the output grad is [N, H, W, 2]
  auto grad_grid_dims = output_grad.dims();
  int64_t n = grad_grid_dims[0];
  int64_t h = grad_grid_dims[1];
  int64_t w = grad_grid_dims[2];

  // The shape of input_grad (theta gradient) should be [N, 2, 3]
  input_grad->Resize(common::make_ddim({n, 2, 3}));
  T* grad_theta_data = dev_ctx.template Alloc<T>(input_grad);

  if (output_grad.numel() == 0) {
    phi::Full<T, Context>(dev_ctx,
                          phi::IntArray(common::vectorize(input_grad->dims())),
                          0,
                          input_grad);
    return;
  }

  // 1. Directly create the basic grid using the same kernel as the forward
  // direction
  DenseTensor base_grid;
  base_grid.Resize(common::make_ddim({n, h, w, 3}));
  T* base_grid_data = dev_ctx.template Alloc<T>(&base_grid);

  int64_t total_elements = n * h * w;
  auto stream = dev_ctx.stream();
  int64_t block_size = 512;
  int64_t grid_size = (total_elements + block_size - 1) / block_size;

  phi::funcs::CreateBaseGridKernel_4D<T><<<grid_size, block_size, 0, stream>>>(
      base_grid_data, n, h, w, align_corners);

  // 2. Reshaping base_grid to [N, H * W, 3]
  DenseTensor base_grid_reshaped;
  base_grid_reshaped.ShareDataWith(base_grid);
  base_grid_reshaped.Resize(common::make_ddim({n, h * w, 3}));

  // 3. Transposition base_grid: [N, H * W, 3] ->[N, 3, H * W]
  DenseTensor base_grid_transposed;
  base_grid_transposed.Resize(common::make_ddim({n, 3, h * w}));
  phi::TransposeKernel<T, Context>(
      dev_ctx, base_grid_reshaped, {0, 2, 1}, &base_grid_transposed);

  // 4. Reshaping Output_grad to [N, H * W, 2]
  DenseTensor grad_grid_reshaped;
  grad_grid_reshaped.ShareDataWith(output_grad);
  grad_grid_reshaped.Resize(common::make_ddim({n, h * w, 2}));

  // 5. Batch matrix multiplication: [N, 3, H * W] x [N, H * W, 2]=[N, 3, 2]
  DenseTensor grad_theta_temp;
  grad_theta_temp.Resize(common::make_ddim({n, 3, 2}));

  phi::BmmKernel<T, Context>(
      dev_ctx, base_grid_transposed, grad_grid_reshaped, &grad_theta_temp);

  // 6. Transposition yields the final result: [N, 3, 2] ->[N, 2, 3]
  phi::TransposeKernel<T, Context>(
      dev_ctx, grad_theta_temp, {0, 2, 1}, input_grad);
}

template <typename T, typename Context>
void AffineGridGrad5DCUDAKernel(const Context& dev_ctx,
                                const DenseTensor& output_grad,
                                const IntArray& outputShape,
                                bool align_corners,
                                DenseTensor* input_grad) {
  // The shape of the output grad is [N, D, H, W, 3]
  auto grad_grid_dims = output_grad.dims();
  int64_t n = grad_grid_dims[0];
  int64_t d = grad_grid_dims[1];
  int64_t h = grad_grid_dims[2];
  int64_t w = grad_grid_dims[3];

  // The shape of input_grad (theta gradient) should be [N, 3, 4]
  input_grad->Resize(common::make_ddim({n, 3, 4}));
  T* grad_theta_data = dev_ctx.template Alloc<T>(input_grad);

  if (output_grad.numel() == 0) {
    phi::Full<T, Context>(dev_ctx,
                          phi::IntArray(common::vectorize(input_grad->dims())),
                          0,
                          input_grad);
    return;
  }

  // 1. Directly create the basic grid using the same kernel as the forward
  // direction
  DenseTensor base_grid;
  base_grid.Resize(common::make_ddim({n, d, h, w, 4}));
  T* base_grid_data = dev_ctx.template Alloc<T>(&base_grid);

  int64_t total_elements = n * d * h * w;
  auto stream = dev_ctx.stream();
  int64_t block_size = 512;
  int64_t grid_size = (total_elements + block_size - 1) / block_size;

  phi::funcs::CreateBaseGridKernel_5D<T><<<grid_size, block_size, 0, stream>>>(
      base_grid_data, n, d, h, w, align_corners);

  // 2. Reshaping base_grid to [N, D * H * W, 4]
  DenseTensor base_grid_reshaped;
  base_grid_reshaped.ShareDataWith(base_grid);
  base_grid_reshaped.Resize(common::make_ddim({n, d * h * w, 4}));

  // 3. Transpose base_grid:[N，D*H*W，4]->[N，4，D*H*W]
  DenseTensor base_grid_transposed;
  base_grid_transposed.Resize(common::make_ddim({n, 4, d * h * w}));
  phi::TransposeKernel<T, Context>(
      dev_ctx, base_grid_reshaped, {0, 2, 1}, &base_grid_transposed);

  // 4. Reshaping Output_grad to [N, D * H * W, 3]
  DenseTensor grad_grid_reshaped;
  grad_grid_reshaped.ShareDataWith(output_grad);
  grad_grid_reshaped.Resize(common::make_ddim({n, d * h * w, 3}));

  // 5. Batch matrix multiplication: [N, 4, D * H * W] x [N, D * H * W, 3]=[N,
  // 4, 3]
  DenseTensor grad_theta_temp;
  grad_theta_temp.Resize(common::make_ddim({n, 4, 3}));

  phi::BmmKernel<T, Context>(
      dev_ctx, base_grid_transposed, grad_grid_reshaped, &grad_theta_temp);

  // 6. Transposition yields the final result: [N, 4, 3] ->[N, 3, 4]
  phi::TransposeKernel<T, Context>(
      dev_ctx, grad_theta_temp, {0, 2, 1}, input_grad);
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
