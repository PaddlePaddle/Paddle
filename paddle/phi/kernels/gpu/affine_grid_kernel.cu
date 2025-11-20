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
#include "paddle/phi/kernels/bmm_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/affine_grid_utils.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void AffineGrid4DCUDAKernel(const Context& dev_ctx,
                            const DenseTensor& input,
                            const IntArray& outputShape,
                            bool align_corners,
                            DenseTensor* output) {
  auto* theta = &input;
  int64_t n = theta->dims()[0];
  auto& size_attr = outputShape.GetData();
  int64_t h = size_attr[2];
  int64_t w = size_attr[3];

  if (input.numel() == 0) {
    output->Resize(common::make_ddim({n, h, w, 2}));
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(output->dims())), 0, output);
    return;
  }

  // Directly create the base mesh
  DenseTensor base_grid;
  base_grid.Resize(common::make_ddim({n, h, w, 3}));
  T* base_grid_data = dev_ctx.template Alloc<T>(&base_grid);

  phi::funcs::CreateBaseGridKernel_4D<T, Context>(
      dev_ctx, base_grid_data, n, h, w, align_corners);

  // Apply affine transformation
  DenseTensor base_grid_new;
  base_grid_new.ShareDataWith(base_grid);
  base_grid_new.Resize(common::make_ddim({n, h * w, 3}));

  // Transpose theta: [N, 2, 3] -> [N, 3, 2]
  DenseTensor theta_transposed;
  theta_transposed.Resize(common::make_ddim({n, 3, 2}));
  phi::TransposeKernel<T, Context>(
      dev_ctx, input, {0, 2, 1}, &theta_transposed);

  DenseTensor grid_flat;
  grid_flat.Resize(common::make_ddim({n, h * w, 2}));
  phi::BmmKernel<T, Context>(
      dev_ctx, base_grid_new, theta_transposed, &grid_flat);

  // Reshaping Output
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
  int64_t n = theta->dims()[0];
  auto& size_attr = outputShape.GetData();
  int64_t d = size_attr[2];  // depth
  int64_t h = size_attr[3];  // height
  int64_t w = size_attr[4];  // width

  if (input.numel() == 0) {
    output->Resize(common::make_ddim({n, d, h, w, 3}));
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(output->dims())), 0, output);
    return;
  }

  // Create a basic grid
  DenseTensor base_grid;
  base_grid.Resize(common::make_ddim({n, d, h, w, 4}));
  T* base_grid_data = dev_ctx.template Alloc<T>(&base_grid);

  phi::funcs::CreateBaseGridKernel_5D<T, Context>(
      dev_ctx, base_grid_data, n, d, h, w, align_corners);

  // Apply affine transformation
  DenseTensor base_grid_new;
  base_grid_new.ShareDataWith(base_grid);
  base_grid_new.Resize(common::make_ddim({n, d * h * w, 4}));

  // Transpose theta: [N, 3, 4] -> [N, 4, 3]
  DenseTensor theta_transposed;
  theta_transposed.Resize(common::make_ddim({n, 4, 3}));
  phi::TransposeKernel<T, Context>(
      dev_ctx, input, {0, 2, 1}, &theta_transposed);

  // Perform batch matrix multiplication
  DenseTensor grid_flat;
  grid_flat.Resize(common::make_ddim({n, d * h * w, 3}));
  phi::BmmKernel<T, Context>(
      dev_ctx, base_grid_new, theta_transposed, &grid_flat);

  // Reshaping Output
  output->ShareDataWith(grid_flat);
  output->Resize(common::make_ddim({n, d, h, w, 3}));
}

template <typename T, typename Context>
void AffineGridCUDAKernel(const Context& dev_ctx,
                          const DenseTensor& input,
                          const IntArray& outputShape,
                          bool align_corners,
                          DenseTensor* output) {
  auto* theta = &input;
  int64_t theta_h = theta->dims()[1];
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
