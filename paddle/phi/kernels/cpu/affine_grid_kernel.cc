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

#include "paddle/phi/kernels/affine_grid_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/affine_grid_utils.h"

namespace phi {

template <typename T>
struct Linspace<phi::CPUContext, T> {
  void operator()(T start,
                  T end,
                  int count,
                  bool align_corners,
                  DenseTensor* numbers,
                  const phi::CPUContext& dev_ctx) {
    numbers->Resize(common::make_ddim({count}));
    T* number_data = dev_ctx.template Alloc<T>(numbers);
    T slice = (end - start) / (T)(count - 1);
    if (!align_corners) {
      slice = (end - start) / (T)count;
      start *= (T)(count - 1) / (T)count;
    }
    for (int i = 0; i < count; ++i) {
      number_data[i] = start + (T)i * slice;
    }
  }
};

template <typename T, typename Context>
void AffineGrid4DKernel(const Context& dev_ctx,
                        const DenseTensor& input,
                        const IntArray& outputShape,
                        bool align_corners,
                        DenseTensor* output) {
  auto* theta = &input;
  int n = static_cast<int>(theta->dims()[0]);
  auto& size_attr = outputShape.GetData();
  int h = 0;
  int w = 0;
  h = static_cast<int>(size_attr[2]);
  w = static_cast<int>(size_attr[3]);
  output->Resize(common::make_ddim({n, h, w, 2}));
  dev_ctx.template Alloc<T>(output);
  phi::funcs::SetConstant<Context, T>()(dev_ctx, output, static_cast<T>(0));
  DenseTensor grid;
  GetIdxMap4D<Context, T>(n, h, w, align_corners, &grid, dev_ctx);
  // output = grid * theta.T
  // TODO(wanghaoshuang): Refine batched matrix multiply
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  for (int i = 0; i < n; ++i) {
    DenseTensor sliced_grid = grid.Slice(i, i + 1).Resize(
        {static_cast<int64_t>(h) * static_cast<int64_t>(w), 3});
    DenseTensor sliced_theta = theta->Slice(i, i + 1).Resize({2, 3});
    DenseTensor sliced_out = output->Slice(i, i + 1).Resize(
        {static_cast<int64_t>(h) * static_cast<int64_t>(w), 2});
    blas.MatMul(
        sliced_grid, false, sliced_theta, true, T(1), &sliced_out, T(0));
  }
}

template <typename T, typename Context>
void AffineGrid5DKernel(const Context& dev_ctx,
                        const DenseTensor& input,
                        const IntArray& outputShape,
                        bool align_corners,
                        DenseTensor* output) {
  auto* theta = &input;
  int n = static_cast<int>(theta->dims()[0]);
  auto& size_attr = outputShape.GetData();
  int d = 0;
  int h = 0;
  int w = 0;
  d = static_cast<int>(size_attr[2]);
  h = static_cast<int>(size_attr[3]);
  w = static_cast<int>(size_attr[4]);
  output->Resize(common::make_ddim({n, d, h, w, 3}));
  dev_ctx.template Alloc<T>(output);
  phi::funcs::SetConstant<Context, T>()(dev_ctx, output, static_cast<T>(0));
  DenseTensor grid;
  GetIdxMap5D<Context, T>(n, d, h, w, align_corners, &grid, dev_ctx);
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  for (int i = 0; i < n; ++i) {
    DenseTensor sliced_grid = grid.Slice(i, i + 1).Resize(
        {static_cast<int64_t>(d) * static_cast<int64_t>(h) *
             static_cast<int64_t>(w),
         4});
    DenseTensor sliced_theta = theta->Slice(i, i + 1).Resize({3, 4});
    DenseTensor sliced_out = output->Slice(i, i + 1).Resize(
        {static_cast<int64_t>(d) * static_cast<int64_t>(h) *
             static_cast<int64_t>(w),
         3});
    blas.MatMul(
        sliced_grid, false, sliced_theta, true, T(1), &sliced_out, T(0));
  }
}

template <typename T, typename Context>
void AffineGridKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const IntArray& outputShape,
                      bool align_corners,
                      DenseTensor* output) {
  auto& size_attr = outputShape.GetData();
  if (size_attr.size() == 4) {
    AffineGrid4DKernel<T, Context>(
        dev_ctx, input, outputShape, align_corners, output);
  } else {
    AffineGrid5DKernel<T, Context>(
        dev_ctx, input, outputShape, align_corners, output);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    affine_grid, CPU, ALL_LAYOUT, phi::AffineGridKernel, float, double){};
