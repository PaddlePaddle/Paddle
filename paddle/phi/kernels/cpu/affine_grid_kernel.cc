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

#include "paddle/phi/kernels/affine_grid.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/backends/cpu/cpu_context.h"

namespace phi {

template <typename T>
struct Linspace<phi::CPUContext, T> {
  void operator()(T start,
                  T end,
                  int count,
                  bool align_corners,
                  DenseTensor* numbers,
                  const Context& dev_ctx) {
    numbers->Resize(phi::make_ddim({count}));
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
void AffineGridKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const paddle::optional<DenseTensor>& outputShape,
                      bool align_corners,
                      std::vector<int> output_shape,
                      DenseTensor* output) {
  //auto* theta = ctx.Input<Tensor>("Theta");
  auto* theta = &input;
  int n = theta->dims()[0];
  auto &size_attr = output_shape;
  int h = 0;
  int w = 0;
  if (size_attr.size() == 0) {
    //auto* output_shape = ctx.Input<Tensor>("OutputShape");
    DenseTensor h_sizes;
    phi::Copy(outputShape, phi::CPUPlace(), &h_sizes);
    const int* h_size_data = h_sizes.data<int>();
    h = h_size_data[2];
    w = h_size_data[3];
  } else {
    h = size_attr[2];
    w = size_attr[3];
  }
  output->Resize(phi::make_ddim({n, h, w, 2}));
  T* output_data = dev_ctx.template Alloc<T>(output);
  //output->mutable_data<T>({n, h, w, 2}, ctx.GetPlace());
  phi::funcs::SetConstant<Context, T>()(
      dev_ctx,
      output_data,
      static_cast<T>(0));
  DenseTensor grid;
  GetIdxMap<Context, T>(n, h, w, align_corners, &grid, dev_ctx);
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

}  // namespace phi

PD_REGISTER_KERNEL(affine_grid,
                   CPU,
                   ALL_LAYOUT,
                   phi::AffineGridKernel,
                   float,
                   double) {};
