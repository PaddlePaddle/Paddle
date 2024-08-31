// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/clip_by_norm_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ClipByNormKernel(const Context& dev_ctx,
                      const DenseTensor& in,
                      float max_norm,
                      DenseTensor* output) {
  auto input = &in;
  dev_ctx.template Alloc<T>(output);

  PADDLE_ENFORCE_NOT_NULL(input,
                          common::errors::InvalidArgument(
                              "Input(X) of ClipByNormOp should not be null. "
                              "Please check if it is created correctly."));

  const auto& x_dims = input->dims();
  std::vector<int> xshape(x_dims.size());
  std::vector<int> rdims(x_dims.size());
  for (int i = 0; i < x_dims.size(); i++) {
    xshape[i] = x_dims[i];
    rdims[i] = i;
  }
  int r = xpu::clip_by_norm<T>(dev_ctx.x_context(),
                               input->data<T>(),
                               output->data<T>(),
                               max_norm,
                               xshape,
                               rdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip_by_norm");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    clip_by_norm, XPU, ALL_LAYOUT, phi::ClipByNormKernel, float) {}
