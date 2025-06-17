// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <cassert>
#include <vector>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"  // NOLINT

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/legacy/gpu/layer_norm_cuda_kernel.h"  // NOLINT

namespace phi {

static void GetRowsCols(const std::vector<int64_t> &shape,
                        int *p_rows,
                        int *p_cols) {
  int rows = 1;
  for (int i = 0; i + 1 < shape.size(); ++i) {
    rows *= shape[i];
  }
  int cols = shape[shape.size() - 1];
  *p_rows = rows;
  *p_cols = cols;
}

template <typename T, typename Context>
void RMSLnFwd(const Context &dev_ctx,
              const DenseTensor &x,
              const DenseTensor &scale,
              float epsilon,
              DenseTensor *y,
              DenseTensor *invvar) {
  const auto &scale_shape = scale.dims();
  int rows, cols;
  GetRowsCols(common::vectorize(x.dims()), &rows, &cols);
  if (scale.dtype() == phi::DataType::BFLOAT16)
    dev_ctx.template Alloc<phi::bfloat16>(y);
  else if (scale.dtype() == phi::DataType::FLOAT32)
    dev_ctx.template Alloc<float>(y);
  invvar->Resize({rows});
  dev_ctx.template Alloc<float>(invvar);
  cuda_rms_norm<T, Context>(dev_ctx, x, scale, rows, cols, epsilon, y, invvar);
}

template <typename T, typename Context>
void RMSLnBwd(const Context &dev_ctx,
              const DenseTensor &x,
              const DenseTensor &scale,
              const DenseTensor &invvar,
              const DenseTensor &y_grad,
              float epsilon,
              DenseTensor *x_grad,
              DenseTensor *scale_grad) {
  int rows, cols;
  GetRowsCols(common::vectorize(x.dims()), &rows, &cols);
  dev_ctx.template Alloc<T>(x_grad);
  if (scale_grad) {
    if (scale.dtype() == phi::DataType::BFLOAT16) {
      dev_ctx.template Alloc<phi::bfloat16>(scale_grad);
    } else if (scale.dtype() == phi::DataType::FLOAT32) {
      dev_ctx.template Alloc<float>(scale_grad);
    }
  } else {
    // lora specific
    if (scale.dtype() == phi::DataType::BFLOAT16) {
      DenseTensor scale_grad_tmp =
          phi::EmptyLike<phi::bfloat16, Context>(dev_ctx, scale);
      cuda_rms_norm_gradient<T, Context>(dev_ctx,
                                         x,
                                         scale,
                                         invvar,
                                         y_grad,
                                         rows,
                                         cols,
                                         epsilon,
                                         x_grad,
                                         &scale_grad_tmp);
    } else if (scale.dtype() == phi::DataType::FLOAT32) {
      DenseTensor scale_grad_tmp =
          phi::EmptyLike<float, Context>(dev_ctx, scale);
      cuda_rms_norm_gradient<T, Context>(dev_ctx,
                                         x,
                                         scale,
                                         invvar,
                                         y_grad,
                                         rows,
                                         cols,
                                         epsilon,
                                         x_grad,
                                         &scale_grad_tmp);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_rms_norm_ext,
                   GPU,
                   ALL_LAYOUT,
                   phi::RMSLnFwd,
                   float,
                   double,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(fused_rms_norm_ext_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RMSLnBwd,
                   float,
                   double,
                   phi::dtype::bfloat16) {}
