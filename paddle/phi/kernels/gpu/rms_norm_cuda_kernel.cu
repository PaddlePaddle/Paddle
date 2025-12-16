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
#include "paddle/phi/kernels/gpu/rms_norm_cuda_kernel.h"  // NOLINT

#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
#include "paddle/phi/kernels/funcs/fast_ln_v2.h"
#endif

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
  if (scale.dtype() == phi::DataType::BFLOAT16) {
    dev_ctx.template Alloc<phi::bfloat16>(y);
  } else if (scale.dtype() == phi::DataType::FLOAT32) {
    dev_ctx.template Alloc<float>(y);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of scale must be FLOAT32, BFLOAT16, but got [%s]",
        scale.dtype()));
  }
  invvar->Resize({rows});
  dev_ctx.template Alloc<float>(invvar);
  cuda_rms_norm<T, Context>(dev_ctx, x, scale, rows, cols, epsilon, y, invvar);
}

template <typename T, typename Context>
void RMSNormFwdKernel(const Context &dev_ctx,
                      const DenseTensor &x,
                      const DenseTensor &scale,
                      float epsilon,
                      DenseTensor *y,
                      DenseTensor *invvar) {
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
  auto input_type = x.type();
  auto weight_type = scale.type();
  auto output_type = weight_type;
  auto compute_type = paddle::DataType::FLOAT32;
  auto hidden_size = scale.numel();

  // using fast_ln_v2 only sm > 70
  auto prop = funcs::fast_ln_v2::GetDeviceProp();
  bool has_fast_rms_norm = funcs::fast_ln_v2::has_fast_ln_v2_fwd_kernel(
      weight_type, input_type, output_type, compute_type, hidden_size);
  if (prop->major > 7 && has_fast_rms_norm) {
    auto sizes = x.dims();
    PD_CHECK(sizes.size() >= 2);

    const int cols = sizes[sizes.size() - 1];
    const int rows = x.numel() / cols;

    PD_CHECK(hidden_size == cols);
    PD_CHECK(epsilon >= 0.f);

    auto place = x.place();
    dev_ctx.template Alloc<T>(y);
    dev_ctx.template Alloc<float>(invvar);
    funcs::fast_ln_v2::LaunchNormFwd<T, Context>(
        dev_ctx,
        dev_ctx.stream(),
        place,
        /* x_ptr */ x.data(),
        /* scale_ptr */ scale.data(),
        /* bias_ptr */ nullptr,
        /* y_ptr */ y->data(),
        /* mean_ptr */ nullptr,
        /* invvar_ptr */ invvar->data(),
        weight_type,
        input_type,
        output_type,
        compute_type,
        hidden_size,
        rows,
        cols,
        epsilon);
  } else {
    RMSLnFwd<T, Context>(dev_ctx, x, scale, epsilon, y, invvar);
  }
#else
  RMSLnFwd<T, Context>(dev_ctx, x, scale, epsilon, y, invvar);
#endif
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
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The dtype of scale must be FLOAT32, BFLOAT16, but got [%s]",
          scale.dtype()));
    }
    cuda_rms_norm_gradient<T, Context>(dev_ctx,
                                       x,
                                       scale,
                                       invvar,
                                       y_grad,
                                       rows,
                                       cols,
                                       epsilon,
                                       x_grad,
                                       scale_grad);
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
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The dtype of scale must be FLOAT32, BFLOAT16, but got [%s]",
          scale.dtype()));
    }
  }
}

template <typename T, typename Context>
void RMSNormBwdKernel(const Context &dev_ctx,
                      const DenseTensor &x,
                      const DenseTensor &scale,
                      const DenseTensor &invvar,
                      const DenseTensor &y_grad,
                      float epsilon,
                      DenseTensor *x_grad,
                      DenseTensor *scale_grad) {
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
  auto input_type = x.type();
  auto weight_type = scale.type();
  auto output_type = weight_type;
  auto compute_type = paddle::DataType::FLOAT32;
  auto hidden_size = scale.numel();

  // using fast_ln_v2 only sm > 70
  auto prop = funcs::fast_ln_v2::GetDeviceProp();
  bool has_fast_rms_norm = funcs::fast_ln_v2::has_fast_ln_v2_bwd_kernel(
      weight_type, input_type, output_type, compute_type, hidden_size);
  if (prop->major > 7 && has_fast_rms_norm) {
    PD_CHECK(y_grad.dtype() == output_type);

    auto sizes = x.dims();
    PD_CHECK(sizes.size() >= 2);
    PD_CHECK(y_grad.dims() == sizes);

    int64_t rows = 1;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
      rows *= sizes[i];
    }
    auto cols = sizes[sizes.size() - 1];

    PD_CHECK(scale.numel() == cols);
    dev_ctx.template Alloc<T>(x_grad);
    dev_ctx.template Alloc<T>(scale_grad);

    auto place = x.place();

    funcs::fast_ln_v2::LaunchNormBwd<T, Context>(
        dev_ctx,
        dev_ctx.stream(),
        place,
        /* x_ptr */ x.data(),
        /* scale_ptr */ scale.data(),
        /* mean_ptr */ nullptr,
        /* invvar_ptr */ invvar.data(),
        /* y_grad_ptr */ y_grad.data(),
        /* x_grad_ptr */ x_grad->data(),
        /* scale_grad_ptr */ scale_grad->data(),
        /* dbias_ptr */ nullptr,
        weight_type,
        input_type,
        output_type,
        compute_type,
        hidden_size,
        rows,
        cols,
        epsilon);
  } else {
    RMSLnBwd<T, Context>(
        dev_ctx, x, scale, invvar, y_grad, epsilon, x_grad, scale_grad);
  }
#else
  RMSLnBwd<T, Context>(
      dev_ctx, x, scale, invvar, y_grad, epsilon, x_grad, scale_grad);
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(rms_norm_nzs,
                   GPU,
                   ALL_LAYOUT,
                   phi::RMSNormFwdKernel,
                   float,
                   double,
                   phi::bfloat16) {}

PD_REGISTER_KERNEL(rms_norm_nzs_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RMSNormBwdKernel,
                   float,
                   double,
                   phi::bfloat16) {}
