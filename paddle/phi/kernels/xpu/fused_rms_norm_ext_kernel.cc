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

#include "paddle/common/exception.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

static void GetRowsCols(const std::vector<int64_t> &shape,
                        int64_t *p_rows,
                        int64_t *p_cols) {
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) {
    rows *= shape[i];
  }
  int64_t cols = shape[shape.size() - 1];
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
  int64_t rows, cols;
  GetRowsCols(common::vectorize(x.dims()), &rows, &cols);

  if (scale.dtype() == phi::DataType::BFLOAT16) {
    dev_ctx.template Alloc<phi::bfloat16>(y);
  } else if (scale.dtype() == phi::DataType::FLOAT16) {
    dev_ctx.template Alloc<phi::float16>(y);
  } else if (scale.dtype() == phi::DataType::FLOAT32) {
    dev_ctx.template Alloc<float>(y);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of scale must be FLOAT32, FLOAT16 or BFLOAT16, but got [%s]",
        scale.dtype()));
  }
  invvar->Resize({rows});
  dev_ctx.template Alloc<float>(invvar);

  /*
  refer to:
  -
  https://github.com/NVIDIA/apex/blob/bfb500c8/csrc/layer_norm_cuda_kernel.cu#L1018
  -
  https://github.com/PaddlePaddle/PaddleNLP/blob/5b9e0b33/ops/csrc/fused_ln/layer_norm_cuda.h#L1087

  Supported Type combinations:

  input    compute   scale   output
  =======================================
  fp32     fp32      fp32      fp32
  fp16     fp32      fp16      fp16
  bf16     fp32      bf16      bf16

  Not supported yet:

  input    compute   scale   output
  =======================================
  fp32     fp32      fp16      fp16
  fp32     fp32      bf16      bf16

  Remarks:
  Output type = Scale type
  Compute always in FP32
  */

#define DISPATCH_FWD_CASE(scalar_t_out)                              \
  using XPUType = typename XPUTypeTrait<scalar_t_out>::Type;         \
  auto ret = xpu::rms_layer_norm<XPUType, XPUType>(                  \
      dev_ctx.x_context(),                                           \
      reinterpret_cast<const XPUType *>(x.data<scalar_t_out>()),     \
      reinterpret_cast<XPUType *>(y->data<scalar_t_out>()),          \
      rows,                                                          \
      cols,                                                          \
      epsilon,                                                       \
      reinterpret_cast<const XPUType *>(scale.data<scalar_t_out>()), \
      /*bias=*/nullptr,                                              \
      invvar->data<float>(),                                         \
      /*is_rstd=*/true);                                             \
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "rms_layer_norm");
  // scale.dtype() same as y->dtype()
  if (x.dtype() == phi::DataType::FLOAT32 &&
      scale.dtype() == phi::DataType::FLOAT32) {
    DISPATCH_FWD_CASE(float);
  } else if (x.dtype() == phi::DataType::FLOAT16 &&
             scale.dtype() == phi::DataType::FLOAT16) {
    DISPATCH_FWD_CASE(phi::float16);
  } else if (x.dtype() == phi::DataType::BFLOAT16 &&
             scale.dtype() == phi::DataType::BFLOAT16) {
    DISPATCH_FWD_CASE(phi::bfloat16);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unsupported dtype combination: x [%s], scale [%s]. "
        "Expected both to be float32, float16, or bfloat16.",
        phi::DataTypeToString(x.dtype()),
        phi::DataTypeToString(scale.dtype())));
  }
#undef DISPATCH_FWD_CASE
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
  int64_t rows, cols;
  GetRowsCols(common::vectorize(x.dims()), &rows, &cols);
  dev_ctx.template Alloc<T>(x_grad);
  DenseTensor actual_scale_grad;
  if (scale_grad) {
    if (scale.dtype() == phi::DataType::BFLOAT16) {
      dev_ctx.template Alloc<phi::bfloat16>(scale_grad);
    } else if (scale.dtype() == phi::DataType::FLOAT16) {
      dev_ctx.template Alloc<phi::float16>(scale_grad);
    } else if (scale.dtype() == phi::DataType::FLOAT32) {
      dev_ctx.template Alloc<float>(scale_grad);
    } else {
      PADDLE_THROW(
          common::errors::InvalidArgument("The dtype of scale must be FLOAT32, "
                                          "FLOAT16 or BFLOAT16, but got [%s]",
                                          scale.dtype()));
    }
    actual_scale_grad = *scale_grad;
  } else {
    // lora specific, scale_grad is nullptr
    if (scale.dtype() == phi::DataType::BFLOAT16) {
      actual_scale_grad =
          phi::EmptyLike<phi::bfloat16, Context>(dev_ctx, scale);
    } else if (scale.dtype() == phi::DataType::FLOAT16) {
      actual_scale_grad = phi::EmptyLike<phi::float16, Context>(dev_ctx, scale);
    } else if (scale.dtype() == phi::DataType::FLOAT32) {
      actual_scale_grad = phi::EmptyLike<float, Context>(dev_ctx, scale);
    } else {
      PADDLE_THROW(
          common::errors::InvalidArgument("The dtype of scale must be FLOAT32, "
                                          "FLOAT16 or BFLOAT16, but got [%s]",
                                          scale.dtype()));
    }
  }

#define DISPATCH_BWD_CASE(scalar_t_out)                                    \
  using XPUType = typename XPUTypeTrait<scalar_t_out>::Type;               \
  auto ret = xpu::rms_layer_norm_grad<XPUType, XPUType>(                   \
      dev_ctx.x_context(),                                                 \
      reinterpret_cast<const XPUType *>(x.data<scalar_t_out>()),           \
      reinterpret_cast<const XPUType *>(y_grad.data<scalar_t_out>()),      \
      reinterpret_cast<XPUType *>(x_grad->data<scalar_t_out>()),           \
      rows,                                                                \
      cols,                                                                \
      epsilon,                                                             \
      reinterpret_cast<const XPUType *>(scale.data<scalar_t_out>()),       \
      invvar.data<float>(),                                                \
      reinterpret_cast<XPUType *>(actual_scale_grad.data<scalar_t_out>()), \
      /*bias=*/nullptr,                                                    \
      /*is_rstd=*/true);                                                   \
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "rms_layer_norm_grad");
  // scale.dtype() same as y->dtype()
  if (x.dtype() == phi::DataType::FLOAT32 &&
      scale.dtype() == phi::DataType::FLOAT32) {
    DISPATCH_BWD_CASE(float);
  } else if (x.dtype() == phi::DataType::FLOAT16 &&
             scale.dtype() == phi::DataType::FLOAT16) {
    DISPATCH_BWD_CASE(phi::float16);
  } else if (x.dtype() == phi::DataType::BFLOAT16 &&
             scale.dtype() == phi::DataType::BFLOAT16) {
    DISPATCH_BWD_CASE(phi::bfloat16);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unsupported dtype combination: x [%s], scale [%s]. "
        "Expected both to be float32, float16, or bfloat16.",
        phi::DataTypeToString(x.dtype()),
        phi::DataTypeToString(scale.dtype())));
  }
#undef DISPATCH_BWD_CASE
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_rms_norm_ext,
                   XPU,
                   ALL_LAYOUT,
                   phi::RMSLnFwd,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(fused_rms_norm_ext_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::RMSLnBwd,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
