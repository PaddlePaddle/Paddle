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

#include "paddle/phi/kernels/bmm_grad_kernel.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/matmul_grad_kernel_impl.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"
namespace phi {

template <typename T, typename FCT>
static void MatMulXPUFunction(const DenseTensor& x,
                              const DenseTensor& y,
                              DenseTensor* out,
                              bool trans_x,
                              bool trans_y,
                              xpu::Context* xpu_ctx) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto& x_dims = x.dims();
  const auto& y_dims = y.dims();

  auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(
      RowMatrixFromVector(x_dims), 0, trans_x);
  auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(
      ColumnMatrixFromVector(y_dims), 0, trans_y);

  T* data_c = out->data<T>();
  int m = mat_dim_a.height_;
  int n = mat_dim_b.width_;
  int k = mat_dim_a.width_;
  int batch_size = mat_dim_a.batch_size_;
  // batch matmul
  int r = xpu::fc_batched<XPUType, XPUType, XPUType, FCT>(
      xpu_ctx,                                        // Context* ctx,
      batch_size,                                     // int batch_size,
      mat_dim_a.trans_,                               // bool x_trans,
      mat_dim_b.trans_,                               // bool w_trans,
      m,                                              // int m,
      n,                                              // int n,
      k,                                              // int k,
      1.0,                                            // float alpha,
      reinterpret_cast<const XPUType*>(x.data<T>()),  // const TX* x,
      mat_dim_a.stride_,                              // int stride_a,
      reinterpret_cast<const XPUType*>(y.data<T>()),  // const TW* w,
      mat_dim_b.stride_,                              // int stride_b,
      0.0,                                            // float beta,
      reinterpret_cast<XPUType*>(data_c),             // TY* y,
      m * n,                                          // int stride_c,
      nullptr,                                        // const float* x_maxptr,
      nullptr);                                       // const float* w_maxptr

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_batched");
}

template <typename T, typename Context>
void MatMul(const Context& dev_ctx,
            const DenseTensor& a,
            bool trans_a,
            const DenseTensor& b,
            bool trans_b,
            DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  xpu::Context* xpu_ctx = dev_ctx.x_context();
  if (std::is_same<paddle::platform::float16, T>::value) {
    MatMulXPUFunction<T, int16_t>(a, b, out, trans_a, trans_b, xpu_ctx);
  } else {
    if (std::getenv("XPU_PADDLE_FC_INT32") != nullptr) {
      MatMulXPUFunction<T, int32_t>(a, b, out, trans_a, trans_b, xpu_ctx);
    } else if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
      MatMulXPUFunction<T, float>(a, b, out, trans_a, trans_b, xpu_ctx);
    } else {
      MatMulXPUFunction<T, int16_t>(a, b, out, trans_a, trans_b, xpu_ctx);
    }
  }
}

template <typename T, typename Context>
void CalcInputGrad(const Context& dev_ctx,
                   const DenseTensor& a,
                   bool trans_a,
                   const DenseTensor& b,
                   bool trans_b,
                   DenseTensor* out) {
  if (out == nullptr) return;
  MatMul<T, Context>(dev_ctx, a, trans_a, b, trans_b, out);
}

template <typename T, typename Context>
void BmmGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   const DenseTensor& out_grad,
                   DenseTensor* x_grad,
                   DenseTensor* y_grad) {
  DenseTensor x_help = x;
  DenseTensor y_help = y;
  DenseTensor out_grad_help = out_grad;
  ReshapeXYOutIntoMatrixSequence(
      &x_help, &y_help, &out_grad_help, false, false);

  phi::DDim dx_dims;
  if (x_grad) {
    dx_dims = x_grad->dims();
    if (dx_dims != x_help.dims()) {
      x_grad->Resize(x_help.dims());
    }
  }

  phi::DDim dy_dims;
  if (y_grad) {
    dy_dims = y_grad->dims();
    if (dy_dims != y_help.dims()) {
      y_grad->Resize(y_help.dims());
    }
  }

  CalcInputGrad<T, Context>(
      dev_ctx, out_grad_help, false, y_help, true, x_grad);
  CalcInputGrad<T, Context>(
      dev_ctx, x_help, true, out_grad_help, false, y_grad);

  if (x_grad) {
    if (dx_dims != x_help.dims()) {
      x_grad->Resize(dx_dims);
    }
  }
  if (y_grad) {
    if (dy_dims != y_help.dims()) {
      y_grad->Resize(dy_dims);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(bmm_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::BmmGradKernel,
                   float,
                   paddle::platform::float16) {}
