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

#include "paddle/phi/kernels/matmul_grad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"
namespace phi {

template <typename T, typename Context>
void MatmulGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& dout,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor* dx,
                      DenseTensor* dy) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
  }

  const XPUType* dout_ptr = reinterpret_cast<const XPUType*>(dout.data<T>());
  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y.data<T>());

  xpu::Context* xpu_ctx = dev_ctx.x_context();

  XpuFcInfo info_forward;
  GetFCInfo(x.dims(), y.dims(), transpose_x, transpose_y, &info_forward);
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  // begin calculate
  const XPUType* a_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* a_2 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_2 = reinterpret_cast<const XPUType*>(NULL);
  XPUType* c_1 = (dx == NULL) ? reinterpret_cast<XPUType*>(NULL)
                              : reinterpret_cast<XPUType*>(dx->data<T>());
  XPUType* c_2 = (dy == NULL) ? reinterpret_cast<XPUType*>(NULL)
                              : reinterpret_cast<XPUType*>(dy->data<T>());

  if (info_forward.is_x_need_broadcast) {
    XPUType* new_c_1 = nullptr;
    new_c_1 = RAII_GUARD.alloc_l3_or_gm<XPUType>(
        info_forward.bs * info_forward.m * info_forward.k);
    PADDLE_ENFORCE_XDNN_NOT_NULL(new_c_1);
    c_1 = new_c_1;
  }

  if (info_forward.is_y_need_broadcast) {
    XPUType* new_c_2 = RAII_GUARD.alloc_l3_or_gm<XPUType>(
        info_forward.bs * info_forward.k * info_forward.n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(new_c_2);
    c_2 = new_c_2;
  }

  XpuFcInfo info_dx;
  XpuFcInfo info_dy;
  std::tuple<XpuFcInfo,
             XpuFcInfo,
             const XPUType*,
             const XPUType*,
             const XPUType*,
             const XPUType*>
      fc_info = MatmulGradFcInfo(xpu_ctx,
                                 &RAII_GUARD,
                                 info_forward,
                                 transpose_x,
                                 transpose_y,
                                 x_ptr,
                                 y_ptr,
                                 dout_ptr);
  std::tie(info_dx, info_dy, a_1, b_1, a_2, b_2) = fc_info;
  if (dx) {
    MatMulXPUFunction<XPUType>(xpu_ctx, a_1, b_1, c_1, info_dx, 1.0f);
    if (info_forward.is_x_need_broadcast) {
      int r = xpu::reduce_sum<XPUType>(
          xpu_ctx,
          c_1,
          reinterpret_cast<XPUType*>(dx->data<T>()),
          {info_forward.bs, info_forward.m, info_forward.k},
          {0});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
    }
  }
  if (dy) {
    MatMulXPUFunction<XPUType>(xpu_ctx, a_2, b_2, c_2, info_dy, 1.0f);
    if (info_forward.is_y_need_broadcast) {
      int r = xpu::reduce_sum<XPUType>(
          xpu_ctx,
          c_2,
          reinterpret_cast<XPUType*>(dy->data<T>()),
          {info_forward.bs, info_forward.k, info_forward.n},
          {0});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
    }
  }
}

template <typename T, typename Context>
void MatmulWithFlattenGradKernel(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 const DenseTensor& out_grad,
                                 int x_num_col_dims,
                                 int y_num_col_dims,
                                 DenseTensor* x_grad,
                                 DenseTensor* y_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto x_matrix = x.dims().size() > 2 ? phi::ReshapeToMatrix(x, x_num_col_dims)
                                      : static_cast<const DenseTensor&>(x);
  auto y_matrix = y.dims().size() > 2 ? phi::ReshapeToMatrix(y, y_num_col_dims)
                                      : static_cast<const DenseTensor&>(y);
  DenseTensor dout_mat;
  dout_mat.Resize({common::flatten_to_2d(x.dims(), x_num_col_dims)[0],
                   common::flatten_to_2d(y.dims(), y_num_col_dims)[1]});

  if (x_grad != nullptr) {
    x_grad->set_lod(x.lod());
  }
  if (y_grad != nullptr) {
    y_grad->set_lod(y.lod());
  }

  phi::XpuFcInfo info_forward;
  phi::GetFCInfo(x_matrix.dims(), y_matrix.dims(), false, false, &info_forward);

  const XPUType* dout_ptr =
      reinterpret_cast<const XPUType*>(out_grad.data<T>());
  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y.data<T>());

  xpu::Context* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  // begin calculate
  const XPUType* a_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* a_2 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_2 = reinterpret_cast<const XPUType*>(NULL);
  XPUType* c_1 =
      (x_grad == NULL)
          ? reinterpret_cast<XPUType*>(NULL)
          : reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(x_grad));
  XPUType* c_2 =
      (y_grad == NULL)
          ? reinterpret_cast<XPUType*>(NULL)
          : reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(y_grad));
  phi::XpuFcInfo info_dx;
  phi::XpuFcInfo info_dy;
  std::tuple<phi::XpuFcInfo,
             phi::XpuFcInfo,
             const XPUType*,
             const XPUType*,
             const XPUType*,
             const XPUType*>
      fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                      &RAII_GUARD,
                                      info_forward,
                                      false,
                                      false,
                                      x_ptr,
                                      y_ptr,
                                      dout_ptr);
  std::tie(info_dx, info_dy, a_1, b_1, a_2, b_2) = fc_info;
  if (x_grad) {
    phi::MatMulXPUFunction<XPUType>(xpu_ctx, a_1, b_1, c_1, info_dx, 1.0f);
  }
  if (y_grad) {
    phi::MatMulXPUFunction<XPUType>(xpu_ctx, a_2, b_2, c_2, info_dy, 1.0f);
  }
}

template <typename T, typename Context>
void LegacyMatmulGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            const DenseTensor& dout,
                            bool transpose_x,
                            bool transpose_y,
                            float alpha UNUSED,
                            DenseTensor* dx,
                            DenseTensor* dy) {
  MatmulGradKernel<T, Context>(
      dev_ctx, x, y, dout, transpose_x, transpose_y, dx, dy);
}
}  // namespace phi

PD_REGISTER_KERNEL(matmul_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MatmulGradKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(matmul_with_flatten_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenGradKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(legacy_matmul_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::LegacyMatmulGradKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
