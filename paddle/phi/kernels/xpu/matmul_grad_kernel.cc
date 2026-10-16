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
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"
namespace phi {

template <typename InT, typename OutT, typename Context>
DenseTensor CastXpuMatmulGradInputImpl(const Context& dev_ctx,
                                       const DenseTensor& input) {
  using XPUInT = typename XPUTypeTrait<InT>::Type;
  using XPUOutT = typename XPUTypeTrait<OutT>::Type;
  DenseTensor casted_input;
  casted_input.Resize(input.dims());
  dev_ctx.template Alloc<OutT>(&casted_input);
  int r = baidu::xpu::api::cast<XPUInT, XPUOutT>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUInT*>(input.data<InT>()),
      reinterpret_cast<XPUOutT*>(casted_input.data<OutT>()),
      input.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  return casted_input;
}

// Use direct XPU casts for mixed matmul grad inputs. This keeps all internal
// matmul operands in dout's compute dtype without relying on generic CastKernel
// for float16/bfloat16/float32 input alignment.
template <typename OutT, typename Context>
DenseTensor CastXpuMatmulGradInput(const Context& dev_ctx,
                                   const DenseTensor& input) {
  if (input.dtype() == phi::CppTypeToDataType<OutT>::Type()) {
    return input;
  }

  switch (input.dtype()) {
    case DataType::FLOAT32:
      return CastXpuMatmulGradInputImpl<float, OutT, Context>(dev_ctx, input);
    case DataType::FLOAT16:
      return CastXpuMatmulGradInputImpl<dtype::float16, OutT, Context>(dev_ctx,
                                                                       input);
    case DataType::BFLOAT16:
      return CastXpuMatmulGradInputImpl<dtype::bfloat16, OutT, Context>(dev_ctx,
                                                                        input);
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "XPU matmul_grad only supports float32, float16 and bfloat16 "
          "inputs, but received %s.",
          input.dtype()));
  }
}

template <typename InT, typename OutT, typename Context>
void CastXpuMatmulGradOutput(const Context& dev_ctx,
                             const DenseTensor& input,
                             DenseTensor* output) {
  using XPUInT = typename XPUTypeTrait<InT>::Type;
  using XPUOutT = typename XPUTypeTrait<OutT>::Type;
  dev_ctx.template Alloc<OutT>(output);
  int r = baidu::xpu::api::cast<XPUInT, XPUOutT>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUInT*>(input.data<InT>()),
      reinterpret_cast<XPUOutT*>(output->data<OutT>()),
      input.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
}

template <typename InT, typename Context>
void CastXpuMatmulGradOutputByDtype(const Context& dev_ctx,
                                    const DenseTensor& input,
                                    DenseTensor* output) {
  switch (output->dtype()) {
    case DataType::FLOAT32:
      CastXpuMatmulGradOutput<InT, float, Context>(dev_ctx, input, output);
      break;
    case DataType::FLOAT16:
      CastXpuMatmulGradOutput<InT, dtype::float16, Context>(
          dev_ctx, input, output);
      break;
    case DataType::BFLOAT16:
      CastXpuMatmulGradOutput<InT, dtype::bfloat16, Context>(
          dev_ctx, input, output);
      break;
    default:
      PADDLE_THROW(
          common::errors::Unavailable("XPU matmul_grad only supports float32, "
                                      "float16 and bfloat16 gradient "
                                      "outputs, but received %s.",
                                      output->dtype()));
  }
}

template <typename Context>
void AllocXpuMatmulGradOutputByDtype(const Context& dev_ctx,
                                     DenseTensor* output) {
  switch (output->dtype()) {
    case DataType::FLOAT32:
      dev_ctx.template Alloc<float>(output);
      break;
    case DataType::FLOAT16:
      dev_ctx.template Alloc<dtype::float16>(output);
      break;
    case DataType::BFLOAT16:
      dev_ctx.template Alloc<dtype::bfloat16>(output);
      break;
    default:
      PADDLE_THROW(
          common::errors::Unavailable("XPU matmul_grad only supports float32, "
                                      "float16 and bfloat16 gradient "
                                      "outputs, but received %s.",
                                      output->dtype()));
  }
}

template <typename Context>
void FullXpuMatmulGradOutputByDtype(const Context& dev_ctx,
                                    const DDim& dims,
                                    DenseTensor* output) {
  switch (output->dtype()) {
    case DataType::FLOAT32:
      Full<float, Context>(dev_ctx, dims, 0, output);
      break;
    case DataType::FLOAT16:
      Full<dtype::float16, Context>(dev_ctx, dims, 0, output);
      break;
    case DataType::BFLOAT16:
      Full<dtype::bfloat16, Context>(dev_ctx, dims, 0, output);
      break;
    default:
      PADDLE_THROW(
          common::errors::Unavailable("XPU matmul_grad only supports float32, "
                                      "float16 and bfloat16 gradient "
                                      "outputs, but received %s.",
                                      output->dtype()));
  }
}

template <typename ComputeT, typename Context>
void MatmulGradKernelImpl(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          const DenseTensor& dout,
                          bool transpose_x,
                          bool transpose_y,
                          DenseTensor* dx,
                          DenseTensor* dy) {
  using XPUType = typename XPUTypeTrait<ComputeT>::Type;
  if (x.numel() == 0) {
    if (dx) {
      AllocXpuMatmulGradOutputByDtype(dev_ctx, dx);
    }
    if (dy) {
      FullXpuMatmulGradOutputByDtype(dev_ctx, y.dims(), dy);
    }
    return;
  }
  if (y.numel() == 0) {
    if (dy) {
      AllocXpuMatmulGradOutputByDtype(dev_ctx, dy);
    }
    if (dx) {
      FullXpuMatmulGradOutputByDtype(dev_ctx, x.dims(), dx);
    }
    return;
  }

  if (!transpose_x && transpose_y && y.dims().size() < 2) {
    transpose_y = false;
  }
  // Compute follows dout's dtype; cast operands directly and avoid accessing
  // tensors through the dispatch template dtype when mixed inputs are present.
  const DenseTensor x_cast =
      x.dtype() == dout.dtype()
          ? x
          : CastXpuMatmulGradInput<ComputeT, Context>(dev_ctx, x);
  const DenseTensor y_cast =
      y.dtype() == dout.dtype()
          ? y
          : CastXpuMatmulGradInput<ComputeT, Context>(dev_ctx, y);
  const XPUType* dout_ptr =
      reinterpret_cast<const XPUType*>(dout.data<ComputeT>());
  const XPUType* x_ptr =
      reinterpret_cast<const XPUType*>(x_cast.data<ComputeT>());
  const XPUType* y_ptr =
      reinterpret_cast<const XPUType*>(y_cast.data<ComputeT>());

  xpu::Context* xpu_ctx = dev_ctx.x_context();

  XpuFcInfo info_forward;
  GetFCInfo(x.dims(), y.dims(), transpose_x, transpose_y, &info_forward);
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  // begin calculate
  const XPUType* a_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* a_2 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_2 = reinterpret_cast<const XPUType*>(NULL);
  DenseTensor dx_tmp;
  XPUType* c_1 = reinterpret_cast<XPUType*>(NULL);
  if (dx) {
    if (dx->dtype() == dout.dtype()) {
      c_1 = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<ComputeT>(dx));
    } else {
      dx_tmp.Resize(dx->dims());
      c_1 =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<ComputeT>(&dx_tmp));
    }
  }
  DenseTensor dy_tmp;
  XPUType* c_2 = reinterpret_cast<XPUType*>(NULL);
  if (dy) {
    if (dy->dtype() == dout.dtype()) {
      c_2 = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<ComputeT>(dy));
    } else {
      dy_tmp.Resize(dy->dims());
      c_2 =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<ComputeT>(&dy_tmp));
    }
  }

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
      XPUType* dx_data =
          dx->dtype() == dout.dtype()
              ? reinterpret_cast<XPUType*>(dx->data<ComputeT>())
              : reinterpret_cast<XPUType*>(dx_tmp.data<ComputeT>());
      int r = xpu::reduce_sum<XPUType>(xpu_ctx,
                                       c_1,
                                       dx_data,
                                       {(int64_t)info_forward.bs,
                                        (int64_t)info_forward.m,
                                        (int64_t)info_forward.k},
                                       {0LL});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
    }
    if (dx->dtype() != dout.dtype()) {
      CastXpuMatmulGradOutputByDtype<ComputeT, Context>(dev_ctx, dx_tmp, dx);
    }
  }
  if (dy) {
    MatMulXPUFunction<XPUType>(xpu_ctx, a_2, b_2, c_2, info_dy, 1.0f);
    if (info_forward.is_y_need_broadcast) {
      XPUType* dy_data =
          dy->dtype() == dout.dtype()
              ? reinterpret_cast<XPUType*>(dy->data<ComputeT>())
              : reinterpret_cast<XPUType*>(dy_tmp.data<ComputeT>());
      int r = xpu::reduce_sum<XPUType>(xpu_ctx,
                                       c_2,
                                       dy_data,
                                       {(int64_t)info_forward.bs,
                                        (int64_t)info_forward.k,
                                        (int64_t)info_forward.n},
                                       {0LL});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
    }
    if (dy->dtype() != dout.dtype()) {
      CastXpuMatmulGradOutputByDtype<ComputeT, Context>(dev_ctx, dy_tmp, dy);
    }
  }
}

template <typename T, typename Context>
void MatmulGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& dout,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor* dx,
                      DenseTensor* dy) {
  switch (dout.dtype()) {
    case DataType::FLOAT32:
      return MatmulGradKernelImpl<float, Context>(
          dev_ctx, x, y, dout, transpose_x, transpose_y, dx, dy);
    case DataType::FLOAT16:
      return MatmulGradKernelImpl<dtype::float16, Context>(
          dev_ctx, x, y, dout, transpose_x, transpose_y, dx, dy);
    case DataType::BFLOAT16:
      return MatmulGradKernelImpl<dtype::bfloat16, Context>(
          dev_ctx, x, y, dout, transpose_x, transpose_y, dx, dy);
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "XPU matmul_grad only supports float32, float16 and bfloat16 dout, "
          "but received %s.",
          dout.dtype()));
  }
}

template <typename ComputeT, typename Context>
void MatmulWithFlattenGradKernelImpl(const Context& dev_ctx,
                                     const DenseTensor& x,
                                     const DenseTensor& y,
                                     const DenseTensor& out_grad,
                                     int x_num_col_dims,
                                     int y_num_col_dims,
                                     DenseTensor* x_grad,
                                     DenseTensor* y_grad) {
  using XPUType = typename XPUTypeTrait<ComputeT>::Type;

  auto x_matrix = x.dims().size() > 2 ? ReshapeToMatrix(x, x_num_col_dims)
                                      : static_cast<const DenseTensor&>(x);
  auto y_matrix = y.dims().size() > 2 ? ReshapeToMatrix(y, y_num_col_dims)
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

  const DenseTensor x_cast =
      out_grad.dtype() == x_matrix.dtype()
          ? x_matrix
          : CastXpuMatmulGradInput<ComputeT, Context>(dev_ctx, x_matrix);
  const DenseTensor y_cast =
      out_grad.dtype() == y_matrix.dtype()
          ? y_matrix
          : CastXpuMatmulGradInput<ComputeT, Context>(dev_ctx, y_matrix);
  const XPUType* dout_ptr =
      reinterpret_cast<const XPUType*>(out_grad.data<ComputeT>());
  const XPUType* x_ptr =
      reinterpret_cast<const XPUType*>(x_cast.data<ComputeT>());
  const XPUType* y_ptr =
      reinterpret_cast<const XPUType*>(y_cast.data<ComputeT>());

  xpu::Context* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  // begin calculate
  const XPUType* a_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* a_2 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_2 = reinterpret_cast<const XPUType*>(NULL);
  DenseTensor x_grad_tmp;
  XPUType* c_1 = reinterpret_cast<XPUType*>(NULL);
  if (x_grad) {
    if (x_grad->dtype() == out_grad.dtype()) {
      c_1 =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<ComputeT>(x_grad));
    } else {
      x_grad_tmp.Resize(x_grad->dims());
      c_1 = reinterpret_cast<XPUType*>(
          dev_ctx.template Alloc<ComputeT>(&x_grad_tmp));
    }
  }
  DenseTensor y_grad_tmp;
  XPUType* c_2 = reinterpret_cast<XPUType*>(NULL);
  if (y_grad) {
    if (y_grad->dtype() == out_grad.dtype()) {
      c_2 =
          reinterpret_cast<XPUType*>(dev_ctx.template Alloc<ComputeT>(y_grad));
    } else {
      y_grad_tmp.Resize(y_grad->dims());
      c_2 = reinterpret_cast<XPUType*>(
          dev_ctx.template Alloc<ComputeT>(&y_grad_tmp));
    }
  }
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
    if (x_grad->dtype() != out_grad.dtype()) {
      CastXpuMatmulGradOutputByDtype<ComputeT, Context>(
          dev_ctx, x_grad_tmp, x_grad);
    }
  }
  if (y_grad) {
    phi::MatMulXPUFunction<XPUType>(xpu_ctx, a_2, b_2, c_2, info_dy, 1.0f);
    if (y_grad->dtype() != out_grad.dtype()) {
      CastXpuMatmulGradOutputByDtype<ComputeT, Context>(
          dev_ctx, y_grad_tmp, y_grad);
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
  switch (out_grad.dtype()) {
    case DataType::FLOAT32:
      return MatmulWithFlattenGradKernelImpl<float, Context>(dev_ctx,
                                                             x,
                                                             y,
                                                             out_grad,
                                                             x_num_col_dims,
                                                             y_num_col_dims,
                                                             x_grad,
                                                             y_grad);
    case DataType::FLOAT16:
      return MatmulWithFlattenGradKernelImpl<dtype::float16, Context>(
          dev_ctx,
          x,
          y,
          out_grad,
          x_num_col_dims,
          y_num_col_dims,
          x_grad,
          y_grad);
    case DataType::BFLOAT16:
      return MatmulWithFlattenGradKernelImpl<dtype::bfloat16, Context>(
          dev_ctx,
          x,
          y,
          out_grad,
          x_num_col_dims,
          y_num_col_dims,
          x_grad,
          y_grad);
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "XPU matmul_with_flatten_grad only supports float32, float16 and "
          "bfloat16 out_grad, but received %s.",
          out_grad.dtype()));
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
                   phi::bfloat16,
                   phi::float16) {}

PD_REGISTER_KERNEL(matmul_with_flatten_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenGradKernel,
                   float,
                   phi::bfloat16,
                   phi::float16) {}

PD_REGISTER_KERNEL(legacy_matmul_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::LegacyMatmulGradKernel,
                   float,
                   phi::bfloat16,
                   phi::float16) {}
