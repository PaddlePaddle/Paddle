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

#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

template <typename InT, typename OutT, typename Context>
DenseTensor CastXpuMatmulInputImpl(const Context& dev_ctx,
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

// Use the direct XPU cast API for mixed matmul inputs. The generic XPU
// CastKernel path can reject valid float16/bfloat16/float32 conversions before
// matmul with XDNN_INVALID_PARAM on some devices.
template <typename OutT, typename Context>
DenseTensor CastXpuMatmulInput(const Context& dev_ctx,
                               const DenseTensor& input) {
  if (input.dtype() == phi::CppTypeToDataType<OutT>::Type()) {
    return input;
  }

  switch (input.dtype()) {
    case DataType::FLOAT32:
      return CastXpuMatmulInputImpl<float, OutT, Context>(dev_ctx, input);
    case DataType::FLOAT16:
      return CastXpuMatmulInputImpl<dtype::float16, OutT, Context>(dev_ctx,
                                                                   input);
    case DataType::BFLOAT16:
      return CastXpuMatmulInputImpl<dtype::bfloat16, OutT, Context>(dev_ctx,
                                                                    input);
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "XPU matmul only supports float32, float16 and bfloat16 inputs, but "
          "received %s.",
          input.dtype()));
  }
}

template <typename ComputeT, typename Context>
void MatmulKernelImpl(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    // input shape [1, 1, 5, 0], [1, 1, 0, 5], result shape is [1, 1, 5, 5]
    Full<ComputeT, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }
  using XPUType = typename XPUTypeTrait<ComputeT>::Type;

  dev_ctx.template Alloc<ComputeT>(out);
  // XPU dispatch may instantiate this kernel from y's dtype while
  // MatmulInferMeta keeps the output dtype aligned with x. Cast both operands
  // to the actual output compute dtype and never read a tensor through the
  // dispatch template dtype.
  const DenseTensor x_cast =
      x.dtype() == out->dtype()
          ? x
          : CastXpuMatmulInput<ComputeT, Context>(dev_ctx, x);
  const DenseTensor y_cast =
      y.dtype() == out->dtype()
          ? y
          : CastXpuMatmulInput<ComputeT, Context>(dev_ctx, y);
  const XPUType* x_ptr =
      reinterpret_cast<const XPUType*>(x_cast.data<ComputeT>());
  const XPUType* y_ptr =
      reinterpret_cast<const XPUType*>(y_cast.data<ComputeT>());
  XPUType* out_ptr = reinterpret_cast<XPUType*>(out->data<ComputeT>());
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  XpuFcInfo fc_info;
  GetFCInfo(x_dims, y_dims, transpose_x, transpose_y, &fc_info);
  xpu::Context* xpu_ctx = dev_ctx.x_context();
  MatMulXPUFunction<XPUType>(xpu_ctx, x_ptr, y_ptr, out_ptr, fc_info, 1.0f);
}

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  DenseTensor* out) {
  switch (out->dtype()) {
    case DataType::FLOAT32:
      return MatmulKernelImpl<float, Context>(
          dev_ctx, x, y, transpose_x, transpose_y, out);
    case DataType::FLOAT16:
      return MatmulKernelImpl<dtype::float16, Context>(
          dev_ctx, x, y, transpose_x, transpose_y, out);
    case DataType::BFLOAT16:
      return MatmulKernelImpl<dtype::bfloat16, Context>(
          dev_ctx, x, y, transpose_x, transpose_y, out);
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "XPU matmul only supports float32, float16 and bfloat16 outputs, but "
          "received %s.",
          out->dtype()));
  }
}

template <typename ComputeT, typename Context>
void MatmulWithFlattenKernelImpl(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 int x_num_col_dims,
                                 int y_num_col_dims,
                                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<ComputeT>::Type;
  const DenseTensor x_matrix =
      x.dims().size() > 2 ? ReshapeToMatrix(x, x_num_col_dims) : x;
  const DenseTensor y_matrix =
      y.dims().size() > 2 ? ReshapeToMatrix(y, y_num_col_dims) : y;
  dev_ctx.template Alloc<ComputeT>(out);

  const DenseTensor x_matrix_cast =
      x_matrix.dtype() == out->dtype()
          ? x_matrix
          : CastXpuMatmulInput<ComputeT, Context>(dev_ctx, x_matrix);
  const DenseTensor y_matrix_cast =
      y_matrix.dtype() == out->dtype()
          ? y_matrix
          : CastXpuMatmulInput<ComputeT, Context>(dev_ctx, y_matrix);
  const XPUType* x_ptr =
      reinterpret_cast<const XPUType*>(x_matrix_cast.data<ComputeT>());
  const XPUType* y_ptr =
      reinterpret_cast<const XPUType*>(y_matrix_cast.data<ComputeT>());
  XPUType* out_ptr = reinterpret_cast<XPUType*>(out->data<ComputeT>());

  bool trans_a = false;
  bool trans_b = false;
  auto x_dims = x_matrix.dims();
  auto y_dims = y_matrix.dims();

  phi::XpuFcInfo fc_info;
  phi::GetFCInfo(x_dims, y_dims, trans_a, trans_b, &fc_info);

  xpu::Context* xpu_ctx = dev_ctx.x_context();

  phi::MatMulXPUFunction<XPUType>(
      xpu_ctx, x_ptr, y_ptr, out_ptr, fc_info, 1.0f);
}

template <typename T, typename Context>
void MatmulWithFlattenKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int x_num_col_dims,
                             int y_num_col_dims,
                             DenseTensor* out) {
  switch (out->dtype()) {
    case DataType::FLOAT32:
      return MatmulWithFlattenKernelImpl<float, Context>(
          dev_ctx, x, y, x_num_col_dims, y_num_col_dims, out);
    case DataType::FLOAT16:
      return MatmulWithFlattenKernelImpl<dtype::float16, Context>(
          dev_ctx, x, y, x_num_col_dims, y_num_col_dims, out);
    case DataType::BFLOAT16:
      return MatmulWithFlattenKernelImpl<dtype::bfloat16, Context>(
          dev_ctx, x, y, x_num_col_dims, y_num_col_dims, out);
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "XPU matmul_with_flatten only supports float32, float16 and bfloat16 "
          "outputs, but received %s.",
          out->dtype()));
  }
}

template <typename T, typename Context>
void LegacyMatmulKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        bool transpose_x,
                        bool transpose_y,
                        float alpha UNUSED,
                        DenseTensor* out) {
  MatmulKernel<T, Context>(dev_ctx, x, y, transpose_x, transpose_y, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(matmul,
                   XPU,
                   ALL_LAYOUT,
                   phi::MatmulKernel,
                   float,
                   phi::bfloat16,
                   phi::float16) {}

PD_REGISTER_KERNEL(matmul_with_flatten,
                   XPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenKernel,
                   float,
                   phi::bfloat16,
                   phi::float16) {}

PD_REGISTER_KERNEL(legacy_matmul,
                   XPU,
                   ALL_LAYOUT,
                   phi::LegacyMatmulKernel,
                   float,
                   phi::bfloat16,
                   phi::float16) {}
