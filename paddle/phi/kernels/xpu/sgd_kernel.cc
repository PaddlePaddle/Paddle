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

#include "paddle/phi/kernels/sgd_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SGDDenseKernel(const Context& dev_ctx,
                    const DenseTensor& param,
                    const DenseTensor& learning_rate,
                    const DenseTensor& grad,
                    const paddle::optional<DenseTensor>& master_param,
                    bool multi_precision,
                    DenseTensor* param_out,
                    DenseTensor* master_param_out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto sz = param_out->numel();
  PADDLE_ENFORCE_EQ(
      param.numel(),
      sz,
      errors::InvalidArgument("The input tensor Param's numel of SgdOp "
                              "should be equal with ParamOut's numel. "
                              "But received Param's "
                              "numel = [%s], ParamOut's numel = [%s]",
                              param.numel(),
                              sz));
  PADDLE_ENFORCE_EQ(
      grad.numel(),
      sz,
      errors::InvalidArgument("The input tensor Grad's numel of SgdOp "
                              "should be equal with ParamOut's numel. "
                              "But received Grad's "
                              "numel = [%s], ParamOut's numel = [%s]",
                              grad.numel(),
                              sz));

  const T* lr_t = learning_rate.data<T>();
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  const float* lr = nullptr;
  if (std::is_same<T, dtype::float16>::value) {
    float* lr_float = RAII_GUARD.alloc_l3_or_gm<float>(learning_rate.numel());
    int r = xpu::cast<XPUType, float>(dev_ctx.x_context(),
                                      reinterpret_cast<const XPUType*>(lr_t),
                                      lr_float,
                                      learning_rate.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    lr = lr_float;
  } else {
    lr = reinterpret_cast<const float*>(lr_t);
  }

  const T* param_data = param.data<T>();
  const T* grad_data = grad.data<T>();

  dev_ctx.template Alloc<T>(param_out);
  T* out_data = param_out->data<T>();

  int r = xpu::sgd(dev_ctx.x_context(),
                   reinterpret_cast<const XPUType*>(grad_data),
                   reinterpret_cast<const XPUType*>(param_data),
                   lr,
                   reinterpret_cast<XPUType*>(out_data),
                   sz);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sgd");
}

template <typename T, typename Context>
void SGDDenseParamSparseGradKernel(
    const Context& dev_ctx,
    const DenseTensor& param,
    const DenseTensor& learning_rate,
    const SelectedRows& grad,
    const paddle::optional<DenseTensor>& master_param,
    bool multi_precision,
    DenseTensor* param_out,
    DenseTensor* master_param_out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(param_out);

  PADDLE_ENFORCE_EQ(
      param.IsSharedBufferWith(*param_out),
      true,
      common::errors::InvalidArgument(
          "The input tensor Param of SgdOp should be equal with ParamOut "
          "if variable's type is SelectedRows."));

  auto in_height = grad.height();
  auto out_dims = param_out->dims();
  PADDLE_ENFORCE_EQ(in_height,
                    out_dims[0],
                    common::errors::InvalidArgument(
                        "The input tensor Grad's height of SgdOp should be "
                        "equal with ParamOut's dims. But received Grad's "
                        "height [%s] and ParamOut's dims [%s]",
                        in_height,
                        out_dims[0]));

  auto& in_value = grad.value();
  auto& in_rows = grad.rows();
  int64_t* in_rows_data = nullptr;
  xpu::VectorParam<int64_t> in_rows_vec{
      in_rows.data(), static_cast<int>(in_rows.size()), in_rows_data};

  int64_t in_row_numel = in_value.numel() / in_rows.size();
  PADDLE_ENFORCE_EQ(in_row_numel,
                    param_out->numel() / in_height,
                    common::errors::InvalidArgument(
                        "The in_row_numel of SgdOp should be equal with "
                        "param_out's numel / in_height."));

  auto* in_data = in_value.data<T>();
  auto* out_data = param_out->data<T>();

  int r = xpu::sparse_sgd<XPUType, int64_t>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(in_data),
      reinterpret_cast<const XPUType*>(param.data<T>()),
      learning_rate.data<float>(),
      in_rows_vec,
      reinterpret_cast<XPUType*>(out_data),
      in_row_numel,
      in_rows.size());

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sparse_sgd");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    sgd, XPU, ALL_LAYOUT, phi::SGDDenseKernel, phi::dtype::float16, float) {}
PD_REGISTER_KERNEL(sgd_dense_param_sparse_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SGDDenseParamSparseGradKernel,
                   phi::dtype::float16,
                   float) {}
