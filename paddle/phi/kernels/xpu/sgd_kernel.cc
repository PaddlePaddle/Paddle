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
void SGDDenseKernel(const Context &dev_ctx,
                    const DenseTensor &param,
                    const DenseTensor &learning_rate,
                    const DenseTensor &grad,
                    const paddle::optional<DenseTensor> &master_param,
                    bool multi_precision,
                    DenseTensor *param_out,
                    DenseTensor *master_param_out) {
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

  const T *lr_t = learning_rate.data<T>();
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  const float *lr = nullptr;
  if (std::is_same<T, dtype::float16>::value) {
    float *lr_float = RAII_GUARD.alloc_l3_or_gm<float>(learning_rate.numel());
    int r =
        xpu::cast_v2<XPUType, float>(dev_ctx.x_context(),
                                     reinterpret_cast<const XPUType *>(lr_t),
                                     lr_float,
                                     learning_rate.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip_v2");
    lr = lr_float;
  } else {
    lr = reinterpret_cast<const float *>(lr_t);
  }

  const T *param_data = param.data<T>();
  const T *grad_data = grad.data<T>();

  dev_ctx.template Alloc<T>(param_out);
  T *out_data = param_out->data<T>();

  int r = xpu::sgd(dev_ctx.x_context(),
                   reinterpret_cast<const XPUType *>(grad_data),
                   reinterpret_cast<const XPUType *>(param_data),
                   lr,
                   reinterpret_cast<XPUType *>(out_data),
                   sz);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sgd");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    sgd, XPU, ALL_LAYOUT, phi::SGDDenseKernel, phi::dtype::float16, float) {}
