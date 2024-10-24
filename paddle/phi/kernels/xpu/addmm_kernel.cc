// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/addmm_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "xblas/cublasLt.h"

#ifndef PADDLE_WITH_XPU_XRE5
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"
#endif

namespace xblas = baidu::xpu::xblas;

namespace phi {

template <typename T, typename Context>
void AddmmKernel(const Context& dev_ctx,
                 const DenseTensor& input,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 float beta,
                 float alpha,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto input_dims = input.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(
      input_dims.size() == 2 || input_dims.size() == 1,
      true,
      common::errors::InvalidArgument(
          "Variable 'input' of AddmmOp must be 1-dimensional or 2-dimensional, "
          "but received shape: [%s]",
          input_dims));
  PADDLE_ENFORCE_EQ(x_dims.size() == 2,
                    true,
                    common::errors::InvalidArgument(
                        "Variable 'x' of AddmmOp must be 2-dimensional, "
                        "but received shape: [%s]",
                        input_dims));
  PADDLE_ENFORCE_EQ(y_dims.size() == 2,
                    true,
                    common::errors::InvalidArgument(
                        "Variable 'y' of AddmmOp must be 2-dimensional, "
                        "but received shape: [%s]",
                        input_dims));

  dev_ctx.template Alloc<T>(out);
  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y.data<T>());
  const XPUType* input_ptr = reinterpret_cast<const XPUType*>(input.data<T>());
  XPUType* out_ptr = reinterpret_cast<XPUType*>(out->data<T>());

  int r;
  if (alpha == 0.f) {
    if (beta == 0.f) {
      r = xpu::constant(dev_ctx.x_context(),
                        out_ptr,
                        out->numel(),
                        static_cast<XPUType>(0.0f));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    } else {
      xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
      T* beta_xpu = RAII_GUARD.alloc_l3_or_gm<T>(1);
      r = xpu::constant(dev_ctx.x_context(),
                        reinterpret_cast<XPUType*>(beta_xpu),
                        out->numel(),
                        static_cast<XPUType>(beta));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
      auto input_dims_vec = common::vectorize<int64_t>(input.dims());
      auto out_dims_vec = common::vectorize<int64_t>(out->dims());
      r = xpu::broadcast_mul<XPUType>(dev_ctx.x_context(),
                                      input_ptr,
                                      reinterpret_cast<XPUType*>(beta_xpu),
                                      out_ptr,
                                      input_dims_vec,
                                      out_dims_vec);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");
    }
#ifdef PADDLE_WITH_XPU_XRE5
  } else {
    xblas::FcFusionTensor<const XPUType> t_input{
        input_ptr,
        nullptr,
        input.dims()[0],
        input.dims()[1],
        input.dims()[1],
        false,
    };
    xblas::FcFusionTensor<const XPUType> t_x{
        x_ptr,
        nullptr,
        x.dims()[0],
        x.dims()[1],
        x.dims()[1],
        false,
    };
    xblas::FcFusionTensor<const XPUType> t_y{
        y_ptr,
        nullptr,
        y.dims()[0],
        y.dims()[1],
        y.dims()[1],
        false,
    };
    xblas::FcFusionTensor<XPUType> t_out{
        out_ptr,
        nullptr,
        out->dims()[0],
        out->dims()[1],
        out->dims()[1],
        false,
    };
    xblas::FcFusionDesc<float, float, XPUType> desc{
        alpha,
        beta,
    };
    xblas::FcFusionEpilogue<float, float> epilogue{
        xdnn::Activation_t::LINEAR,
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        nullptr,
    };
    r = xblas::fc_fusion<XPUType,
                         XPUType,
                         XPUType,
                         XPUType,
                         float,
                         float,
                         XPUType,
                         float,
                         float>(
        dev_ctx.x_context(), t_x, t_y, t_input, t_out, desc, epilogue);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_fusion");
#else
  } else {
    Copy(dev_ctx, input, dev_ctx.GetPlace(), false, out);
    XpuFcInfo fc_info;
    GetFCInfo(x_dims, y_dims, false, false, &fc_info);
    MatMulXPUFunction<XPUType>(
        dev_ctx.x_context(), x_ptr, y_ptr, out_ptr, fc_info, alpha, beta);
#endif
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(addmm,
                   XPU,
                   ALL_LAYOUT,
                   phi::AddmmKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
