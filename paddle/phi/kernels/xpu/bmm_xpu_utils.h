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
#pragma once
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/matmul_grad_kernel_impl.h"
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
  int fc_calc_type = FCCalcType<XPUType>();
  decltype(&xblas_fc_batch_wrapper<XPUType, int16_t, float>)
      xblas_fc_batch_api_list[6] = {
          &xblas_fc_batch_wrapper<XPUType, int16_t, float>,
          &xblas_fc_batch_wrapper<XPUType, int32_t, float>,
          &xblas_fc_batch_wrapper<XPUType, float, float>,
          &xblas_fc_batch_wrapper<XPUType, int_with_ll_t, float>,
          &xblas_fc_batch_wrapper<XPUType, tfloat32, float>,
          &xblas_fc_batch_wrapper<XPUType, XPUTypeFP16, float>,
      };

  auto xblas_fc_batch_api = xblas_fc_batch_api_list[fc_calc_type];
  if (fc_calc_type == XPUFCCalcType::FC_FLOAT16 &&
      std::getenv("XPU_PADDLE_FC_FLOAT16") != nullptr) {
    xblas_fc_batch_api =
        &xblas_fc_batch_wrapper<XPUType, XPUTypeFP16, XPUTypeFP16>;
  }

  xblas_fc_batch_api(xpu_ctx,
                     batch_size,
                     mat_dim_a.trans_,
                     mat_dim_b.trans_,
                     m,
                     n,
                     k,
                     1.0,
                     reinterpret_cast<const XPUType*>(x.data<T>()),
                     mat_dim_a.stride_,
                     reinterpret_cast<const XPUType*>(y.data<T>()),
                     mat_dim_b.stride_,
                     0.0,
                     reinterpret_cast<XPUType*>(data_c),
                     m * n,
                     nullptr,
                     nullptr);
}

}  // namespace phi
