// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/matmul_grad_kernel_impl.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
template <typename T>
static void MGroupedGemmXPUFunction(const DenseTensor &a,
                                    const DenseTensor &b,
                                    DenseTensor *out,
                                    const bool trans_rhs,
                                    const std::vector<int64_t> &batch_sizes,
                                    xpu::Context *xpu_ctx) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto &a_shape = a.dims();
  const auto &b_shape = b.dims();
  const int64_t num_experts = batch_sizes.size();

  int fc_calc_type = FCCalcType<XPUType>();
  decltype(&xblas_fc_wrapper<XPUType, int16_t>) xblas_fc_api_list[6] = {
      &xblas_fc_wrapper<XPUType, int16_t>,
      &xblas_fc_wrapper<XPUType, int32_t>,
      &xblas_fc_wrapper<XPUType, float>,
      &xblas_fc_wrapper<XPUType, int_with_ll_t>,
      &xblas_fc_wrapper<XPUType, tfloat32>,
      &xblas_fc_wrapper<XPUType, XPUTypeFP16>,
  };
  auto xblas_fc_api = xblas_fc_api_list[fc_calc_type];

  int64_t n = trans_rhs ? b_shape[1] : b_shape[2];
  int64_t k = a_shape[1];
  int64_t ldx = k;
  int64_t ldy = trans_rhs ? k : n;
  int64_t ldout = n;
  bool trans_x = false;
  bool trans_y = trans_rhs;
  float *max_x = nullptr;
  float *max_y = nullptr;
  float *max_out = nullptr;
  const float *bias = nullptr;
  const float *scale_x = nullptr;
  const float *scale_y = nullptr;
  int scale_x_mode = 0;
  int scale_y_mode = 0;
  xpu::Activation_t act = xpu::Activation_t::LINEAR;

  if constexpr (std::is_same<T, paddle::bfloat16>::value ||
                std::is_same<T, float>::value) {
    T *a_data = const_cast<T *>(a.data<T>());  // alias for a.data
    T *b_data = const_cast<T *>(b.data<T>());  // alias for b.data
    T *output_data = out->data<T>();

#pragma unroll
    for (int64_t i = 0; i < num_experts; ++i) {
      const int64_t expert_bs = batch_sizes[i];
      int64_t m = expert_bs;
      if (expert_bs != 0) {
        xblas_fc_api(xpu_ctx,
                     reinterpret_cast<const XPUType *>(a_data),
                     reinterpret_cast<const XPUType *>(b_data),
                     reinterpret_cast<XPUType *>(output_data),
                     m,
                     n,
                     k,
                     trans_x,
                     trans_y,
                     max_x,
                     max_y,
                     max_out,
                     ldx,
                     ldy,
                     ldout,
                     1.0f,
                     0.0f,
                     bias,
                     act,
                     scale_x,
                     scale_y,
                     scale_x_mode,
                     scale_y_mode);
      }
      a_data += expert_bs * a_shape[1];
      b_data += b_shape[1] * b_shape[2];
      output_data += expert_bs * n;
    }

  } else {
    PD_CHECK(false, "Unsupported data type, only support bfloat16 and float32");
  }
}

template <typename T>
static void KGroupedGemmXPUFunction(const DenseTensor &a,
                                    const DenseTensor &b,
                                    DenseTensor *out,
                                    const std::vector<int64_t> &batch_sizes,
                                    xpu::Context *xpu_ctx) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto &a_shape = a.dims();
  const auto &b_shape = b.dims();

  const int64_t num_experts = batch_sizes.size();
  const int64_t input_hidden_size = a_shape[1];
  const int64_t output_hidden_size = b_shape[1];

  int fc_calc_type = FCCalcType<XPUType>();
  decltype(&xblas_fc_wrapper<XPUType, int16_t>) xblas_fc_api_list[6] = {
      &xblas_fc_wrapper<XPUType, int16_t>,
      &xblas_fc_wrapper<XPUType, int32_t>,
      &xblas_fc_wrapper<XPUType, float>,
      &xblas_fc_wrapper<XPUType, int_with_ll_t>,
      &xblas_fc_wrapper<XPUType, tfloat32>,
      &xblas_fc_wrapper<XPUType, XPUTypeFP16>,
  };
  auto xblas_fc_api = xblas_fc_api_list[fc_calc_type];

  int64_t m = input_hidden_size;
  int64_t n = output_hidden_size;
  int64_t ldy = n;
  int64_t ldout = n;
  int64_t ldx = m;
  bool trans_x = true;
  bool trans_y = false;
  float *max_x = nullptr;
  float *max_y = nullptr;
  float *max_out = nullptr;
  const float *bias = nullptr;
  const float *scale_x = nullptr;
  const float *scale_y = nullptr;
  int scale_x_mode = 0;
  int scale_y_mode = 0;
  xpu::Activation_t act = xpu::Activation_t::LINEAR;

  if constexpr (std::is_same<T, paddle::bfloat16>::value ||
                std::is_same<T, float>::value) {
    T *a_data = const_cast<T *>(a.data<T>());  // alias for a.data
    T *b_data = const_cast<T *>(b.data<T>());  // alias for b.data
    T *output_data = out->data<T>();

#pragma unroll
    for (int64_t i = 0; i < num_experts; ++i) {
      const int64_t expert_bs = batch_sizes[i];
      int64_t k = expert_bs;
      if (expert_bs != 0) {
        xblas_fc_api(xpu_ctx,
                     reinterpret_cast<const XPUType *>(a_data),
                     reinterpret_cast<const XPUType *>(b_data),
                     reinterpret_cast<XPUType *>(output_data),
                     m,
                     n,
                     k,
                     trans_x,
                     trans_y,
                     max_x,
                     max_y,
                     max_out,
                     ldx,
                     ldy,
                     ldout,
                     1.0f,
                     0.0f,
                     bias,
                     act,
                     scale_x,
                     scale_y,
                     scale_x_mode,
                     scale_y_mode);
      }
      a_data += expert_bs * input_hidden_size;
      b_data += expert_bs * output_hidden_size;
      output_data += input_hidden_size * output_hidden_size;
    }
  } else {
    PD_CHECK(false, "Unsupported data type, only support bfloat16 and float32");
  }
}

}  // namespace phi
