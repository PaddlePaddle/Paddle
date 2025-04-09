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

#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
namespace fusion {

template <typename T_X,
          typename T_W,
          typename T_QKV,
          typename T_GEMM,
          typename Context>
void CrossAttentionXPUKernelImpl(
    const Context& ctx,
    const DenseTensor& input_q,
    const DenseTensor& input_kv,
    const std::vector<const DenseTensor*>& fc_weight,
    const std::vector<const DenseTensor*>& fc_weight_max,
    const std::vector<const DenseTensor*>& fc_bias,
    const DenseTensor& mask,
    int head_num,
    int head_dim,
    float alpha,
    DataType qkv_dtype,
    DenseTensor* qkv,
    DenseTensor* qkv_max) {
  using XPUTypeX = typename XPUTypeTrait<T_X>::Type;
  using XPUTypeW = typename XPUTypeTrait<T_W>::Type;
  using XPUTypeOut = typename XPUTypeTrait<T_QKV>::Type;
  using XPUTypeGEMM = typename XPUTypeTrait<T_GEMM>::Type;
  auto* input_q_data = reinterpret_cast<const XPUTypeX*>(input_q.data<T_X>());
  auto* input_kv_data = reinterpret_cast<const XPUTypeX*>(input_kv.data<T_X>());

  xpu::ctx_guard RAII_GUARD(ctx.x_context());

  XPUTypeFP16* q_data = RAII_GUARD.alloc_l3_or_gm<XPUTypeFP16>(input_q.numel());
  XPUTypeFP16* k_data =
      RAII_GUARD.alloc_l3_or_gm<XPUTypeFP16>(input_kv.numel());
  XPUTypeFP16* v_data =
      RAII_GUARD.alloc_l3_or_gm<XPUTypeFP16>(input_kv.numel());

  const XPUTypeX* loop_x[3] = {input_q_data, input_kv_data, input_kv_data};
  XPUTypeFP16* loop_y[3] = {q_data, k_data, v_data};
  std::vector<const int16_t*> fc_weight_data_int16_t;
  std::vector<const float*> fc_weight_max_data;
  std::vector<const float*> fc_bias_data;
  for (size_t i = 0; i < fc_weight.size(); i++) {
    fc_weight_data_int16_t.emplace_back(
        reinterpret_cast<const int16_t*>(fc_weight[i]->data()));
    fc_weight_max_data.push_back(fc_weight_max[i]->data<float>());
    fc_bias_data.emplace_back(fc_bias[i]->data<float>());
  }

  int batch = input_q.dims()[0];
  int max_q_len = input_q.dims()[1];
  int max_kv_len = input_kv.dims()[1];
  int max_seq_len = std::max(max_q_len, max_kv_len);
  int qkv_shape = 0;  // B x L x H x D
  int hidden_dim = head_num * head_dim;
  int q_mul_m = batch * max_q_len;
  int kv_mul_m = batch * max_kv_len;
  int loop_m[3] = {q_mul_m, kv_mul_m, kv_mul_m};
  int n = hidden_dim;
  int k = hidden_dim;
  bool do_fc_qkv_fusion = false;
  xpu::Activation_t act_type = xpu::Activation_t::LINEAR;

  // q_mul + k_mul + v_mul
  for (int i = 0; i < 3; ++i) {
    int r = xpu::
        fc_fusion<XPUTypeX, XPUTypeW, XPUTypeFP16, T_GEMM>(  // TX/TW/TY/TGEMM
            ctx.x_context(),                                 // ctx
            loop_x[i],                                       // x
            fc_weight_data_int16_t[i],                       // w
            loop_y[i],                                       // y
            loop_m[i],                                       // m
            n,                                               // n
            k,                                               // k
            false,                                           // x_trans
            false,                                           // w_trans
            nullptr,                                         // x_maxptr
            fc_weight_max_data[i],                           // w_maxptr
            nullptr,                                         // y_maxptr
            hidden_dim,                                      // ldx
            hidden_dim,                                      // ldw
            hidden_dim,                                      // ldy
            1.0f,                                            // alpha
            0.0f,                                            // beta
            fc_bias_data[i],                                 // bias
            act_type);                                       // act

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_xpu");
  }
  auto mask_dim = mask.dims();
  int mask_dim_size = mask_dim.size();
  const float* mask_data = mask.data<float>();
  auto* qkv_data =
      reinterpret_cast<XPUTypeOut*>(ctx.template Alloc<T_QKV>(qkv));
  auto* qkv_max_data = ctx.template Alloc<float>(qkv_max);
  std::vector<int64_t> z_shape(4, 1);
  if (mask_dim_size < 4) {
    int index = 4 - mask_dim_size;
    for (int i = 0; i < mask_dim_size; ++i) {
      z_shape[index + i] = mask_dim[i];
    }
  } else {
    // mask_dim_size = 4
    // The check in fusion.cc has ensured that it is not greater than 4
    for (int i = 0; i < mask_dim_size; ++i) {
      z_shape[i] = mask_dim[i];
    }
  }
  // no vsl
  xpu::CrossAttnParam qkv_attn_param(batch,
                                     max_seq_len,
                                     head_num,
                                     head_dim,
                                     do_fc_qkv_fusion,
                                     max_q_len,
                                     max_kv_len);
  qkv_attn_param.qkv_shape = qkv_shape;
  qkv_attn_param.alpha = alpha;
  qkv_attn_param.zshape = z_shape;
  XPUTypeFP16* qkv_temp_data =
      RAII_GUARD.alloc_l3_or_gm<XPUTypeFP16>(input_q.numel());

  // qk_matmul + qkv_matmul
  int r = xpu::qkv_attention<XPUTypeFP16,
                             XPUTypeFP16,
                             XPUTypeFP16,
                             XPUTypeFP16,
                             XPUTypeGEMM>(ctx.x_context(),
                                          q_data,
                                          k_data,
                                          v_data,
                                          qkv_temp_data,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          qkv_max_data,
                                          qkv_attn_param,
                                          mask_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "qkv_attention_xpu");

  if (input_q.dtype() == DataType::FLOAT32) {
    int r_cast_out = xpu::cast<XPUTypeFP16, XPUTypeOut>(
        ctx.x_context(), qkv_temp_data, qkv_data, qkv->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(
        r_cast_out, "cross_attention_xpu(cast out from fp16 to fp32)");
  }
  if (input_q.dtype() == DataType::FLOAT16) {
    int r_copy =
        xpu::copy(ctx.x_context(), qkv_temp_data, qkv_data, qkv->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r_copy, "cross_attention_xpu(copy out)");
  }
}

#define CROSS_ATTENTION_XPU_KERNEL_IMPL(              \
    x_dtype_, w_dtype_, qkv_dtype_, gemm_dtype_)      \
  CrossAttentionXPUKernelImpl<x_dtype_,               \
                              w_dtype_,               \
                              qkv_dtype_,             \
                              gemm_dtype_,            \
                              Context>(ctx,           \
                                       input_q,       \
                                       input_kv,      \
                                       fc_weight,     \
                                       fc_weight_max, \
                                       fc_bias,       \
                                       mask,          \
                                       head_num,      \
                                       head_dim,      \
                                       alpha,         \
                                       qkv_dtype,     \
                                       qkv,           \
                                       qkv_max);

template <typename T, typename Context>
void CrossAttentionXPUKernel(
    const Context& ctx,
    const DenseTensor& input_q,
    const DenseTensor& input_kv,
    const std::vector<const DenseTensor*>& fc_weight,
    const std::vector<const DenseTensor*>& fc_weight_max,
    const std::vector<const DenseTensor*>& fc_bias,
    const DenseTensor& mask,
    int head_num,
    int head_dim,
    float alpha,
    DataType qkv_dtype,
    DenseTensor* qkv,
    DenseTensor* qkv_max) {
  VLOG(4) << "cross-attn data type: " << input_q.dtype() << " ,"
          << input_kv.dtype() << " ," << qkv_dtype;

  // Temporarily only supports the case of TY=TX
  if (input_q.dtype() == DataType::FLOAT16 &&
      input_kv.dtype() == DataType::FLOAT16 && qkv_dtype == DataType::FLOAT16) {
    // float16 kernel
    CROSS_ATTENTION_XPU_KERNEL_IMPL(
        phi::dtype::float16, int16_t, phi::dtype::float16, int16_t);
    return;
  }
  if (input_q.dtype() == DataType::FLOAT32 &&
      input_kv.dtype() == DataType::FLOAT32 && qkv_dtype == DataType::FLOAT32) {
    // float32 kernel
    CROSS_ATTENTION_XPU_KERNEL_IMPL(
        float, int16_t, phi::dtype::float16, int16_t);
    return;
  }
  PADDLE_THROW(common::errors::Unimplemented(
      "Not support q_dtype is %s, k_dtype is %s, k_dtype is %s"
      "and qkv_dtype is %s.",
      DataTypeToString(input_q.dtype()),
      DataTypeToString(input_kv.dtype()),
      DataTypeToString(qkv_dtype)));
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(cross_attention_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::CrossAttentionXPUKernel,
                   float,
                   phi::dtype::float16) {}
