// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

template <typename T_X, typename T_QKV, typename T_GEMM, typename Context>
void QKVAttentionXPUKernelImpl(const Context& ctx,
                               const DenseTensor& q,
                               const DenseTensor& k,
                               const DenseTensor& v,
                               const paddle::optional<DenseTensor>& q_max,
                               const paddle::optional<DenseTensor>& k_max,
                               const paddle::optional<DenseTensor>& v_max,
                               const paddle::optional<DenseTensor>& qk_max,
                               const paddle::optional<DenseTensor>& qkv_max,
                               float alpha,
                               int head_num,
                               int head_dim,
                               bool qkv_fc_fusion,
                               DenseTensor* qkv) {
  using XPUTypeX = typename XPUTypeTrait<T_X>::Type;
  using XPUTypeOut = typename XPUTypeTrait<T_QKV>::Type;
  using XPUTypeGEMM = typename XPUTypeTrait<T_GEMM>::Type;
  auto* q_data = reinterpret_cast<const XPUTypeX*>(q.data<T_X>());
  auto* k_data = reinterpret_cast<const XPUTypeX*>(k.data<T_X>());
  auto* v_data = reinterpret_cast<const XPUTypeX*>(v.data<T_X>());
  if (qkv_fc_fusion) {
    k_data += head_num * head_dim;
    v_data += 2 * head_num * head_dim;
  }
  const float* q_max_data =
      q_max.get_ptr() == nullptr ? nullptr : q_max.get_ptr()->data<float>();
  const float* k_max_data =
      k_max.get_ptr() == nullptr ? nullptr : k_max.get_ptr()->data<float>();
  const float* v_max_data =
      v_max.get_ptr() == nullptr ? nullptr : v_max.get_ptr()->data<float>();
  const float* qk_max_data =
      qk_max.get_ptr() == nullptr ? nullptr : qk_max.get_ptr()->data<float>();
  float* qkv_max_data =
      qkv_max.get_ptr() == nullptr
          ? nullptr
          : const_cast<float*>(qkv_max.get_ptr()->data<float>());

  auto* qkv_data =
      reinterpret_cast<XPUTypeOut*>(ctx.template Alloc<T_QKV>(qkv));
  float* tmp_mask = nullptr;
  int batch = q.dims()[0];
  int max_seq_len = q.dims()[1];
  int qkv_shape = 0;  // B x L x H x D
  int hidden_dim = head_num * head_dim;
  // no mask input, construct a fake LOD to compute via vsl
  std::vector<int> lod;
  for (int i = 0; i < batch + 1; i++) {
    lod.emplace_back(i * max_seq_len);
  }
  xpu::VectorParam<int> query_lod = {
      lod.data(), static_cast<int>(lod.size()), nullptr};
  xpu::QKVAttnParam qkv_attn_param(query_lod, head_num, head_dim);
  qkv_attn_param.qkv_shape = qkv_shape;
  qkv_attn_param.hidden_dim = hidden_dim;
  qkv_attn_param.alpha = alpha;
  qkv_attn_param.do_fc_qkv_fusion = qkv_fc_fusion;

  // TODO(tianrui): ctrl by env
  // This feature may cause precision diff,
  // but it is more efficient, especially in long seqL cases
  bool apply_flash_attention = true;
  // Even if the switch is turned on here,
  // not all cases apply flash_attention,
  // xdnn will make further internal judgments
  if (apply_flash_attention) {
    if (std::is_same<T_GEMM, int8_t>::value) {
      if (std::is_same<T_X, float>::value) {
        phi::DenseTensor x_fp16, out_fp16;
        out_fp16.set_type(phi::DataType::FLOAT16);
        out_fp16.Resize(qkv->dims());
        x_fp16.set_type(phi::DataType::FLOAT16);
        if (qkv_fc_fusion) {
          x_fp16.Resize(q.dims());
        } else {
          std::vector<int64_t> out_dims = common::vectorize(q.dims());
          out_dims.insert(out_dims.begin(), 3);
          x_fp16.Resize(common::make_ddim(out_dims));
        }
        auto* x_fp16_data_t = reinterpret_cast<XPUTypeFP16*>(
            ctx.template Alloc<phi::dtype::float16>(&x_fp16));
        int r_cast_x;
        XPUTypeFP16* q_data_fp16 = nullptr;
        XPUTypeFP16* k_data_fp16 = nullptr;
        XPUTypeFP16* v_data_fp16 = nullptr;
        if (qkv_fc_fusion) {
          r_cast_x = xpu::cast<float, XPUTypeFP16>(
              ctx.x_context(), q.data<float>(), x_fp16_data_t, q.numel());
          PADDLE_ENFORCE_XDNN_SUCCESS(r_cast_x, "cast");
          q_data_fp16 = x_fp16_data_t;
          k_data_fp16 = x_fp16_data_t + head_num * head_dim;
          v_data_fp16 = x_fp16_data_t + 2 * head_num * head_dim;
        } else {
          r_cast_x = xpu::cast<float, XPUTypeFP16>(
              ctx.x_context(), q.data<float>(), x_fp16_data_t, q.numel());
          PADDLE_ENFORCE_XDNN_SUCCESS(r_cast_x, "cast");
          r_cast_x = xpu::cast<float, XPUTypeFP16>(ctx.x_context(),
                                                   k.data<float>(),
                                                   x_fp16_data_t + q.numel(),
                                                   k.numel());
          PADDLE_ENFORCE_XDNN_SUCCESS(r_cast_x, "cast");
          r_cast_x = xpu::cast<float, XPUTypeFP16>(
              ctx.x_context(),
              v.data<float>(),
              x_fp16_data_t + q.numel() + k.numel(),
              v.numel());
          PADDLE_ENFORCE_XDNN_SUCCESS(r_cast_x, "cast");
          q_data_fp16 = x_fp16_data_t;
          k_data_fp16 = x_fp16_data_t + q.numel();
          v_data_fp16 = x_fp16_data_t + q.numel() + k.numel();
        }
        PADDLE_ENFORCE_XDNN_SUCCESS(
            r_cast_x, "multi_encoder_xpu(cast x from fp32 to fp16)");
        auto* out_fp16_data = reinterpret_cast<XPUTypeFP16*>(
            ctx.template Alloc<phi::dtype::float16>(&out_fp16));
        int r = xpu::qkv_attention<XPUTypeFP16,
                                   XPUTypeFP16,
                                   XPUTypeFP16,
                                   XPUTypeFP16,
                                   int8_t,
                                   float,
                                   int,
                                   float,
                                   int16_t>(ctx.x_context(),
                                            q_data_fp16,
                                            k_data_fp16,
                                            v_data_fp16,
                                            out_fp16_data,
                                            q_max_data,
                                            k_max_data,
                                            v_max_data,
                                            qkv_max_data,
                                            qkv_attn_param,
                                            tmp_mask,
                                            qk_max_data);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "qkv_attention_xpu");
        int r_cast_out = xpu::cast<XPUTypeFP16, float>(
            ctx.x_context(), out_fp16_data, qkv->data<float>(), qkv->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(
            r_cast_out, "multi_encoder_xpu(cast out from fp16 to fp32)");
      } else if (std::is_same<T_X, float16>::value) {
        int r = xpu::qkv_attention<XPUTypeFP16,
                                   XPUTypeFP16,
                                   XPUTypeFP16,
                                   XPUTypeFP16,
                                   int8_t,
                                   float,
                                   int,
                                   float,
                                   int16_t>(
            ctx.x_context(),
            reinterpret_cast<const XPUTypeFP16*>(q_data),
            reinterpret_cast<const XPUTypeFP16*>(k_data),
            reinterpret_cast<const XPUTypeFP16*>(v_data),
            reinterpret_cast<XPUTypeFP16*>(qkv_data),
            q_max_data,
            k_max_data,
            v_max_data,
            qkv_max_data,
            qkv_attn_param,
            tmp_mask,
            qk_max_data);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "qkv_attention_xpu");
      } else if (std::is_same<T_X, int8_t>::value) {
        int r =
            xpu::qkv_attention<int8_t,
                               int8_t,
                               int8_t,
                               int8_t,
                               int8_t,
                               float,
                               int,
                               float,
                               int8_t>(ctx.x_context(),
                                       reinterpret_cast<const int8_t*>(q_data),
                                       reinterpret_cast<const int8_t*>(k_data),
                                       reinterpret_cast<const int8_t*>(v_data),
                                       reinterpret_cast<int8_t*>(qkv_data),
                                       q_max_data,
                                       k_max_data,
                                       v_max_data,
                                       qkv_max_data,
                                       qkv_attn_param,
                                       tmp_mask,
                                       qk_max_data);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "qkv_attention_xpu");
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "flas attention with int8 Not support q_dtype is %s.",
            DataTypeToString(q.dtype())));
      }
    } else {
      int r = xpu::qkv_attention<XPUTypeX,
                                 XPUTypeX,
                                 XPUTypeX,
                                 XPUTypeOut,
                                 int16_t,
                                 float,
                                 int,
                                 float,
                                 int16_t>(ctx.x_context(),
                                          q_data,
                                          k_data,
                                          v_data,
                                          qkv_data,
                                          q_max_data,
                                          k_max_data,
                                          v_max_data,
                                          qkv_max_data,
                                          qkv_attn_param,
                                          tmp_mask,
                                          qk_max_data);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "qkv_attention_xpu");
    }

  } else {
    int r = xpu::
        qkv_attention<XPUTypeX, XPUTypeX, XPUTypeX, XPUTypeOut, XPUTypeGEMM>(
            ctx.x_context(),
            q_data,
            k_data,
            v_data,
            qkv_data,
            q_max_data,
            k_max_data,
            v_max_data,
            qkv_max_data,
            qkv_attn_param,
            tmp_mask,
            qk_max_data);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "qkv_attention_xpu");
  }
}

#define QKV_ATTENTION_XPU_KERNEL_IMPL(x_dtype_, qkv_dtype_, gemm_dtype_) \
  QKVAttentionXPUKernelImpl<x_dtype_, qkv_dtype_, gemm_dtype_, Context>( \
      ctx,                                                               \
      q,                                                                 \
      k,                                                                 \
      v,                                                                 \
      q_max,                                                             \
      k_max,                                                             \
      v_max,                                                             \
      qk_max,                                                            \
      qkv_max,                                                           \
      alpha,                                                             \
      head_num,                                                          \
      head_dim,                                                          \
      qkv_fc_fusion,                                                     \
      qkv);

template <typename T, typename Context>
void QKVAttentionXPUKernel(const Context& ctx,
                           const DenseTensor& q,
                           const DenseTensor& k,
                           const DenseTensor& v,
                           const paddle::optional<DenseTensor>& q_max,
                           const paddle::optional<DenseTensor>& k_max,
                           const paddle::optional<DenseTensor>& v_max,
                           const paddle::optional<DenseTensor>& qk_max,
                           const paddle::optional<DenseTensor>& qkv_max,
                           float alpha,
                           int head_num,
                           int head_dim,
                           bool qkv_fc_fusion,
                           DataType qkv_dtype,
                           DenseTensor* qkv) {
  VLOG(4) << "QKV kernel type: " << q.dtype() << " ," << k.dtype() << " ,"
          << v.dtype() << " ," << qkv_dtype;
  auto use_int8 = false;
  if (q_max.get_ptr() != nullptr) {
    use_int8 = true;
  }
  if (q.dtype() == DataType::FLOAT16 && k.dtype() == DataType::FLOAT16 &&
      v.dtype() == DataType::FLOAT16 && qkv_dtype == DataType::FLOAT16) {
    // float16 kernel
    if (use_int8) {
      QKV_ATTENTION_XPU_KERNEL_IMPL(
          phi::dtype::float16, phi::dtype::float16, int8_t);
    } else {
      QKV_ATTENTION_XPU_KERNEL_IMPL(
          phi::dtype::float16, phi::dtype::float16, int16_t);
    }

  } else if (q.dtype() == DataType::FLOAT32 && k.dtype() == DataType::FLOAT32 &&
             v.dtype() == DataType::FLOAT32 && qkv_dtype == DataType::FLOAT32) {
    // float32 kernel
    if (use_int8) {
      QKV_ATTENTION_XPU_KERNEL_IMPL(float, float, int8_t);
    } else {
      QKV_ATTENTION_XPU_KERNEL_IMPL(float, float, int16_t);
    }
  } else if (q.dtype() == DataType::INT8 && k.dtype() == DataType::INT8 &&
             v.dtype() == DataType::INT8 && qkv_dtype == DataType::INT8) {
    QKV_ATTENTION_XPU_KERNEL_IMPL(int8_t, int8_t, int8_t);

  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Not support q_dtype is %s, k_dtype is %s, k_dtype is %s"
        "and qkv_dtype is %s.",
        DataTypeToString(q.dtype()),
        DataTypeToString(k.dtype()),
        DataTypeToString(v.dtype()),
        DataTypeToString(qkv_dtype)));
  }
  return;
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(qkv_attention_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::QKVAttentionXPUKernel,
                   float,
                   phi::dtype::float16,
                   int8_t) {}
