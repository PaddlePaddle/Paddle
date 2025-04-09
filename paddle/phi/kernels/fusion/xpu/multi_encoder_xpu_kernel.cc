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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/assign_kernel.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
namespace fusion {

#define TRANSFORMER_ENCODER_KERNEL_IMPL(x_dtype_, w_dtype_, gemm_dtype_) \
  int r = xpu::transformer_encoder<x_dtype_, w_dtype_, gemm_dtype_>(     \
      ctx.x_context(),                                                   \
      x_fp16_data,                                                       \
      fc_weight_data_##w_dtype_,                                         \
      out_fp16_data,                                                     \
      fc_input_max_data,                                                 \
      fc_weight_max_data,                                                \
      fc_bias_data,                                                      \
      ln_scale_data,                                                     \
      ln_bias_data,                                                      \
      qkv_attn_param,                                                    \
      mask_data);                                                        \
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "multi_encoder_xpu");

template <typename T, typename Context>
void MultiEncoderXPUKernel(
    const Context& ctx,
    const DenseTensor& x,
    const std::vector<const DenseTensor*>& fc_input_max,
    const std::vector<const DenseTensor*>& fc_weight,
    const std::vector<const DenseTensor*>& fc_weight_max,
    const std::vector<const DenseTensor*>& fc_bias,
    const std::vector<const DenseTensor*>& ln_scale,
    const std::vector<const DenseTensor*>& ln_bias,
    const std::vector<const DenseTensor*>& smooth_scale_weight,
    const std::vector<const DenseTensor*>& roformer_embedding,
    const paddle::optional<DenseTensor>& mask,
    const paddle::optional<DenseTensor>& seq_lod,
    const paddle::optional<DenseTensor>& max_seq_len,
    int layer_num,
    bool norm_before,
    int hidden_dim,
    int head_num,
    int size_per_head,
    int ffn_hidden_dim_scale,
    int act_type,
    int relative_type,
    int slice_idx,
    bool is_per_channel,
    int max_pos_len,
    const std::vector<float>& softmax_max_value,
    const std::vector<std::string>& quant_types,
    DenseTensor* out,
    DenseTensor* x_fp16,
    DenseTensor* out_fp16) {
  const int* seq_lod_data =
      seq_lod.get_ptr() == nullptr ? nullptr : seq_lod.get_ptr()->data<int>();
  const int* max_seq_len_data = max_seq_len.get_ptr() == nullptr
                                    ? nullptr
                                    : max_seq_len.get_ptr()->data<int>();
  int batch_size = x.dims()[0];
  int seq_len = 1;
  int head_dim;
  if (x.dims().size() == 2) {
    head_dim = x.dims()[1];
  } else if (x.dims().size() == 3) {
    seq_len = x.dims()[1];
    head_dim = x.dims()[2];
  } else {
    PADDLE_ENFORCE(
        false,
        common::errors::PreconditionNotMet(
            "x.dims().size() MUST be 2 or 3, but get [%d].", x.dims().size()));
  }
  DDim out_dims;
  if (seq_lod_data) {
    batch_size = seq_lod.get_ptr()->numel() - 1;
    seq_len = max_seq_len_data[0];
  }
  out_dims = {batch_size, seq_len, head_dim};
  if (slice_idx != -1) {
    out_dims = {batch_size, head_dim};
  }
  out->Resize(out_dims);
  out_fp16->Resize(out_dims);
  // XPU2 only support fp16 input/output.
  auto x_dtype = x.dtype();
  const XPUTypeFP16* x_fp16_data = nullptr;
  XPUTypeFP16* out_fp16_data = nullptr;
  if (x_dtype == phi::DataType::FLOAT32) {
    auto* x_fp16_data_t = reinterpret_cast<XPUTypeFP16*>(
        ctx.template Alloc<phi::dtype::float16>(x_fp16));
    int r_cast_x = xpu::cast<float, XPUTypeFP16>(
        ctx.x_context(), x.data<float>(), x_fp16_data_t, x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r_cast_x,
                                "multi_encoder_xpu(cast x from fp32 to fp16)");
    x_fp16_data = x_fp16_data_t;
    out_fp16_data = reinterpret_cast<XPUTypeFP16*>(
        ctx.template Alloc<phi::dtype::float16>(out_fp16));
  } else {
    x_fp16_data =
        reinterpret_cast<const XPUTypeFP16*>(x.data<phi::dtype::float16>());
    out_fp16_data = reinterpret_cast<XPUTypeFP16*>(
        ctx.template Alloc<phi::dtype::float16>(out));
  }

  // q,k,v weight are fused.
  // Each encoder's weight should be: w0, null, null, w3, w4, w5
  auto enable_int8 = fc_weight[0]->dtype() == phi::DataType::INT8;
  auto local_quant = fc_weight[0]->dtype() == phi::DataType::FLOAT16;
  std::vector<xpu::QuantType> set_quant_types(8 * layer_num,
                                              xpu::QuantType::NOT_QUANT);
  if (enable_int8) {
    for (size_t i = 0; i < quant_types.size(); i++) {
      if (quant_types[i] == "enable_int8") {
        set_quant_types[i] = xpu::QuantType::QUANT_INT8;
      }
    }
  }
  std::vector<const float*> fc_input_max_data;
  std::vector<const int16_t*> fc_weight_data_int16_t;
  std::vector<const XPUTypeFP16*> fc_weight_data_XPUTypeFP16;
  std::vector<const float*> fc_weight_max_data;
  std::vector<const float*> fc_bias_data;
  for (size_t i = 0; i < fc_weight.size(); i++) {
    if (!enable_int8 && local_quant) {
      fc_weight_data_XPUTypeFP16.push_back(
          reinterpret_cast<const XPUTypeFP16*>(fc_weight[i]->data()));
    } else {
      // Int8 weight also convert to int16_t* for temporary storage.
      // The kernel dtype of int8 is chosen by quant_type in
      // xpu::transformer_encoder
      fc_weight_data_int16_t.push_back(
          reinterpret_cast<const int16_t*>(fc_weight[i]->data()));
    }
    fc_weight_max_data.push_back(fc_weight_max[i]->data<float>());
    fc_bias_data.push_back(fc_bias[i]->data<float>());
    if (i % 4 == 0) {
      fc_weight_data_int16_t.push_back(nullptr);
      fc_weight_data_int16_t.push_back(nullptr);
      fc_weight_data_XPUTypeFP16.push_back(nullptr);
      fc_weight_data_XPUTypeFP16.push_back(nullptr);
      fc_weight_max_data.push_back(nullptr);
      fc_weight_max_data.push_back(nullptr);
      fc_bias_data.push_back(nullptr);
      fc_bias_data.push_back(nullptr);
    }
  }

  for (size_t i = 0; i < fc_input_max.size(); i++) {
    fc_input_max_data.push_back(fc_input_max[i]->data<float>());
  }

  std::vector<const float*> ln_scale_data;
  std::vector<const float*> ln_bias_data;
  for (size_t i = 0; i < ln_scale.size(); i++) {
    ln_scale_data.push_back(ln_scale[i]->data<float>());
    ln_bias_data.push_back(ln_bias[i]->data<float>());
  }
  const float* mask_data =
      mask.get_ptr() == nullptr ? nullptr : mask.get_ptr()->data<float>();

  xpu::Activation_t qkv_act(static_cast<xpu::Activation_t::act_enum>(act_type));

  int batch = x.dims()[0];
  // matmul_size * layer_num
  if (seq_lod_data) {
    xpu::VectorParam<int> query_lod = {
        seq_lod_data, seq_lod.get_ptr()->numel(), nullptr};
    int max_seq_len_value = slice_idx == -1 ? max_seq_len_data[0] : -1;
    xpu::QKVAttnParam qkv_attn_param(query_lod,
                                     head_num,
                                     size_per_head,
                                     qkv_act,
                                     slice_idx,
                                     true,
                                     max_seq_len_value,
                                     hidden_dim,
                                     norm_before,
                                     is_per_channel);
    if (!softmax_max_value.empty()) {
      qkv_attn_param.ptq_max_value = softmax_max_value;
    }
    if (!smooth_scale_weight.empty()) {
      qkv_attn_param.is_smooth_quant = true;
      std::vector<const XPUTypeFP16*> smooth_scale_weight_ptr;
      for (const auto& weight : smooth_scale_weight) {
        auto tmp_ptr = reinterpret_cast<const XPUTypeFP16*>(
            weight->data<phi::dtype::float16>());
        smooth_scale_weight_ptr.push_back(tmp_ptr);
      }
      qkv_attn_param.smooth_scale.assign(smooth_scale_weight_ptr.begin(),
                                         smooth_scale_weight_ptr.end());
    }
    qkv_attn_param.quant_type_.assign(set_quant_types.begin(),
                                      set_quant_types.end());
    qkv_attn_param.scale_of_hidden_units = ffn_hidden_dim_scale;
    if (!roformer_embedding.empty()) {
      std::vector<const float*> roformer_embedding_data;
      for (size_t i = 0; i < roformer_embedding.size(); i++) {
        roformer_embedding_data.push_back(roformer_embedding[i]->data<float>());
      }
      qkv_attn_param.relative_type = relative_type;
      qkv_attn_param.max_pos_len = max_pos_len;
      qkv_attn_param.relative_pos.assign(roformer_embedding_data.begin(),
                                         roformer_embedding_data.end());
    }
    if (!enable_int8 && local_quant) {
      TRANSFORMER_ENCODER_KERNEL_IMPL(XPUTypeFP16, XPUTypeFP16, float)
    } else {
      // The kernel dtype of int8 is chosen by quant_type in
      // xpu::transformer_encoder This template args, int16_t, is only for skip
      // quant fc
      TRANSFORMER_ENCODER_KERNEL_IMPL(XPUTypeFP16, int16_t, int16_t)
    }
  } else if (mask_data) {
    auto mask_dims = mask.get_ptr()->dims();
    std::vector<int> mask_shape(mask_dims.Get(),
                                mask_dims.Get() + mask_dims.size());
    int max_seq_len_value = x.dims()[1];
    xpu::QKVAttnParam qkv_attn_param(batch,
                                     max_seq_len_value,
                                     head_num,
                                     size_per_head,
                                     mask_shape,
                                     qkv_act,
                                     slice_idx,
                                     true,
                                     hidden_dim,
                                     norm_before,
                                     is_per_channel);
    if (!softmax_max_value.empty()) {
      qkv_attn_param.ptq_max_value = softmax_max_value;
    }
    if (!smooth_scale_weight.empty()) {
      qkv_attn_param.is_smooth_quant = true;
      std::vector<const XPUTypeFP16*> smooth_scale_weight_ptr;
      for (const auto& weight : smooth_scale_weight) {
        auto tmp_ptr = reinterpret_cast<const XPUTypeFP16*>(
            weight->data<phi::dtype::float16>());
        smooth_scale_weight_ptr.push_back(tmp_ptr);
      }
      qkv_attn_param.smooth_scale.assign(smooth_scale_weight_ptr.begin(),
                                         smooth_scale_weight_ptr.end());
    }
    qkv_attn_param.quant_type_.assign(set_quant_types.begin(),
                                      set_quant_types.end());
    qkv_attn_param.scale_of_hidden_units = ffn_hidden_dim_scale;
    if (!roformer_embedding.empty()) {
      std::vector<const float*> roformer_embedding_data;
      for (size_t i = 0; i < roformer_embedding.size(); i++) {
        roformer_embedding_data.push_back(roformer_embedding[i]->data<float>());
      }
      qkv_attn_param.relative_type = relative_type;
      qkv_attn_param.max_pos_len = max_pos_len;
      qkv_attn_param.relative_pos.assign(roformer_embedding_data.begin(),
                                         roformer_embedding_data.end());
    }
    if (!enable_int8 && local_quant) {
      TRANSFORMER_ENCODER_KERNEL_IMPL(XPUTypeFP16, XPUTypeFP16, float)
    } else {
      TRANSFORMER_ENCODER_KERNEL_IMPL(XPUTypeFP16, int16_t, int16_t)
    }
  } else {
    // When no mask input, like VIT, create LOD to act as vsl.
    int max_seq_len_value = x.dims()[1];
    std::vector<int> lod;
    for (int i = 0; i < batch + 1; i++) {
      lod.push_back(i * max_seq_len_value);
    }
    xpu::VectorParam<int> query_lod = {
        lod.data(), static_cast<int>(lod.size()), nullptr};
    // No need to pad, no matter slice or not
    xpu::QKVAttnParam qkv_attn_param(query_lod,
                                     head_num,
                                     size_per_head,
                                     qkv_act,
                                     slice_idx,
                                     true,
                                     -1,
                                     hidden_dim,
                                     norm_before,
                                     is_per_channel);
    if (!softmax_max_value.empty()) {
      qkv_attn_param.ptq_max_value = softmax_max_value;
    }
    if (!smooth_scale_weight.empty()) {
      qkv_attn_param.is_smooth_quant = true;
      std::vector<const XPUTypeFP16*> smooth_scale_weight_ptr;
      for (const auto& weight : smooth_scale_weight) {
        auto tmp_ptr = reinterpret_cast<const XPUTypeFP16*>(
            weight->data<phi::dtype::float16>());
        smooth_scale_weight_ptr.push_back(tmp_ptr);
      }
      qkv_attn_param.smooth_scale.assign(smooth_scale_weight_ptr.begin(),
                                         smooth_scale_weight_ptr.end());
    }
    qkv_attn_param.quant_type_.assign(set_quant_types.begin(),
                                      set_quant_types.end());
    qkv_attn_param.scale_of_hidden_units = ffn_hidden_dim_scale;
    if (!roformer_embedding.empty()) {
      std::vector<const float*> roformer_embedding_data;
      for (size_t i = 0; i < roformer_embedding.size(); i++) {
        roformer_embedding_data.push_back(roformer_embedding[i]->data<float>());
      }
      qkv_attn_param.relative_type = relative_type;
      qkv_attn_param.max_pos_len = max_pos_len;
      qkv_attn_param.relative_pos.assign(roformer_embedding_data.begin(),
                                         roformer_embedding_data.end());
    }
    if (!enable_int8 && local_quant) {
      TRANSFORMER_ENCODER_KERNEL_IMPL(XPUTypeFP16, XPUTypeFP16, float)
    } else {
      TRANSFORMER_ENCODER_KERNEL_IMPL(XPUTypeFP16, int16_t, int16_t)
    }
  }

  if (x_dtype == phi::DataType::FLOAT32) {
    int r_cast_out =
        xpu::cast<XPUTypeFP16, float>(ctx.x_context(),
                                      out_fp16_data,
                                      ctx.template Alloc<float>(out),
                                      out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(
        r_cast_out, "multi_encoder_xpu(cast out from fp16 to fp32)");
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(multi_encoder_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::MultiEncoderXPUKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(10).SetBackend(phi::Backend::CPU);
  kernel->InputAt(11).SetBackend(phi::Backend::CPU);
}
