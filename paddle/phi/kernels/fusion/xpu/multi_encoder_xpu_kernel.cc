// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, sofint16_tare
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void MultiEncoderXPUKernel(const Context& ctx,
                           const DenseTensor& x,
                           const std::vector<const DenseTensor*>& fc_weight,
                           const std::vector<const DenseTensor*>& fc_weight_max,
                           const std::vector<const DenseTensor*>& fc_bias,
                           const std::vector<const DenseTensor*>& ln_scale,
                           const std::vector<const DenseTensor*>& ln_bias,
                           const paddle::optional<DenseTensor>& mask,
                           int layer_num,
                           bool norm_before,
                           int hidden_dim,
                           int head_num,
                           int size_per_head,
                           int ffn_hidden_dim_scale,
                           int act_type,
                           int relative_type,
                           int slice_idx,
                           DenseTensor* out,
                           DenseTensor* x_fp16,
                           DenseTensor* out_fp16) {
  using float16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

  // XPU2 only support fp16 input/output.
  float16* x_fp16_data = reinterpret_cast<float16*>(
      ctx.template Alloc<phi::dtype::float16>(x_fp16));
  int r_cast_x = xpu::cast_v2<float, float16>(
      ctx.x_context(), x.data<T>(), x_fp16_data, x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r_cast_x,
                              "multi_encoder_xpu(cast x from fp32 to fp16)");

  float16* out_fp16_data = reinterpret_cast<float16*>(
      ctx.template Alloc<phi::dtype::float16>(out_fp16));

  // q,k,v weight are fused.
  // Each encoder's weight should be: w0, null, null, w3, w4, w5
  std::vector<const float*> fc_input_max_data;
  std::vector<const int16_t*> fc_weight_data;
  std::vector<const float*> fc_weight_max_data;
  std::vector<const float*> fc_bias_data;
  for (size_t i = 0; i < fc_weight.size(); i++) {
    fc_weight_data.push_back(fc_weight[i]->data<int16_t>());
    fc_weight_max_data.push_back(fc_weight_max[i]->data<float>());
    fc_bias_data.push_back(fc_bias[i]->data<float>());
    if (i % 4 == 0) {
      fc_weight_data.push_back(nullptr);
      fc_weight_data.push_back(nullptr);
      fc_weight_max_data.push_back(nullptr);
      fc_weight_max_data.push_back(nullptr);
      fc_bias_data.push_back(nullptr);
      fc_bias_data.push_back(nullptr);
    }
  }
  std::vector<const float*> ln_scale_data;
  std::vector<const float*> ln_bias_data;
  for (size_t i = 0; i < ln_scale.size(); i++) {
    ln_scale_data.push_back(ln_scale[i]->data<float>());
    ln_bias_data.push_back(ln_bias[i]->data<float>());
  }
  const T* mask_data =
      mask.get_ptr() == nullptr ? nullptr : mask.get_ptr()->data<T>();
  xpu::Activation_t qkv_act(static_cast<xpu::Activation_t::act_enum>(act_type));

  int batch = x.dims()[0];
  int max_seqlen = x.dims()[1];
  // matmul_size * layer_num
  std::vector<xpu::QuantType> quant_types(8 * layer_num,
                                          xpu::QuantType::NOT_QUANT);
  if (mask_data) {
    auto mask_dims = mask.get_ptr()->dims();
    std::vector<int> mask_shape(mask_dims.Get(),
                                mask_dims.Get() + mask_dims.size());
    xpu::QKVAttnParam qkv_attn_param(batch,
                                     max_seqlen,
                                     head_num,
                                     size_per_head,
                                     mask_shape,
                                     qkv_act,
                                     slice_idx,
                                     true,
                                     hidden_dim,
                                     norm_before,
                                     false);
    qkv_attn_param.quant_type_.assign(quant_types.begin(), quant_types.end());
    qkv_attn_param.scale_of_hidden_units = ffn_hidden_dim_scale;
    int r =
        xpu::transformer_encoder<float16, int16_t, int16_t>(ctx.x_context(),
                                                            x_fp16_data,
                                                            fc_weight_data,
                                                            out_fp16_data,
                                                            fc_input_max_data,
                                                            fc_weight_max_data,
                                                            fc_bias_data,
                                                            ln_scale_data,
                                                            ln_bias_data,
                                                            qkv_attn_param,
                                                            mask_data);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "multi_encoder_xpu");
  } else {
    // When no mask input, like VIT, create LOD to act as vsl.
    std::vector<int> lod;
    for (int i = 0; i < batch + 1; i++) {
      lod.push_back(i * max_seqlen);
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
                                     false);
    qkv_attn_param.quant_type_.assign(quant_types.begin(), quant_types.end());
    qkv_attn_param.scale_of_hidden_units = ffn_hidden_dim_scale;
    int r =
        xpu::transformer_encoder<float16, int16_t, int16_t>(ctx.x_context(),
                                                            x_fp16_data,
                                                            fc_weight_data,
                                                            out_fp16_data,
                                                            fc_input_max_data,
                                                            fc_weight_max_data,
                                                            fc_bias_data,
                                                            ln_scale_data,
                                                            ln_bias_data,
                                                            qkv_attn_param);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "multi_encoder_xpu");
  }

  int r_cast_out = xpu::cast_v2<float16, float>(
      ctx.x_context(), out_fp16_data, ctx.template Alloc<T>(out), out->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r_cast_out,
                              "multi_encoder_xpu(cast out from fp16 to fp32)");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(multi_encoder_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::MultiEncoderXPUKernel,
                   float) {}
