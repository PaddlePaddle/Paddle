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
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_XPU_XFT
#include "models/fused_multi_transformer_op.h"
namespace xft = baidu::xpu::xft;
#endif

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedMultiTransformerXpuKernel(
    const Context& ctx,
    const DenseTensor& x,
    const std::vector<const DenseTensor*>& ln_scale,
    const std::vector<const DenseTensor*>& ln_bias,
    const std::vector<const DenseTensor*>& qkvw,
    const std::vector<const DenseTensor*>& qkvw_max,
    const std::vector<const DenseTensor*>& qkv_bias,
    const std::vector<const DenseTensor*>& out_linear_w,
    const std::vector<const DenseTensor*>& out_linear_w_max,
    const std::vector<const DenseTensor*>& out_linear_bias,
    const std::vector<const DenseTensor*>& ffn_ln_scale,
    const std::vector<const DenseTensor*>& ffn_ln_bias,
    const std::vector<const DenseTensor*>& ffn1_weight,
    const std::vector<const DenseTensor*>& ffn1_weight_max,
    const std::vector<const DenseTensor*>& ffn1_bias,
    const std::vector<const DenseTensor*>& ffn2_weight,
    const std::vector<const DenseTensor*>& ffn2_weight_max,
    const std::vector<const DenseTensor*>& ffn2_bias,
    const paddle::optional<std::vector<const DenseTensor*>>& cache_kv,
    const paddle::optional<std::vector<const DenseTensor*>>& pre_caches,
    const paddle::optional<DenseTensor>& rotary_pos_emb,
    const paddle::optional<DenseTensor>& time_step,
    const paddle::optional<DenseTensor>& seq_lengths,
    const paddle::optional<DenseTensor>& src_mask,
    bool pre_layer_norm,
    float epsilon,
    const std::string& act_method,
    bool trans_qkvw,
    DenseTensor* out,
    std::vector<DenseTensor*> cache_kv_out) {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

  PADDLE_ENFORCE_EQ(pre_layer_norm,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Only support pre_layer_norm = true at now."));
  PADDLE_ENFORCE_EQ(
      seq_lengths.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("SeqLengths not support at now."));
  PADDLE_ENFORCE_EQ(
      rotary_pos_emb.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("RotaryPosEmb not support at now."));
  PADDLE_ENFORCE_EQ(
      pre_caches.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("PreCaches not support at now."));
  PADDLE_ENFORCE_NE(
      src_mask.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("SrcMask should not be nullptr."));
  PADDLE_ENFORCE_EQ(trans_qkvw,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Only support trans_qkvw == true at now."));

  const auto x_dims = x.dims();
  int seq_len = x_dims[1];
  const auto qkv_w_dims = qkvw[0]->dims();
  int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
  int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];

  int time_step_value = -1;
  if (time_step) {
    PADDLE_ENFORCE_EQ(time_step.get_ptr()->place(),
                      phi::CPUPlace(),
                      phi::errors::PreconditionNotMet(
                          "The place of input(TimeStep) must be CPUPlace."));
    // cache_seq_len
    time_step_value = time_step.get_ptr()->data<int>()[0];
    PADDLE_ENFORCE_GT(
        time_step_value,
        0,
        phi::errors::PreconditionNotMet(
            "The value of time_step must > 0, but now is %d", time_step_value));
    PADDLE_ENFORCE_EQ(
        seq_len,
        1,
        phi::errors::PreconditionNotMet(
            "In decode stage, the seq_len of input must be 1, but now is %d",
            seq_len));
  }

  XPUTypeT* x_data = reinterpret_cast<XPUTypeT*>(const_cast<T*>(x.data<T>()));
  XPUTypeT* src_mask_data = reinterpret_cast<XPUTypeT*>(
      const_cast<T*>(src_mask.get_ptr()->data<T>()));
  auto* out_data = reinterpret_cast<XPUTypeT*>(ctx.template Alloc<T>(out));
  auto src_mask_dims = src_mask.get_ptr()->dims();
  auto out_dims = out->dims();
  auto X = xft::xftTensor<XPUTypeT, 3>(
      x_data, std::array<int64_t, 3>{x_dims[0], x_dims[1], x_dims[2]});
  auto SrcMask =
      xft::xftTensor<XPUTypeT, 4>(src_mask_data,
                                  std::array<int64_t, 4>{src_mask_dims[0],
                                                         src_mask_dims[1],
                                                         src_mask_dims[2],
                                                         src_mask_dims[3]});
  auto Out = xft::xftTensor<XPUTypeT, 3>(
      out_data, std::array<int64_t, 3>{out_dims[0], out_dims[1], out_dims[2]});

  typedef int16_t TW;
  std::vector<xft::xftVec<float>> LnScale;
  std::vector<xft::xftVec<float>> LnBias;
  std::vector<xft::xftMat<TW>> QKVW;
  std::vector<xft::xftVec<float>> QKVBias;
  std::vector<xft::xftMat<TW>> OutLinearW;
  std::vector<xft::xftVec<float>> OutLinearBias;
  std::vector<xft::xftVec<float>> FFNLnScale;
  std::vector<xft::xftVec<float>> FFNLnBias;
  std::vector<xft::xftMat<TW>> FFN1Weight;
  std::vector<xft::xftVec<float>> FFN1Bias;
  std::vector<xft::xftMat<TW>> FFN2Weight;
  std::vector<xft::xftVec<float>> FFN2Bias;
  std::vector<xft::xftTensor<float, 5>> CacheKVIn;
  std::vector<xft::xftTensor<float, 5>> CacheKVOut;

  int layers = qkvw.size();
  for (int i = 0; i < layers; ++i) {
    // step1. layer_norm
    LnScale.emplace_back(const_cast<float*>(ln_scale[i]->data<float>()),
                         std::array<int64_t, 1>{ln_scale[i]->dims()[0]});
    LnBias.emplace_back(const_cast<float*>(ln_bias[i]->data<float>()),
                        std::array<int64_t, 1>{ln_bias[i]->dims()[0]});
    // step2. qkv
    auto qkvw_dims = qkvw[i]->dims();
    QKVW.emplace_back(
        const_cast<TW*>(qkvw[i]->data<TW>()),
        const_cast<float*>(qkvw_max[i]->data<float>()),
        std::array<int64_t, 2>{qkvw_dims[0] * qkvw_dims[1] * qkvw_dims[2],
                               qkvw_dims[3]});
    auto qkvb_dims = qkv_bias[i]->dims();
    QKVBias.emplace_back(
        const_cast<float*>(qkv_bias[i]->data<float>()),
        std::array<int64_t, 1>{qkvb_dims[0] * qkvb_dims[1] * qkvb_dims[2]});
    // attn out
    auto outw_dims = out_linear_w[i]->dims();
    OutLinearW.emplace_back(
        const_cast<TW*>(out_linear_w[i]->data<TW>()),
        const_cast<float*>(out_linear_w_max[i]->data<float>()),
        std::array<int64_t, 2>{outw_dims[0], outw_dims[1]});
    OutLinearBias.emplace_back(
        const_cast<float*>(out_linear_bias[i]->data<float>()),
        std::array<int64_t, 1>{out_linear_bias[i]->dims()[0]});
    // ffn ln
    FFNLnScale.emplace_back(const_cast<float*>(ffn_ln_scale[i]->data<float>()),
                            std::array<int64_t, 1>{ffn_ln_scale[i]->dims()[0]});
    FFNLnBias.emplace_back(const_cast<float*>(ffn_ln_bias[i]->data<float>()),
                           std::array<int64_t, 1>{ffn_ln_bias[i]->dims()[0]});
    // ffn1
    auto ffn1w_dims = ffn1_weight[i]->dims();
    FFN1Weight.emplace_back(
        const_cast<TW*>(ffn1_weight[i]->data<TW>()),
        const_cast<float*>(ffn1_weight_max[i]->data<float>()),
        std::array<int64_t, 2>{ffn1w_dims[0], ffn1w_dims[1]});
    FFN1Bias.emplace_back(const_cast<float*>(ffn1_bias[i]->data<float>()),
                          std::array<int64_t, 1>{ffn1_bias[i]->dims()[0]});
    // ffn2
    auto ffn2w_dims = ffn2_weight[i]->dims();
    FFN2Weight.emplace_back(
        const_cast<TW*>(ffn2_weight[i]->data<TW>()),
        const_cast<float*>(ffn2_weight_max[i]->data<float>()),
        std::array<int64_t, 2>{ffn2w_dims[0], ffn2w_dims[1]});
    FFN2Bias.emplace_back(const_cast<float*>(ffn2_bias[i]->data<float>()),
                          std::array<int64_t, 1>{ffn2_bias[i]->dims()[0]});
    // cache kv in
    if (time_step_value > 0) {
      auto cachekv_dims = cache_kv.get_ptr()->at(i)->dims();
      CacheKVIn.emplace_back(reinterpret_cast<XPUTypeT*>(const_cast<T*>(
                                 cache_kv.get_ptr()->at(i)->data<T>())),
                             std::array<int64_t, 5>{cachekv_dims[0],
                                                    cachekv_dims[1],
                                                    cachekv_dims[2],
                                                    cachekv_dims[3],
                                                    cachekv_dims[4]});
    }
    // cache kv out
    auto cachekv_out_dims = cache_kv_out[i]->dims();
    CacheKVOut.emplace_back(
        reinterpret_cast<XPUTypeT*>(ctx.template Alloc<T>(cache_kv_out[i])),
        std::array<int64_t, 5>{cachekv_out_dims[0],
                               cachekv_out_dims[1],
                               cachekv_out_dims[2],
                               cachekv_out_dims[3],
                               cachekv_out_dims[4]});
  }

  xft::NlpParam param;
  param.num_layer = layers;
  param.n_head = num_head;
  param.size_per_head = dim_head;
  param.hidden_act = act_method;
  param.is_fuse_qkv = true;
  int r = xft::fused_multi_transformer<T, TW, int16_t>(ctx.x_context(),
                                                       X,
                                                       CacheKVIn,
                                                       SrcMask,
                                                       LnScale,
                                                       LnBias,
                                                       QKVW,
                                                       QKVBias,
                                                       OutLinearW,
                                                       OutLinearBias,
                                                       FFNLnScale,
                                                       FFNLnBias,
                                                       FFN1Weight,
                                                       FFN1Bias,
                                                       FFN2Weight,
                                                       FFN2Bias,
                                                       param,
                                                       time_step_value,
                                                       &Out,
                                                       CacheKVOut);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xft::fused_multi_transformer");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_multi_transformer_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMultiTransformerXpuKernel,
                   float) {}
