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
#include "paddle/phi/kernels/memcpy_kernel.h"
#ifdef PADDLE_WITH_XPU_XFT
#include "models/fused_multi_transformer_op.h"
namespace xft = baidu::xpu::xft;
#endif

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedMultiTransformerXpuKernel(
    const Context& ctx,
    const DenseTensor& X,
    const std::vector<const DenseTensor*>& LnScale,
    const std::vector<const DenseTensor*>& LnBias,
    const std::vector<const DenseTensor*>& QKVW,
    const std::vector<const DenseTensor*>& QKVWMax,
    const std::vector<const DenseTensor*>& QKVBias,
    const std::vector<const DenseTensor*>& OutLinearW,
    const std::vector<const DenseTensor*>& OutLinearWMax,
    const std::vector<const DenseTensor*>& OutLinearBias,
    const std::vector<const DenseTensor*>& FFNLnScale,
    const std::vector<const DenseTensor*>& FFNLnBias,
    const std::vector<const DenseTensor*>& FFN1Weight,
    const std::vector<const DenseTensor*>& FFN1WeightMax,
    const std::vector<const DenseTensor*>& FFN1Bias,
    const std::vector<const DenseTensor*>& FFN2Weight,
    const std::vector<const DenseTensor*>& FFN2WeightMax,
    const std::vector<const DenseTensor*>& FFN2Bias,
    const paddle::optional<std::vector<const DenseTensor*>>& CacheKV,
    const paddle::optional<std::vector<const DenseTensor*>>& PreCaches,
    const paddle::optional<DenseTensor>& RotaryPosEmb,
    const paddle::optional<DenseTensor>& TimeStep,
    const paddle::optional<DenseTensor>& SeqLengths,
    const paddle::optional<DenseTensor>& SrcMask,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    int ring_id,
    DenseTensor* Out,
    std::vector<DenseTensor*> CacheKVOut) {
#ifdef PADDLE_WITH_XPU_XFT
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

  PADDLE_ENFORCE_EQ(pre_layer_norm,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Only support pre_layer_norm = true at now."));
  PADDLE_ENFORCE_EQ(
      SeqLengths.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("SeqLengths not support at now."));
  PADDLE_ENFORCE_EQ(
      RotaryPosEmb.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("RotaryPosEmb not support at now."));
  PADDLE_ENFORCE_EQ(
      PreCaches.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("PreCaches not support at now."));
  PADDLE_ENFORCE_NE(
      SrcMask.get_ptr(),
      nullptr,
      phi::errors::PreconditionNotMet("SrcMask should not be nullptr."));
  PADDLE_ENFORCE_EQ(trans_qkvw,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Only support trans_qkvw == true at now."));

  const auto x_dims = X.dims();
  int seq_len = x_dims[1];
  const auto qkv_w_dims = QKVW[0]->dims();
  int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
  int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];

  int time_step_value = -1;
  if (TimeStep) {
    PADDLE_ENFORCE_EQ(TimeStep.get_ptr()->place(),
                      phi::CPUPlace(),
                      phi::errors::PreconditionNotMet(
                          "The place of input(TimeStep) must be CPUPlace."));
    // cache_seq_len
    time_step_value = TimeStep.get_ptr()->data<int>()[0];
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

  XPUTypeT* x_data = reinterpret_cast<XPUTypeT*>(const_cast<T*>(X.data<T>()));
  XPUTypeT* src_mask_data =
      reinterpret_cast<XPUTypeT*>(const_cast<T*>(SrcMask.get_ptr()->data<T>()));
  auto* out_data = reinterpret_cast<XPUTypeT*>(ctx.template Alloc<T>(Out));
  auto src_mask_dims = SrcMask.get_ptr()->dims();
  auto out_dims = Out->dims();
  auto x = xft::xftTensor<XPUTypeT, 3>(
      x_data, std::array<int64_t, 3>{x_dims[0], x_dims[1], x_dims[2]});
  // TODO(mayang02): xft support mask.dtype = float16
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  float* src_mask_fp32_data =
      RAII_GUARD.alloc<float>(SrcMask.get_ptr()->numel());
  int r = xpu::cast<XPUTypeT, float>(ctx.x_context(),
                                     src_mask_data,
                                     src_mask_fp32_data,
                                     SrcMask.get_ptr()->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu::cast");
  auto src_mask =
      xft::xftTensor<float, 4>(src_mask_fp32_data,
                               std::array<int64_t, 4>{src_mask_dims[0],
                                                      src_mask_dims[1],
                                                      src_mask_dims[2],
                                                      src_mask_dims[3]});
  auto out = xft::xftTensor<XPUTypeT, 3>(
      out_data, std::array<int64_t, 3>{out_dims[0], out_dims[1], out_dims[2]});

  typedef int16_t TW;
  std::vector<xft::xftVec<float>> ln_scale;
  std::vector<xft::xftVec<float>> ln_bias;
  std::vector<xft::xftMat<TW>> qkvw;
  std::vector<xft::xftVec<float>> qkv_bias;
  std::vector<xft::xftMat<TW>> out_linear_w;
  std::vector<xft::xftVec<float>> out_linear_bias;
  std::vector<xft::xftVec<float>> ffn_ln_scale;
  std::vector<xft::xftVec<float>> ffn_ln_bias;
  std::vector<xft::xftMat<TW>> ffn1_weight;
  std::vector<xft::xftVec<float>> ffn1_bias;
  std::vector<xft::xftMat<TW>> ffn2_weight;
  std::vector<xft::xftVec<float>> ffn2_bias;
  std::vector<xft::xftTensor<XPUTypeT, 5>> cache_kv_in;
  std::vector<xft::xftTensor<XPUTypeT, 5>> cache_kv_out;

  int layers = QKVW.size();
  for (int i = 0; i < layers; ++i) {
    // step1. layer_norm
    ln_scale.emplace_back(const_cast<float*>(LnScale[i]->data<float>()),
                          std::array<int64_t, 1>{LnScale[i]->dims()[0]});
    ln_bias.emplace_back(const_cast<float*>(LnBias[i]->data<float>()),
                         std::array<int64_t, 1>{LnBias[i]->dims()[0]});
    // step2. qkv
    auto qkvw_dims = QKVW[i]->dims();
    qkvw.emplace_back(
        const_cast<TW*>(QKVW[i]->data<TW>()),
        const_cast<float*>(QKVWMax[i]->data<float>()),
        std::array<int64_t, 2>{qkvw_dims[0] * qkvw_dims[1] * qkvw_dims[2],
                               qkvw_dims[3]});
    auto qkvb_dims = QKVBias[i]->dims();
    qkv_bias.emplace_back(
        const_cast<float*>(QKVBias[i]->data<float>()),
        std::array<int64_t, 1>{qkvb_dims[0] * qkvb_dims[1] * qkvb_dims[2]});
    // attn out
    auto outw_dims = OutLinearW[i]->dims();
    out_linear_w.emplace_back(
        const_cast<TW*>(OutLinearW[i]->data<TW>()),
        const_cast<float*>(OutLinearWMax[i]->data<float>()),
        std::array<int64_t, 2>{outw_dims[0], outw_dims[1]});
    out_linear_bias.emplace_back(
        const_cast<float*>(OutLinearBias[i]->data<float>()),
        std::array<int64_t, 1>{OutLinearBias[i]->dims()[0]});
    // ffn ln
    ffn_ln_scale.emplace_back(const_cast<float*>(FFNLnScale[i]->data<float>()),
                              std::array<int64_t, 1>{FFNLnScale[i]->dims()[0]});
    ffn_ln_bias.emplace_back(const_cast<float*>(FFNLnBias[i]->data<float>()),
                             std::array<int64_t, 1>{FFNLnBias[i]->dims()[0]});
    // ffn1
    auto ffn1w_dims = FFN1Weight[i]->dims();
    ffn1_weight.emplace_back(
        const_cast<TW*>(FFN1Weight[i]->data<TW>()),
        const_cast<float*>(FFN1WeightMax[i]->data<float>()),
        std::array<int64_t, 2>{ffn1w_dims[0], ffn1w_dims[1]});
    ffn1_bias.emplace_back(const_cast<float*>(FFN1Bias[i]->data<float>()),
                           std::array<int64_t, 1>{FFN1Bias[i]->dims()[0]});
    // ffn2
    auto ffn2w_dims = FFN2Weight[i]->dims();
    ffn2_weight.emplace_back(
        const_cast<TW*>(FFN2Weight[i]->data<TW>()),
        const_cast<float*>(FFN2WeightMax[i]->data<float>()),
        std::array<int64_t, 2>{ffn2w_dims[0], ffn2w_dims[1]});
    ffn2_bias.emplace_back(const_cast<float*>(FFN2Bias[i]->data<float>()),
                           std::array<int64_t, 1>{FFN2Bias[i]->dims()[0]});
    // cache kv in
    if (time_step_value > 0) {
      auto cachekv_dims = CacheKV.get_ptr()->at(i)->dims();
      cache_kv_in.emplace_back(reinterpret_cast<XPUTypeT*>(const_cast<T*>(
                                   CacheKV.get_ptr()->at(i)->data<T>())),
                               std::array<int64_t, 5>{cachekv_dims[0],
                                                      cachekv_dims[1],
                                                      cachekv_dims[2],
                                                      cachekv_dims[3],
                                                      cachekv_dims[4]});
    }
    // cache kv out
    auto cachekv_out_dims = CacheKVOut[i]->dims();
    cache_kv_out.emplace_back(
        reinterpret_cast<XPUTypeT*>(ctx.template Alloc<T>(CacheKVOut[i])),
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
  r = xft::fused_multi_transformer<XPUTypeT, TW, int16_t>(ctx.x_context(),
                                                          x,
                                                          cache_kv_in,
                                                          src_mask,
                                                          ln_scale,
                                                          ln_bias,
                                                          qkvw,
                                                          qkv_bias,
                                                          out_linear_w,
                                                          out_linear_bias,
                                                          ffn_ln_scale,
                                                          ffn_ln_bias,
                                                          ffn1_weight,
                                                          ffn1_bias,
                                                          ffn2_weight,
                                                          ffn2_bias,
                                                          param,
                                                          time_step_value,
                                                          &out,
                                                          cache_kv_out);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xft::fused_multi_transformer");
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "fused_multi_transformer_xpu is not supported since it's not compiled "
      "with XPU_XFT"));
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_multi_transformer_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMultiTransformerXpuKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(20).SetBackend(phi::Backend::CPU);
}
