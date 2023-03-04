/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/xpu_api_wrapper.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/phi/api/include/tensor.h"
#ifdef PADDLE_WITH_XPU_XFT
#include "models/fused_multi_transformer_op.h"
#endif

namespace xft = baidu::xpu::xft;

namespace paddle {
namespace operators {

/*
template <typename T, typename TW>
static std::pair<TW *, float *> quant_weight(xpu::Context *xpu_ctx,
                                             xpu::ctx_guard &RAII_GUARD,
                                             const phi::DenseTensor *x_tensor,
                                             bool need_transpose = false) {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

  auto x_dims = x_tensor->dims();
  auto x_numel = x_tensor->numel();
  XPUTypeT *x =
      reinterpret_cast<XPUTypeT *>(const_cast<T *>(x_tensor->data<T>()));
  TW *y = RAII_GUARD.alloc<TW>(x_numel);
  float *y_max = RAII_GUARD.alloc<float>(xpu_ctx->max_ptr_size());

  xpu::ctx_guard tmp_RAII_GUARD(xpu_ctx);

  int r = 0;
  if (need_transpose) {
    PADDLE_ENFORCE_EQ(x_dims.size(),
                      2,
                      platform::errors::PreconditionNotMet(
                          "We expect the dims size of weight is 2, but got %d.",
                          static_cast<int>(x_dims.size())));

    XPUTypeT *x_trans = tmp_RAII_GUARD.alloc<XPUTypeT>(x_numel);
    r = xpu::transpose<XPUTypeT>(
        xpu_ctx,
        x,
        x_trans,
        {static_cast<int>(x_dims[0]), static_cast<int>(x_dims[1])},
        {1, 0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "quant_weight::transpose");
    x = x_trans;
  }

  r = xpu::findmax<XPUTypeT>(xpu_ctx, x, y_max, x_numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "quant_weight::findmax");
  r = xpu::quantization<XPUTypeT, TW>(xpu_ctx, x, y, x_numel, y_max);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "quant_weight::quantization");

  return std::make_pair(y, y_max);
}
*/

template <typename DeviceContext, typename T>
class FusedMultiTransformerOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using XPUTypeT = typename XPUTypeTrait<T>::Type;

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    xpu::Context *xpu_ctx = dev_ctx.x_context();
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    // 0. input
    auto *time_step = ctx.Input<phi::DenseTensor>("TimeStep");
    auto *input_x = ctx.Input<phi::DenseTensor>("X");
    // 1. layer norm
    auto ln_scales = ctx.MultiInput<phi::DenseTensor>("LnScale");
    auto ln_biases = ctx.MultiInput<phi::DenseTensor>("LnBias");
    // 2. qkv
    auto qkv_weights = ctx.MultiInput<phi::DenseTensor>("QKVW");
    auto qkv_biases = ctx.MultiInput<phi::DenseTensor>("QKVBias");
    // 3. fmha
    auto *src_mask = ctx.Input<phi::DenseTensor>("SrcMask");
    auto cache_kvs = ctx.MultiInput<phi::DenseTensor>("CacheKV");
    auto cache_kv_outs = ctx.MultiOutput<phi::DenseTensor>("CacheKVOut");
    // 4. out_linear
    auto out_linear_weights = ctx.MultiInput<phi::DenseTensor>("OutLinearW");
    auto out_linear_biases = ctx.MultiInput<phi::DenseTensor>("OutLinearBias");
    // (transA, transB, compute_bias) = (false, false, false)
    // 5. ln(residual + bias)
    auto ffn_ln_scales = ctx.MultiInput<phi::DenseTensor>("FFNLnScale");
    auto ffn_ln_biases = ctx.MultiInput<phi::DenseTensor>("FFNLnBias");
    // 6. ffn1 matmul + act + bias
    auto ffn1_weights = ctx.MultiInput<phi::DenseTensor>("FFN1Weight");
    auto ffn1_biases = ctx.MultiInput<phi::DenseTensor>("FFN1Bias");
    // 7. ffn2 matmul + bias + residual.
    auto ffn2_weights = ctx.MultiInput<phi::DenseTensor>("FFN2Weight");
    auto ffn2_biases = ctx.MultiInput<phi::DenseTensor>("FFN2Bias");
    // 8. out
    auto *out = ctx.Output<phi::DenseTensor>("Out");

    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    //    const float epsilon = ctx.Attr<float>("epsilon");
    const std::string act_method = ctx.Attr<std::string>("act_method");
    const bool trans_qkvw = ctx.Attr<bool>("trans_qkvw");
    PADDLE_ENFORCE_EQ(pre_layer_norm,
                      true,
                      platform::errors::PreconditionNotMet(
                          "Only support pre_layer_norm = true at now."));

    auto *rotary_tensor = ctx.Input<phi::DenseTensor>("RotaryPosEmb");
    auto pre_caches = ctx.MultiInput<phi::DenseTensor>("PreCaches");
    auto *sequence_lengths = ctx.Input<phi::DenseTensor>("SeqLengths");
    PADDLE_ENFORCE_EQ(
        sequence_lengths,
        nullptr,
        platform::errors::PreconditionNotMet("SeqLengths not support at now."));
    PADDLE_ENFORCE_EQ(rotary_tensor,
                      nullptr,
                      platform::errors::PreconditionNotMet(
                          "RotaryPosEmb not support at now."));
    PADDLE_ENFORCE_EQ(
        pre_caches.size(),
        0,
        platform::errors::PreconditionNotMet("PreCaches not support at now."));

    const auto input_x_dims = input_x->dims();
    //    int bsz = input_x_dims[0];
    int seq_len = input_x_dims[1];
    //    int dim_embed = input_x_dims[2];
    //    int bsz_seq = bsz * seq_len;
    //    int token_num = bsz_seq;
    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed] when trans_qkvw ==
    // true
    const auto qkv_w_dims = qkv_weights[0]->dims();
    int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
    int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
    //    int hidden_size = num_head * dim_head;
    //    auto ffn1_weight_dim = ffn1_weights[0]->dims();
    //    int dim_ffn = ffn1_weight_dim[1];

    int cache_offset = 0;
    auto out_seq_len = seq_len;
    int time_step_value = -1;
    if (time_step) {
      PADDLE_ENFORCE_EQ(time_step->place(),
                        platform::CPUPlace(),
                        platform::errors::PreconditionNotMet(
                            "The place of input(TimeStep) must be CPUPlace."));
      // cache_seq_len
      time_step_value = time_step->data<int>()[0];
      PADDLE_ENFORCE_GT(time_step_value,
                        0,
                        platform::errors::PreconditionNotMet(
                            "The value of time_step must > 0, but now is %d",
                            time_step_value));
      PADDLE_ENFORCE_EQ(
          seq_len,
          1,
          platform::errors::PreconditionNotMet(
              "In decode stage, the seq_len of input must be 1, but now is %d",
              seq_len));
      out_seq_len += time_step_value;
    } else {
      out_seq_len += cache_offset;
    }

    XPUTypeT *x_data =
        reinterpret_cast<XPUTypeT *>(const_cast<T *>(input_x->data<T>()));
    XPUTypeT *src_mask_data =
        reinterpret_cast<XPUTypeT *>(const_cast<T *>(src_mask->data<T>()));
    auto *out_data =
        reinterpret_cast<XPUTypeT *>(out->mutable_data<T>(ctx.GetPlace()));
    auto x_dims = input_x->dims();
    auto src_mask_dims = src_mask->dims();
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
        out_data,
        std::array<int64_t, 3>{out_dims[0], out_dims[1], out_dims[2]});

    typedef float TW;
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

    int layers = qkv_weights.size();
    //    TW* w_ptr = nullptr;
    //    float* w_max = nullptr;
    for (int i = 0; i < layers; ++i) {
      // step1. layer_norm
      LnScale.emplace_back(const_cast<float *>(ln_scales[i]->data<float>()),
                           std::array<int64_t, 1>{ln_scales[i]->dims()[0]});
      LnBias.emplace_back(const_cast<float *>(ln_biases[i]->data<float>()),
                          std::array<int64_t, 1>{ln_biases[i]->dims()[0]});
      // step2. qkv
      // asset trans_qkv_w = true
      auto qkvw_dims = qkv_weights[i]->dims();
      QKVW.emplace_back(
          const_cast<TW *>(qkv_weights[i]->data<TW>()),
          std::array<int64_t, 2>{qkvw_dims[0] * qkvw_dims[1] * qkvw_dims[2],
                                 qkvw_dims[3]});
      //      std::tie(w_ptr, w_max) = quant_weight<T, TW>(xpu_ctx, RAII_GUARD,
      //      qkv_weights[i]); QKVW.emplace_back(w_ptr, w_max,
      //      std::array<int64_t, 2>{qkvw_dims[0] * qkvw_dims[1] * qkvw_dims[2],
      //      qkvw_dims[3]});
      // assert qkv_biases.size() > 0
      auto qkvb_dims = qkv_biases[i]->dims();
      QKVBias.emplace_back(
          const_cast<float *>(qkv_biases[i]->data<float>()),
          std::array<int64_t, 1>{qkvb_dims[0] * qkvb_dims[1] * qkvb_dims[2]});
      // attn out
      auto outw_dims = out_linear_weights[i]->dims();
      OutLinearW.emplace_back(
          const_cast<TW *>(out_linear_weights[i]->data<TW>()),
          std::array<int64_t, 2>{outw_dims[0], outw_dims[1]});
      //      std::tie(w_ptr, w_max) = quant_weight<T, TW>(xpu_ctx, RAII_GUARD,
      //      out_linear_weights[i], true); OutLinearW.emplace_back(w_ptr,
      //      w_max, std::array<int64_t, 2>{outw_dims[0], outw_dims[1]});
      OutLinearBias.emplace_back(
          const_cast<float *>(out_linear_biases[i]->data<float>()),
          std::array<int64_t, 1>{out_linear_biases[i]->dims()[0]});
      // ffn ln
      FFNLnScale.emplace_back(
          const_cast<float *>(ffn_ln_scales[i]->data<float>()),
          std::array<int64_t, 1>{ffn_ln_scales[i]->dims()[0]});
      FFNLnBias.emplace_back(
          const_cast<float *>(ffn_ln_biases[i]->data<float>()),
          std::array<int64_t, 1>{ffn_ln_biases[i]->dims()[0]});
      // ffn1
      auto ffn1w_dims = ffn1_weights[i]->dims();
      FFN1Weight.emplace_back(
          const_cast<TW *>(ffn1_weights[i]->data<TW>()),
          std::array<int64_t, 2>{ffn1w_dims[0], ffn1w_dims[1]});
      //      std::tie(w_ptr, w_max) = quant_weight<T, TW>(xpu_ctx, RAII_GUARD,
      //      ffn1_weights[i], true); FFN1Weight.emplace_back(w_ptr, w_max,
      //      std::array<int64_t, 2>{ffn1w_dims[0], ffn1w_dims[1]});
      FFN1Bias.emplace_back(const_cast<float *>(ffn1_biases[i]->data<float>()),
                            std::array<int64_t, 1>{ffn1_biases[i]->dims()[0]});
      // ffn2
      auto ffn2w_dims = ffn2_weights[i]->dims();
      FFN2Weight.emplace_back(
          const_cast<TW *>(ffn2_weights[i]->data<TW>()),
          std::array<int64_t, 2>{ffn2w_dims[0], ffn2w_dims[1]});
      //      std::tie(w_ptr, w_max) = quant_weight<T, TW>(xpu_ctx, RAII_GUARD,
      //      ffn2_weights[i], true); FFN2Weight.emplace_back(w_ptr, w_max,
      //      std::array<int64_t, 2>{ffn2w_dims[0], ffn2w_dims[1]});
      FFN2Bias.emplace_back(const_cast<float *>(ffn2_biases[i]->data<float>()),
                            std::array<int64_t, 1>{ffn2_biases[i]->dims()[0]});
      // cache kv in
      if (time_step_value > 0) {
        auto cachekv_dims = cache_kvs[i]->dims();
        CacheKVIn.emplace_back(reinterpret_cast<XPUTypeT *>(
                                   const_cast<T *>(cache_kvs[i]->data<T>())),
                               std::array<int64_t, 5>{cachekv_dims[0],
                                                      cachekv_dims[1],
                                                      cachekv_dims[2],
                                                      cachekv_dims[3],
                                                      cachekv_dims[4]});
      }
      // cache kv out
      auto cachekv_dims = cache_kv_outs[i]->dims();
      CacheKVOut.emplace_back(
          reinterpret_cast<XPUTypeT *>(
              cache_kv_outs[i]->mutable_data<T>(ctx.GetPlace())),
          std::array<int64_t, 5>{cachekv_dims[0],
                                 cachekv_dims[1],
                                 cachekv_dims[2],
                                 cachekv_dims[3],
                                 cachekv_dims[4]});
    }

    xft::NlpParam param;
    param.num_layer = layers;
    param.n_head = num_head;
    param.size_per_head = dim_head;
    param.hidden_act = act_method;
    param.is_fuse_qkv = trans_qkvw;

    int r = xft::fused_multi_transformer<T, TW, int16_t>(xpu_ctx,
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
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(
    fused_multi_transformer,
    ops::FusedMultiTransformerOpKernel<phi::XPUContext, float>);
//    ops::FusedMultiTransformerOpKernel<phi::XPUContext, plat::float16>);
#endif
