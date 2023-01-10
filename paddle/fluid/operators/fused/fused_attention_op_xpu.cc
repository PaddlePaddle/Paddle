/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/fused/xpu_fused_common_function.h"
#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/fluid/operators/xpu_api_wrapper.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FusedAttentionOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using XPUTypeT = typename XPUTypeTrait<T>::Type;

    // inputs tensor
    auto *input_x = ctx.Input<phi::DenseTensor>("X");

    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");

    // shape [3, num_head, dim_head, dim_embed]
    auto *qkv_weight = ctx.Input<phi::DenseTensor>("QKVW");
    // shape [3 , num_head, dim_head]
    auto *qkv_bias = ctx.Input<phi::DenseTensor>("QKVBias");

    // shape [batch_size, 1, 1, seq_len]
    auto *src_mask = ctx.Input<phi::DenseTensor>("SrcMask");

    // shape [dim_embed, dim_embed]
    auto *out_linear_weight = ctx.Input<phi::DenseTensor>("OutLinearW");
    // shape [dim_embed]
    auto *out_linear_bias = ctx.Input<phi::DenseTensor>("OutLinearBias");

    const phi::DenseTensor *ln_scale = nullptr;
    const phi::DenseTensor *ln_bias = nullptr;
    float epsilon = 0.0f;

    if (pre_layer_norm) {
      ln_scale = ctx.Input<phi::DenseTensor>("LnScale");
      ln_bias = ctx.Input<phi::DenseTensor>("LnBias");
      epsilon = ctx.Attr<float>("epsilon");
    } else {
      ln_scale = ctx.Input<phi::DenseTensor>("Ln2Scale");
      ln_bias = ctx.Input<phi::DenseTensor>("Ln2Bias");
      epsilon = ctx.Attr<float>("ln_epsilon");
    }

    // outputs tensor
    // qkv 的值，并已经做了transpos后的值
    // shape [3, batch_size, num_head, seq_len, dim_head]
    auto *TransposeOut2 = ctx.Output<phi::DenseTensor>("TransposeOut2");

    // shape [batch_size, num_head, seq_len, seq_len]
    auto *softmax_out = ctx.Output<phi::DenseTensor>("SoftmaxOut");
    // shape [batch_size, num_head, seq_len, seq_len]
    auto *attn_dropout_mask_out =
        ctx.Output<phi::DenseTensor>("AttnDropoutMaskOut");
    // shape [batch_size, num_head, seq_len, seq_len]
    auto *attn_dropout_out = ctx.Output<phi::DenseTensor>("AttnDropoutOut");

    // shape [[batch_size, seq_len, num_head, dim_head]]
    auto *fmha_out = ctx.Output<phi::DenseTensor>("FMHAOut");

    // shape [batch_size, seq_len, dim_embed]
    auto *dropout_mask_out = ctx.Output<phi::DenseTensor>("DropoutMaskOut");

    // final output
    // shape [batch_size, seq_len, dim_embed]
    auto *out = ctx.Output<phi::DenseTensor>("Y");

    // 下面这个tensor是不需要返回, 但是新的动态图需要
    auto *QKOut = ctx.Output<phi::DenseTensor>("QKOut");
    QKOut->mutable_data<T>(ctx.GetPlace());
    auto *QKTVOut = ctx.Output<phi::DenseTensor>("QKTVOut");
    QKTVOut->mutable_data<T>(ctx.GetPlace());
    auto *OutLinearOut = ctx.Output<phi::DenseTensor>("OutLinearOut");
    OutLinearOut->mutable_data<T>(ctx.GetPlace());
    auto *QKVBiasOut = ctx.Output<phi::DenseTensor>("QKVBiasOut");
    QKVBiasOut->mutable_data<T>(ctx.GetPlace());
    auto *SrcMaskOut = ctx.Output<phi::DenseTensor>("SrcMaskOut");
    SrcMaskOut->mutable_data<T>(ctx.GetPlace());
    auto *qkv_out = ctx.Output<phi::DenseTensor>("QKVOut");
    qkv_out->mutable_data<T>(ctx.GetPlace());

    phi::DenseTensor *bias_dropout_residual_out = nullptr;
    phi::DenseTensor *ln_mean = nullptr;
    phi::DenseTensor *ln_var = nullptr;
    phi::DenseTensor *ln_out = nullptr;

    if (pre_layer_norm) {
      ln_mean = ctx.Output<phi::DenseTensor>("LnMean");
      ln_var = ctx.Output<phi::DenseTensor>("LnVariance");
      ln_out = ctx.Output<phi::DenseTensor>("LnOut");
    } else {
      ln_mean = ctx.Output<phi::DenseTensor>("Ln2Mean");
      ln_var = ctx.Output<phi::DenseTensor>("Ln2Variance");
      bias_dropout_residual_out =
          ctx.Output<phi::DenseTensor>("BiasDropoutResidualOut");
    }

    // dropout info
    float attn_dropout_rate = ctx.Attr<float>("attn_dropout_rate");

    bool is_test_1 = ctx.Attr<bool>("is_test");

    auto &dropout_implementation_1 =
        ctx.Attr<std::string>("attn_dropout_implementation");

    bool is_upscale_in_train_1 =
        (dropout_implementation_1 == "upscale_in_train");
    auto *seed_1 =
        ctx.HasInput("Seed1") ? ctx.Input<phi::DenseTensor>("Seed1") : nullptr;

    bool is_fix_seed_1 = ctx.Attr<bool>("attn_dropout_fix_seed");

    int seed_val_1 = ctx.Attr<int>("attn_dropout_seed");

    XPUDropoutParam attn_dropout_param;
    attn_dropout_param.initXPUDropoutParam(attn_dropout_rate,
                                           is_upscale_in_train_1,
                                           is_test_1,
                                           is_fix_seed_1,
                                           seed_1,
                                           seed_val_1);

    XPUDropoutParam dropout_param(ctx, 0);

    // 先计算纬度
    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = qkv_weight->dims();

    int batch_size = input_x_dims[0];
    int seq_len = input_x_dims[1];
    int embed_dims = input_x_dims[2];
    int num_heads = qkv_w_dims[1];
    int head_dims = qkv_w_dims[2];

    // 输入指针
    const XPUTypeT *input_x_ptr =
        reinterpret_cast<const XPUTypeT *>(input_x->data<T>());

    const XPUTypeT *qkv_weight_ptr =
        reinterpret_cast<const XPUTypeT *>(qkv_weight->data<T>());
    const XPUTypeT *qkv_bias_ptr =
        reinterpret_cast<const XPUTypeT *>(qkv_bias->data<T>());
    const XPUTypeT *src_mask_ptr =
        (src_mask == nullptr)
            ? (nullptr)
            : (reinterpret_cast<const XPUTypeT *>(src_mask->data<T>()));

    const XPUTypeT *out_linear_weight_ptr =
        reinterpret_cast<const XPUTypeT *>(out_linear_weight->data<T>());

    const XPUTypeT *out_linear_bias_ptr =
        reinterpret_cast<const XPUTypeT *>(out_linear_bias->data<T>());

    const float *ln_scale_ptr =
        (ln_scale == nullptr) ? (nullptr) : ln_scale->data<float>();

    const float *ln_bias_ptr =
        (ln_bias == nullptr) ? (nullptr) : ln_bias->data<float>();

    // 输出指针
    XPUTypeT *qkv_transpose_out_ptr = reinterpret_cast<XPUTypeT *>(
        TransposeOut2->mutable_data<T>(ctx.GetPlace()));

    XPUTypeT *softmax_out_ptr = reinterpret_cast<XPUTypeT *>(
        softmax_out->mutable_data<T>(ctx.GetPlace()));

    XPUTypeT *attn_dropout_mask_out_ptr = reinterpret_cast<XPUTypeT *>(
        attn_dropout_mask_out->mutable_data<T>(ctx.GetPlace()));

    XPUTypeT *attn_dropout_out_ptr = reinterpret_cast<XPUTypeT *>(
        attn_dropout_out->mutable_data<T>(ctx.GetPlace()));

    XPUTypeT *fmha_out_ptr =
        reinterpret_cast<XPUTypeT *>(fmha_out->mutable_data<T>(ctx.GetPlace()));

    XPUTypeT *dropout_mask_out_ptr = reinterpret_cast<XPUTypeT *>(
        dropout_mask_out->mutable_data<T>(ctx.GetPlace()));

    XPUTypeT *out_ptr =
        reinterpret_cast<XPUTypeT *>(out->mutable_data<T>(ctx.GetPlace()));

    XPUTypeT *bias_dropout_residual_out_ptr =
        (bias_dropout_residual_out == nullptr)
            ? (nullptr)
            : (reinterpret_cast<XPUTypeT *>(
                  bias_dropout_residual_out->mutable_data<T>(ctx.GetPlace())));

    float *ln_mean_ptr = (ln_mean == nullptr)
                             ? (nullptr)
                             : ln_mean->mutable_data<float>(ctx.GetPlace());

    float *ln_var_ptr = (ln_var == nullptr)
                            ? (nullptr)
                            : ln_var->mutable_data<float>(ctx.GetPlace());

    XPUTypeT *ln_out_ptr = (ln_out == nullptr)
                               ? (nullptr)
                               : (reinterpret_cast<XPUTypeT *>(
                                     ln_out->mutable_data<T>(ctx.GetPlace())));

    auto &dev_ctx = ctx.template device_context<DeviceContext>();

    xpu::Context *xpu_ctx = dev_ctx.x_context();

    xpu::ctx_guard RAII_GUARD(xpu_ctx);

    int l3_total_size = xpu_ctx->_l3_mgr.get_size();

    XPUTypeT *qkv_before_transpos_ptr =
        NULL;                 // x2[batch_size, seq_len, 3, num_heads,head_dims]
    XPUTypeT *qk_ptr = NULL;  // qk [batch_size, num_heads, seq_len, seq_len]
    XPUTypeT *qkv_ptr = NULL;  // qkv[batch_size, num_heads, seq_len, head_dims]
    XPUTypeT *linear_out_ptr =
        NULL;  // x4, x5 [batch_size, seq_len, embed_dims]

    int temp_size_1 = batch_size * seq_len * 3 * num_heads * head_dims;
    int temp_size_2 = batch_size * num_heads * seq_len * seq_len;
    int temp_size_3 = batch_size * num_heads * seq_len * head_dims;
    int temp_size_4 = batch_size * seq_len * embed_dims;

    std::vector<int> temp_vec = {
        temp_size_1, temp_size_2, temp_size_3, temp_size_4};
    std::sort(temp_vec.begin(), temp_vec.end(), std::greater<int>());
    XPUTypeT *max_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(temp_vec[0]);
    PADDLE_ENFORCE_XDNN_NOT_NULL(max_gm_ptr);
    qkv_before_transpos_ptr = max_gm_ptr;
    qk_ptr = max_gm_ptr;
    qkv_ptr = max_gm_ptr;
    linear_out_ptr = max_gm_ptr;
    int sizeof_t = sizeof(XPUTypeT);
    for (size_t i = 0; i < temp_vec.size(); ++i) {
      if (l3_total_size >= temp_vec[i] * sizeof_t) {
        XPUTypeT *l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(temp_vec[i]);
        qkv_before_transpos_ptr =
            (temp_size_1 <= temp_vec[i]) ? l3_ptr : max_gm_ptr;
        qk_ptr = (temp_size_2 <= temp_vec[i]) ? l3_ptr : max_gm_ptr;
        qkv_ptr = (temp_size_3 <= temp_vec[i]) ? l3_ptr : max_gm_ptr;
        linear_out_ptr = (temp_size_4 <= temp_vec[i]) ? l3_ptr : max_gm_ptr;
        break;
      }
    }

    int r = 0;
    const XPUTypeT *x_cacl_ptr = input_x_ptr;
    if (pre_layer_norm) {
      r = xpu::layer_norm(xpu_ctx,
                          input_x_ptr,
                          ln_out_ptr,
                          batch_size * seq_len,
                          embed_dims,
                          epsilon,
                          ln_scale_ptr,
                          ln_bias_ptr,
                          ln_mean_ptr,
                          ln_var_ptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
      x_cacl_ptr = ln_out_ptr;
    }

    // fc
    phi::XpuFcInfo qkv_fc_info;
    qkv_fc_info.InitFcInfo(0,
                           batch_size * seq_len,
                           3 * num_heads * head_dims,
                           embed_dims,
                           false,
                           true,
                           nullptr,
                           nullptr,
                           nullptr);

    phi::MatMulXPUFunction<XPUTypeT>(xpu_ctx,
                                     x_cacl_ptr,
                                     qkv_weight_ptr,
                                     qkv_before_transpos_ptr,
                                     qkv_fc_info,
                                     1.0f);

    // bias
    r = xpu::broadcast_add(xpu_ctx,
                           qkv_before_transpos_ptr,
                           qkv_bias_ptr,
                           qkv_before_transpos_ptr,
                           {batch_size * seq_len, 3 * num_heads * head_dims},
                           {3 * num_heads * head_dims});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

    // transpose
    r = xpu::transpose(xpu_ctx,
                       qkv_before_transpos_ptr,
                       qkv_transpose_out_ptr,
                       {batch_size, seq_len, 3, num_heads, head_dims},
                       {2, 0, 3, 1, 4});

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

    int qkv_every_size = batch_size * seq_len * num_heads * head_dims;
    {
      float alpha = 1.0 / sqrt(head_dims);
      r = scale(xpu_ctx,
                qkv_transpose_out_ptr,
                qkv_transpose_out_ptr,
                qkv_every_size,
                false,
                alpha,
                0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }

    // begin fhma
    // 1. qk 2. qk + mask 3. softmax 4.dropout 5. qkv 6. transpos
    {
      const XPUTypeT *q_ptr = qkv_transpose_out_ptr;
      const XPUTypeT *k_ptr = q_ptr + qkv_every_size;
      const XPUTypeT *v_ptr = k_ptr + qkv_every_size;
      phi::XpuFcInfo qk_fc_info;
      qk_fc_info.InitFcInfo(batch_size * num_heads,
                            seq_len,
                            seq_len,
                            head_dims,
                            false,
                            true,
                            nullptr,
                            nullptr,
                            nullptr);
      phi::MatMulXPUFunction<XPUTypeT>(
          xpu_ctx, q_ptr, k_ptr, qk_ptr, qk_fc_info, 1.0f);

      if (src_mask_ptr) {
        r = xpu::broadcast_add(xpu_ctx,
                               qk_ptr,
                               src_mask_ptr,
                               qk_ptr,
                               {batch_size, num_heads, seq_len, seq_len},
                               {batch_size, 1, 1, seq_len});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
      }
      // do softmax
      r = xpu::softmax(xpu_ctx,
                       qk_ptr,
                       softmax_out_ptr,
                       {batch_size, num_heads, seq_len, seq_len},
                       3);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax");

      // do dropout
      Dropout<XPUTypeT>(xpu_ctx,
                        softmax_out_ptr,
                        attn_dropout_mask_out_ptr,
                        attn_dropout_out_ptr,
                        attn_dropout_param,
                        batch_size * num_heads * seq_len * seq_len);

      phi::XpuFcInfo qktv_fc_info;
      qktv_fc_info.InitFcInfo(batch_size * num_heads,
                              seq_len,
                              head_dims,
                              seq_len,
                              false,
                              false,
                              nullptr,
                              nullptr,
                              nullptr);
      phi::MatMulXPUFunction<XPUTypeT>(
          xpu_ctx, attn_dropout_out_ptr, v_ptr, qkv_ptr, qktv_fc_info, 1.0f);
      r = xpu::transpose(xpu_ctx,
                         qkv_ptr,
                         fmha_out_ptr,
                         {batch_size, num_heads, seq_len, head_dims},
                         {0, 2, 1, 3});

      PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    }

    // linear_out
    phi::XpuFcInfo linear_fc_info;
    linear_fc_info.InitFcInfo(0,
                              batch_size * seq_len,
                              embed_dims,
                              embed_dims,
                              false,
                              false,
                              nullptr,
                              nullptr,
                              nullptr);
    phi::MatMulXPUFunction<XPUTypeT>(xpu_ctx,
                                     fmha_out_ptr,
                                     out_linear_weight_ptr,
                                     linear_out_ptr,
                                     linear_fc_info,
                                     1.0f);

    // out_linear_bias_ptr
    r = xpu::broadcast_add(xpu_ctx,
                           linear_out_ptr,
                           out_linear_bias_ptr,
                           linear_out_ptr,
                           {batch_size * seq_len, embed_dims},
                           {embed_dims});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

    Dropout(xpu_ctx,
            linear_out_ptr,
            dropout_mask_out_ptr,
            linear_out_ptr,
            dropout_param,
            batch_size * seq_len * embed_dims);

    XPUTypeT *real_out_ptr = out_ptr;
    if (pre_layer_norm == false) {
      real_out_ptr = bias_dropout_residual_out_ptr;
    }

    r = xpu::add(xpu_ctx,
                 linear_out_ptr,
                 input_x_ptr,
                 real_out_ptr,
                 batch_size * seq_len * embed_dims);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");

    if (pre_layer_norm == false) {
      r = xpu::layer_norm(xpu_ctx,
                          real_out_ptr,
                          out_ptr,
                          batch_size * seq_len,
                          embed_dims,
                          epsilon,
                          ln_scale_ptr,
                          ln_bias_ptr,
                          ln_mean_ptr,
                          ln_var_ptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
    }
  }
};

// template <typename T>
template <typename DeviceContext, typename T>
class FusedAttentionGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using XPUTypeT = typename XPUTypeTrait<T>::Type;
    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");

    // dropout info
    float attn_dropout_prob = ctx.Attr<float>("attn_dropout_rate");
    bool is_test_1 = ctx.Attr<bool>("is_test");
    auto &dropout_implementation_1 =
        ctx.Attr<std::string>("attn_dropout_implementation");
    bool is_upscale_in_train_1 =
        (dropout_implementation_1 == "upscale_in_train");
    auto *seed_1 =
        ctx.HasInput("Seed1") ? ctx.Input<phi::DenseTensor>("Seed1") : nullptr;
    bool is_fix_seed_1 = ctx.Attr<bool>("attn_dropout_fix_seed");
    int seed_val_1 = ctx.Attr<int>("attn_dropout_seed");

    XPUDropoutParam attn_dropout_param;
    attn_dropout_param.initXPUDropoutParam(attn_dropout_prob,
                                           is_upscale_in_train_1,
                                           is_test_1,
                                           is_fix_seed_1,
                                           seed_1,
                                           seed_val_1);

    XPUDropoutParam dropout_param(ctx, 0);
    // get inputs.
    auto *d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    const XPUTypeT *d_y_ptr =
        reinterpret_cast<const XPUTypeT *>(d_y->data<T>());
    // 前向必要参数
    auto *input_x = ctx.Input<phi::DenseTensor>("X");
    const XPUTypeT *input_x_ptr =
        reinterpret_cast<const XPUTypeT *>(input_x->data<T>());
    auto *qkv_transpose_out = ctx.Input<phi::DenseTensor>("TransposeOut2");
    const XPUTypeT *qkv_transpose_out_ptr =
        reinterpret_cast<const XPUTypeT *>(qkv_transpose_out->data<T>());
    auto *qkv_weight = ctx.Input<phi::DenseTensor>("QKVW");
    const XPUTypeT *qkv_weight_ptr =
        reinterpret_cast<const XPUTypeT *>(qkv_weight->data<T>());

    auto *softmax_out = ctx.Input<phi::DenseTensor>("SoftmaxOut");
    const XPUTypeT *softmax_out_ptr =
        reinterpret_cast<const XPUTypeT *>(softmax_out->data<T>());
    auto *attn_dropout_out = ctx.Input<phi::DenseTensor>("AttnDropoutOut");
    const XPUTypeT *attn_dropout_out_ptr =
        reinterpret_cast<const XPUTypeT *>(attn_dropout_out->data<T>());

    auto *attn_dropout_mask = ctx.Input<phi::DenseTensor>("AttnDropoutMaskOut");
    const XPUTypeT *attn_dropout_mask_ptr =
        reinterpret_cast<const XPUTypeT *>(attn_dropout_mask->data<T>());
    auto *fmha_out = ctx.Input<phi::DenseTensor>("FMHAOut");
    const XPUTypeT *fmha_out_ptr =
        reinterpret_cast<const XPUTypeT *>(fmha_out->data<T>());

    auto *out_linear_weight = ctx.Input<phi::DenseTensor>("OutLinearW");
    const XPUTypeT *out_linear_weight_ptr =
        reinterpret_cast<const XPUTypeT *>(out_linear_weight->data<T>());

    auto *dropout_mask_out = ctx.Input<phi::DenseTensor>("DropoutMaskOut");
    const XPUTypeT *dropout_mask_out_ptr =
        reinterpret_cast<const XPUTypeT *>(dropout_mask_out->data<T>());
    // 需要计算的梯度
    auto *d_qkv_weight =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("QKVW"));
    XPUTypeT *d_qkv_weight_ptr = reinterpret_cast<XPUTypeT *>(
        d_qkv_weight->mutable_data<T>(ctx.GetPlace()));

    auto *d_qkv_bias =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("QKVBias"));
    XPUTypeT *d_qkv_bias_ptr = reinterpret_cast<XPUTypeT *>(
        d_qkv_bias->mutable_data<T>(ctx.GetPlace()));
    auto *d_out_linear_weight =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("OutLinearW"));

    XPUTypeT *d_out_linear_weight_ptr = reinterpret_cast<XPUTypeT *>(
        d_out_linear_weight->mutable_data<T>(ctx.GetPlace()));

    auto *d_out_linear_bias =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("OutLinearBias"));
    XPUTypeT *d_out_linear_bias_ptr = reinterpret_cast<XPUTypeT *>(
        d_out_linear_bias->mutable_data<T>(ctx.GetPlace()));
    // 有可能需要
    auto *d_src_mask_out =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("SrcMaskOut"));
    XPUTypeT *d_src_mask_out_ptr =
        (d_src_mask_out == nullptr)
            ? (nullptr)
            : (reinterpret_cast<XPUTypeT *>(
                  d_src_mask_out->mutable_data<T>(ctx.GetPlace())));
    // 输出 dx
    auto *d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    XPUTypeT *d_x_ptr =
        reinterpret_cast<XPUTypeT *>(d_x->mutable_data<T>(ctx.GetPlace()));

    const phi::DenseTensor *ln_out = nullptr;
    const phi::DenseTensor *bias_dropout_residual_out = nullptr;
    const phi::DenseTensor *ln_scale = nullptr;
    const phi::DenseTensor *ln_mean = nullptr;
    const phi::DenseTensor *ln_var = nullptr;
    phi::DenseTensor *d_ln_scale = nullptr;
    phi::DenseTensor *d_ln_bias = nullptr;

    const XPUTypeT *ln_out_ptr = NULL;
    const float *ln_scale_ptr = NULL;
    const float *ln_mean_ptr = NULL;
    const float *ln_var_ptr = NULL;
    const XPUTypeT *bias_dropout_residual_out_ptr = NULL;
    float *d_ln_scale_ptr = nullptr;
    float *d_ln_bias_ptr = nullptr;

    float epsilon = 0.0f;

    if (pre_layer_norm) {
      ln_out = ctx.Input<phi::DenseTensor>("LnOut");
      ln_out_ptr = reinterpret_cast<const XPUTypeT *>(ln_out->data<T>());
      ln_scale = ctx.Input<phi::DenseTensor>("LnScale");
      ln_mean = ctx.Input<phi::DenseTensor>("LnMean");
      ln_var = ctx.Input<phi::DenseTensor>("LnVariance");
      epsilon = ctx.Attr<float>("epsilon");
      d_ln_scale =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("LnScale"));
      d_ln_bias =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("LnBias"));

    } else {
      ln_scale = ctx.Input<phi::DenseTensor>("Ln2Scale");
      ln_mean = ctx.Input<phi::DenseTensor>("Ln2Mean");
      ln_var = ctx.Input<phi::DenseTensor>("Ln2Variance");
      epsilon = ctx.Attr<float>("ln_epsilon");
      d_ln_scale =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("Ln2Scale"));
      d_ln_bias =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("Ln2Bias"));
      bias_dropout_residual_out =
          ctx.Input<phi::DenseTensor>("BiasDropoutResidualOut");
      bias_dropout_residual_out_ptr = reinterpret_cast<const XPUTypeT *>(
          bias_dropout_residual_out->data<T>());
    }

    ln_scale_ptr = ln_scale->data<float>();
    ln_mean_ptr = ln_mean->data<float>();
    ln_var_ptr = ln_var->data<float>();
    d_ln_scale_ptr = d_ln_scale->mutable_data<float>(ctx.GetPlace());
    d_ln_bias_ptr = d_ln_bias->mutable_data<float>(ctx.GetPlace());

    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = qkv_weight->dims();

    int batch_size = input_x_dims[0];
    int seq_len = input_x_dims[1];
    int embed_dims = input_x_dims[2];
    int num_heads = qkv_w_dims[1];
    int head_dims = qkv_w_dims[2];

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    xpu::Context *xpu_ctx = dev_ctx.x_context();
    xpu::ctx_guard RAII_GUARD(xpu_ctx);

    int r = 0;
    // int l3_total_size = xpu_ctx->_l3_mgr.get_size();
    XPUTypeT *d_ln_grad_ptr = NULL;       // dx5 [batch_size, seq_len, hidden]
    XPUTypeT *d_dropout_grad_ptr = NULL;  // dx5 [batch_size, seq_len, hidden]

    XPUTypeT *d_fmha_out_ptr =
        NULL;  //  d_fmha_out [batch_size, seq_len, num_heads, head_dims]
    XPUTypeT *d_fmha_out_transpos_tmp_ptr =
        NULL;  // d_fmha_out_transpos [batch_size, seq_len, num_heads,
               // head_dims]

    XPUTypeT *d_qk_ptr =
        NULL;  // d_qk_ptr[batch_size, num_heads, seq_len, seq_len]

    XPUTypeT *d_combination_qkv_ptr =
        NULL;  // d_combination_qkv_ptr[3, batch_size, num_heads, seq_len,
               // head_dims]
    XPUTypeT *d_transpos_qkv_ptr =
        NULL;  // dx2 [batch_size, seq_len, 3, num_heads, head_dims]

    XPUTypeT *d_last_layernorm_grad_ptr =
        NULL;  // d_layer_out [batch_size, seq_len, embed_dims]

    const XPUTypeT *dy_input_ptr = d_y_ptr;

    d_ln_grad_ptr =
        RAII_GUARD.alloc<XPUTypeT>(batch_size * seq_len * embed_dims);
    d_dropout_grad_ptr =
        RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(batch_size * seq_len * embed_dims);
    d_fmha_out_ptr = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(batch_size * seq_len *
                                                         num_heads * head_dims);
    d_combination_qkv_ptr =
        RAII_GUARD.alloc<XPUTypeT>(batch_size * seq_len * embed_dims * 3);
    d_transpos_qkv_ptr = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(
        batch_size * seq_len * embed_dims * 3);
    d_fmha_out_transpos_tmp_ptr =
        RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(batch_size * seq_len * embed_dims);
    d_qk_ptr = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(batch_size * seq_len *
                                                   seq_len * num_heads);
    d_last_layernorm_grad_ptr =
        RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(batch_size * seq_len * embed_dims);

    if (pre_layer_norm == false) {
      r = xpu::layer_norm_grad(xpu_ctx,
                               bias_dropout_residual_out_ptr,
                               d_y_ptr,
                               d_ln_grad_ptr,
                               batch_size * seq_len,
                               embed_dims,
                               epsilon,
                               ln_scale_ptr,
                               ln_mean_ptr,
                               ln_var_ptr,
                               d_ln_scale_ptr,
                               d_ln_bias_ptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
      dy_input_ptr = d_ln_grad_ptr;
    }
    // dropout_grad
    DropoutGrad<XPUTypeT>(xpu_ctx,
                          dy_input_ptr,
                          dropout_mask_out_ptr,
                          d_dropout_grad_ptr,
                          dropout_param,
                          batch_size * num_heads * seq_len * head_dims);

    // linear_out
    phi::XpuFcInfo linear_fc_info;
    linear_fc_info.InitFcInfo(0,
                              batch_size * seq_len,
                              embed_dims,
                              embed_dims,
                              false,
                              false,
                              nullptr,
                              nullptr,
                              nullptr);
    const XPUTypeT *a_1 = reinterpret_cast<const XPUTypeT *>(NULL);
    const XPUTypeT *b_1 = reinterpret_cast<const XPUTypeT *>(NULL);
    const XPUTypeT *a_2 = reinterpret_cast<const XPUTypeT *>(NULL);
    const XPUTypeT *b_2 = reinterpret_cast<const XPUTypeT *>(NULL);

    XPUTypeT *c_1 = d_fmha_out_ptr;
    XPUTypeT *c_2 = d_out_linear_weight_ptr;
    phi::XpuFcInfo info_dfmha;
    phi::XpuFcInfo info_dlinear_w;

    std::tuple<phi::XpuFcInfo,
               phi::XpuFcInfo,
               const XPUTypeT *,
               const XPUTypeT *,
               const XPUTypeT *,
               const XPUTypeT *>
        fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                        &RAII_GUARD,
                                        linear_fc_info,
                                        false,
                                        false,
                                        fmha_out_ptr,
                                        out_linear_weight_ptr,
                                        d_dropout_grad_ptr);

    std::tie(info_dfmha, info_dlinear_w, a_1, b_1, a_2, b_2) = fc_info;
    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_2, b_2, c_2, info_dlinear_w, 1.0f, true);

    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_1, b_1, c_1, info_dfmha, 1.0f, true);

    // dlinear_bias
    r = xpu::reduce_sum(xpu_ctx,
                        d_dropout_grad_ptr,
                        d_out_linear_bias_ptr,
                        {batch_size * seq_len, embed_dims},
                        {0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
    {
      int qkv_size = batch_size * seq_len * num_heads * head_dims;
      const XPUTypeT *q_out_ptr = qkv_transpose_out_ptr;
      const XPUTypeT *k_out_ptr = q_out_ptr + qkv_size;
      const XPUTypeT *v_out_ptr = k_out_ptr + qkv_size;
      XPUTypeT *d_q_out_ptr = d_combination_qkv_ptr;
      XPUTypeT *d_k_out_ptr = d_q_out_ptr + qkv_size;
      XPUTypeT *d_v_out_ptr = d_k_out_ptr + qkv_size;
      r = xpu::transpose<XPUTypeT>(xpu_ctx,
                                   d_fmha_out_ptr,
                                   d_fmha_out_transpos_tmp_ptr,
                                   {batch_size, seq_len, num_heads, head_dims},
                                   {0, 2, 1, 3});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

      phi::XpuFcInfo qktv_fc_info;
      qktv_fc_info.InitFcInfo(batch_size * num_heads,
                              seq_len,
                              head_dims,
                              seq_len,
                              false,
                              false,
                              nullptr,
                              nullptr,
                              nullptr);

      const XPUTypeT *a_1 = reinterpret_cast<const XPUTypeT *>(NULL);
      const XPUTypeT *b_1 = reinterpret_cast<const XPUTypeT *>(NULL);
      const XPUTypeT *a_2 = reinterpret_cast<const XPUTypeT *>(NULL);
      const XPUTypeT *b_2 = reinterpret_cast<const XPUTypeT *>(NULL);
      XPUTypeT *c_1 = d_qk_ptr;
      XPUTypeT *c_2 = d_v_out_ptr;
      phi::XpuFcInfo info_d_qk;
      phi::XpuFcInfo info_d_v;

      std::tuple<phi::XpuFcInfo,
                 phi::XpuFcInfo,
                 const XPUTypeT *,
                 const XPUTypeT *,
                 const XPUTypeT *,
                 const XPUTypeT *>
          fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                          &RAII_GUARD,
                                          qktv_fc_info,
                                          false,
                                          false,
                                          attn_dropout_out_ptr,
                                          v_out_ptr,
                                          d_fmha_out_transpos_tmp_ptr);

      std::tie(info_d_qk, info_d_v, a_1, b_1, a_2, b_2) = fc_info;
      phi::MatMulXPUFunction<XPUTypeT>(
          xpu_ctx, a_1, b_1, c_1, info_d_qk, 1.0f, true);
      phi::MatMulXPUFunction<XPUTypeT>(
          xpu_ctx, a_2, b_2, c_2, info_d_v, 1.0f, true);

      DropoutGrad<XPUTypeT>(xpu_ctx,
                            d_qk_ptr,
                            attn_dropout_mask_ptr,
                            d_qk_ptr,
                            attn_dropout_param,
                            batch_size * seq_len * seq_len * num_heads);

      r = xpu::softmax_grad<XPUTypeT>(xpu_ctx,
                                      softmax_out_ptr,
                                      d_qk_ptr,
                                      d_qk_ptr,
                                      {batch_size, num_heads, seq_len, seq_len},
                                      3);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax_grad");

      if (d_src_mask_out_ptr) {
        r = xpu::copy<XPUTypeT>(xpu_ctx,
                                d_qk_ptr,
                                d_src_mask_out_ptr,
                                batch_size * seq_len * seq_len * num_heads);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
      }
      phi::XpuFcInfo qk_fc_info;
      qk_fc_info.InitFcInfo(batch_size * num_heads,
                            seq_len,
                            seq_len,
                            head_dims,
                            false,
                            true,
                            nullptr,
                            nullptr,
                            nullptr);

      a_1 = reinterpret_cast<const XPUTypeT *>(NULL);
      b_1 = reinterpret_cast<const XPUTypeT *>(NULL);
      a_2 = reinterpret_cast<const XPUTypeT *>(NULL);
      b_2 = reinterpret_cast<const XPUTypeT *>(NULL);
      c_1 = d_q_out_ptr;
      c_2 = d_k_out_ptr;
      phi::XpuFcInfo info_d_q;
      phi::XpuFcInfo info_d_k;

      fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                      &RAII_GUARD,
                                      qk_fc_info,
                                      false,
                                      true,
                                      q_out_ptr,
                                      k_out_ptr,
                                      d_qk_ptr);

      std::tie(info_d_q, info_d_k, a_1, b_1, a_2, b_2) = fc_info;

      phi::MatMulXPUFunction<XPUTypeT>(
          xpu_ctx, a_1, b_1, c_1, info_d_q, 1.0f / sqrt(head_dims), true);

      phi::MatMulXPUFunction<XPUTypeT>(
          xpu_ctx, a_2, b_2, c_2, info_d_k, 1.0f, true);
    }

    //
    r = xpu::transpose<XPUTypeT>(xpu_ctx,
                                 d_combination_qkv_ptr,
                                 d_transpos_qkv_ptr,
                                 {3, batch_size, num_heads, seq_len, head_dims},
                                 {1, 3, 0, 2, 4});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    // dx and d_qkv_w
    phi::XpuFcInfo qkv_fc_info;
    qkv_fc_info.InitFcInfo(0,
                           batch_size * seq_len,
                           3 * num_heads * head_dims,
                           embed_dims,
                           false,
                           true,
                           nullptr,
                           nullptr,
                           nullptr);

    a_1 = reinterpret_cast<const XPUTypeT *>(NULL);
    b_1 = reinterpret_cast<const XPUTypeT *>(NULL);
    a_2 = reinterpret_cast<const XPUTypeT *>(NULL);
    b_2 = reinterpret_cast<const XPUTypeT *>(NULL);
    c_1 = (pre_layer_norm == true) ? d_last_layernorm_grad_ptr : d_x_ptr;
    c_2 = d_qkv_weight_ptr;
    phi::XpuFcInfo info_d_x;
    phi::XpuFcInfo info_d_qkv_w;

    const XPUTypeT *use_calc_input_x_ptr =
        (pre_layer_norm == true) ? ln_out_ptr : input_x_ptr;

    fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                    &RAII_GUARD,
                                    qkv_fc_info,
                                    false,
                                    true,
                                    use_calc_input_x_ptr,
                                    qkv_weight_ptr,
                                    d_transpos_qkv_ptr);

    std::tie(info_d_x, info_d_qkv_w, a_1, b_1, a_2, b_2) = fc_info;
    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_1, b_1, c_1, info_d_x, 1.0f, true);
    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_2, b_2, c_2, info_d_qkv_w, 1.0f, true);

    // d_qkv_bias
    r = xpu::reduce_sum(xpu_ctx,
                        d_transpos_qkv_ptr,
                        d_qkv_bias_ptr,
                        {batch_size * seq_len, 3 * embed_dims},
                        {0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

    if (pre_layer_norm) {
      r = xpu::layer_norm_grad(xpu_ctx,
                               input_x_ptr,
                               c_1,
                               d_x_ptr,
                               batch_size * seq_len,
                               embed_dims,
                               epsilon,
                               ln_scale_ptr,
                               ln_mean_ptr,
                               ln_var_ptr,
                               d_ln_scale_ptr,
                               d_ln_bias_ptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
    }

    // add rediaus dy
    r = xpu::add(xpu_ctx,
                 dy_input_ptr,
                 d_x_ptr,
                 d_x_ptr,
                 batch_size * seq_len * embed_dims);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    fused_attention,
    ops::FusedAttentionOpKernel<phi::XPUContext, float>,
    ops::FusedAttentionOpKernel<phi::XPUContext, paddle::platform::float16>);

REGISTER_OP_XPU_KERNEL(
    fused_attention_grad,
    ops::FusedAttentionGradXPUKernel<phi::XPUContext, float>,
    ops::FusedAttentionGradXPUKernel<phi::XPUContext,
                                     paddle::platform::float16>);

#endif
