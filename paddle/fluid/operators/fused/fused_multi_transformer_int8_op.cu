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

#include "paddle/fluid/operators/fused/attn_gemm_int8.h"
#include "paddle/fluid/operators/fused/fused_multi_transformer_op.h"

namespace paddle {
namespace operators {

template <typename T>
class FusedMultiTransformerINT8OpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    auto &dev_ctx = ctx.cuda_device_context();

    auto *time_step = ctx.Input<Tensor>("TimeStep");
    // 0. input
    auto *input_x = ctx.Input<Tensor>("X");
    const auto input_x_dims = input_x->dims();
    int bsz = input_x_dims[0];
    int seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];
    int bsz_seq = bsz * seq_len;

    // quant input scales, vector, size = num_layers
    auto qkv_in_scale = ctx.Attr<std::vector<float>>("qkv_in_scale");
    auto out_linear_in_scale =
        ctx.Attr<std::vector<float>>("out_linear_in_scale");
    auto ffn1_in_scale = ctx.Attr<std::vector<float>>("ffn1_in_scale");
    auto ffn2_in_scale = ctx.Attr<std::vector<float>>("ffn2_in_scale");

    // quant round type and bound
    auto quant_round_type = ctx.Attr<int>("quant_round_type");
    auto quant_max_bound = ctx.Attr<float>("quant_max_bound");
    auto quant_min_bound = ctx.Attr<float>("quant_min_bound");

    // dequant output scales, tensor, size = [num_layers, n], n is gemm output
    // size
    auto *qkv_out_scale = ctx.Input<Tensor>("QKVOutScale");
    auto *out_linear_out_scale = ctx.Input<Tensor>("OutLinearOutScale");
    auto *ffn1_out_scale = ctx.Input<Tensor>("FFN1OutScale");
    auto *ffn2_out_scale = ctx.Input<Tensor>("FFN2OutScale");

    int qkv_out_scale_n = qkv_out_scale->dims()[1];
    int out_linear_out_scale_n = out_linear_out_scale->dims()[1];
    int ffn1_out_scale_n = ffn1_out_scale->dims()[1];
    int ffn2_out_scale_n = ffn2_out_scale->dims()[1];

    // 1. layer norm
    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const float epsilon = ctx.Attr<float>("epsilon");
    auto ln_scales = ctx.MultiInput<Tensor>("LnScale");
    auto ln_biases = ctx.MultiInput<Tensor>("LnBias");

    auto ln_compute =
        AttnLayerNorm<T, T, int8_t>(dev_ctx, epsilon, bsz_seq, dim_embed);
    Tensor ln_mean, ln_var;
    ln_mean.Resize({{bsz_seq}});
    auto *ln_mean_data =
        dev_ctx.Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
    ln_var.Resize({{bsz_seq}});
    auto *ln_var_data = dev_ctx.Alloc<U>(&ln_var, ln_var.numel() * sizeof(U));

    // 2. qkv
    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto qkv_weights = ctx.MultiInput<Tensor>("QKVW");
    auto qkv_biases = ctx.MultiInput<Tensor>("QKVBias");
    const bool trans_qkvw = ctx.Attr<bool>("trans_qkvw");
    const auto qkv_w_dims = qkv_weights[0]->dims();
    int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
    int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
    int hidden_size = num_head * dim_head;
    int output_size = 3 * hidden_size;
    int input_size = dim_embed;

    bool compute_bias = qkv_biases.size() > 0 && time_step == nullptr;
    // (transA, transB, compute_bias) = (false, trans_qkvw, false)
    AttnMatmulINT8<T> qkv_compute(
        dev_ctx, bsz_seq, output_size, input_size, compute_bias);
    Tensor qkv_out;
    qkv_out.Resize({{bsz, seq_len, 3, num_head, dim_head}});
    auto *qkv_out_data =
        dev_ctx.Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

    // 3. fmha
    AttnDropoutParam attn_param(
        true, "upscale_in_train", 0.0, true, true, 0, nullptr);
    auto fmha_compute =
        FMHARef<T>(dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);
    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto cache_kvs = ctx.MultiInput<Tensor>("CacheKV");
    auto cache_kv_outs = ctx.MultiOutput<Tensor>("CacheKVOut");
    // auto *time_step = ctx.Input<Tensor>("TimeStep");

    auto out_seq_len = seq_len;
    if (time_step) {
      PADDLE_ENFORCE_EQ(time_step->place(),
                        platform::CPUPlace(),
                        platform::errors::PreconditionNotMet(
                            "The place of input(TimeStep) must be CPUPlace."));
      // cache_seq_len
      int time_step_value = time_step->data<int>()[0];
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
    }

    Tensor transpose_out_2, qk_out;
    transpose_out_2.Resize({{3, bsz, num_head, seq_len, dim_head}});
    auto *transpose_out_2_data =
        dev_ctx.Alloc<T>(&transpose_out_2, transpose_out_2.numel() * sizeof(T));
    qk_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *qk_out_data = dev_ctx.Alloc<T>(&qk_out, qk_out.numel() * sizeof(T));

    Tensor softmax_out;
    Tensor attn_dropout_mask_out, attn_dropout_out;
    Tensor qktv_out, fmha_out;
    softmax_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *softmax_out_data =
        dev_ctx.Alloc<T>(&softmax_out, softmax_out.numel() * sizeof(T));

    attn_dropout_mask_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *attn_dropout_mask_out_data = dev_ctx.Alloc<T>(
        &attn_dropout_mask_out, attn_dropout_mask_out.numel() * sizeof(T));
    attn_dropout_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
    auto *attn_dropout_data_data = dev_ctx.Alloc<T>(
        &attn_dropout_out, attn_dropout_out.numel() * sizeof(T));

    qktv_out.Resize({{bsz, num_head, seq_len, dim_head}});
    auto *qktv_out_data =
        dev_ctx.Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
    fmha_out.Resize({{bsz, seq_len, num_head, dim_head}});
    auto *fmha_out_data =
        dev_ctx.Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));

    // 4. out_linear
    auto out_linear_weights = ctx.MultiInput<Tensor>("OutLinearW");
    auto out_linear_biases = ctx.MultiInput<Tensor>("OutLinearBias");
    int ring_id = ctx.Attr<int>("ring_id");
    // (transA, transB, compute_bias) = (false, false, false)
    AttnMatmulINT8<T> out_linear_compute(
        dev_ctx, bsz_seq, dim_embed, hidden_size, false);

    // 5. ln(residual + bias)
    DropoutParam dropout_param2(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutLayerNormHelper<T, uint8_t, int32_t, int8_t>
        fused_dropout_layernorm_helper(
            dev_ctx, bsz_seq, dim_embed, dropout_param2, epsilon);
    FusedDropoutLayerNormHelper<T, uint8_t>
        fused_dropout_layernorm_helper_for_post_layernorm(
            dev_ctx, bsz_seq, dim_embed, dropout_param2, epsilon);
    auto ffn_ln_scales = ctx.MultiInput<Tensor>("FFNLnScale");
    auto ffn_ln_biases = ctx.MultiInput<Tensor>("FFNLnBias");
    Tensor bias_dropout_residual_out, dropout_mask_out;
    T *bias_dropout_residual_out_data = nullptr;
    if (pre_layer_norm) {
      bias_dropout_residual_out.Resize({{bsz, seq_len, dim_embed}});
      bias_dropout_residual_out_data =
          dev_ctx.Alloc<T>(&bias_dropout_residual_out,
                           bias_dropout_residual_out.numel() * sizeof(T));
    }
    dropout_mask_out.Resize({{bsz, seq_len, dim_embed}});
    auto *dropout_mask_out_data = dev_ctx.Alloc<uint8_t>(
        &dropout_mask_out, dropout_mask_out.numel() * sizeof(uint8_t));

    // 6. ffn matmul1
    auto ffn1_weights = ctx.MultiInput<Tensor>("FFN1Weight");
    auto ffn1_biases = ctx.MultiInput<Tensor>("FFN1Bias");
    auto ffn1_weight_dim = ffn1_weights[0]->dims();

    int dim_ffn = ffn1_weight_dim[0];
    AttnMatmulINT8<T> ffn1_linear_compute(
        dev_ctx, bsz_seq, dim_ffn, dim_embed, false);
    Tensor ffn1_out;
    ffn1_out.Resize({{bsz_seq, dim_ffn}});
    auto *ffn1_out_data =
        dev_ctx.Alloc<T>(&ffn1_out, ffn1_out.numel() * sizeof(T));

    // 7. ffn act + bias
    DropoutParam ffn1_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutHelper<T, uint8_t, int32_t, int8_t> fused_act_dropout_helper(
        dev_ctx, bsz_seq, dim_ffn, ffn1_dropout_param);
    FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper_for_post_layernorm(
        dev_ctx, bsz_seq, dim_ffn, ffn1_dropout_param);
    Tensor ffn1_dropout_out, ffn1_dropout_mask;
    ffn1_dropout_out.Resize({{bsz_seq, dim_ffn}});
    auto *ffn1_dropout_out_data = dev_ctx.Alloc<T>(
        &ffn1_dropout_out, ffn1_dropout_out.numel() * sizeof(T));
    ffn1_dropout_mask.Resize({{bsz_seq, dim_ffn}});
    auto *ffn1_dropout_mask_data = dev_ctx.Alloc<uint8_t>(
        &ffn1_dropout_mask, ffn1_dropout_mask.numel() * sizeof(uint8_t));

    // 8. ffn2 matmul
    auto ffn2_weights = ctx.MultiInput<Tensor>("FFN2Weight");
    auto ffn2_biases = ctx.MultiInput<Tensor>("FFN2Bias");
    AttnMatmulINT8<T> ffn2_linear_compute(
        dev_ctx, bsz_seq, dim_embed, dim_ffn, false);

    // 9. ffn2 residual bias
    DropoutParam ffn2_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutLayerNormHelper<T, uint8_t, int32_t, int8_t>
        ffn2_fused_dropout_helper(
            dev_ctx, bsz_seq, dim_embed, ffn2_dropout_param, epsilon);
    FusedDropoutLayerNormHelper<T, uint8_t, int32_t, T>
        ffn2_fused_dropout_dequant_helper(
            dev_ctx, bsz_seq, dim_embed, ffn2_dropout_param, epsilon);
    FusedDropoutLayerNormHelper<T, uint8_t>
        ffn2_fused_dropout_helper_for_post_layernorm(
            dev_ctx, bsz_seq, dim_embed, ffn2_dropout_param, epsilon);

    // []. init workspace for cublasLt transform
    Tensor input_workspace, output_workspace;
    // for input and output transform data is CUBLASLT_ORDER_COL32 format,
    int m_max = bsz_seq, k_max = std::max(dim_embed, dim_ffn),
        n_max = std::max({output_size, dim_embed, dim_ffn});

    input_workspace.Resize(
        {{32 * ((m_max + 32 - 1) / 32), (k_max + 31) / 32 * 32}});
    dev_ctx.Alloc<int8_t>(&input_workspace,
                          input_workspace.numel() * sizeof(int8_t));
    output_workspace.Resize({{n_max * 4, (m_max + 31) / 32 * 32 * 4}});
    dev_ctx.Alloc<int32_t>(&output_workspace,
                           output_workspace.numel() * sizeof(int32_t));

    // calc
    auto *out = ctx.Output<Tensor>("Out");
    auto *from_data = dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));
    Tensor *from_tensor = out;
    Tensor tmp_out;
    tmp_out.Resize({{bsz, seq_len, dim_embed}});
    auto *tmp_out_data =
        dev_ctx.Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

    auto *x_data = input_x->data<T>();
    Tensor *buf0 = nullptr;
    Tensor *buf1 = nullptr;

    // step0:  x   --> buf1
    // step1: buf1 --> buf0
    // step2: buf0 --> buf1
    int layers = qkv_weights.size();
    if (pre_layer_norm) {
      buf1 = out;
    } else {
      buf0 = &tmp_out;
      buf1 = out;
    }

    for (int i = 0; i < layers; ++i) {
      // step1. layer_norm
      if (i == 0 && pre_layer_norm) {
        auto *ln_scale_data = ln_scales[i]->data<U>();
        auto *ln_bias_data = ln_biases[i]->data<U>();
        // TODO(wangxi): can remove mean var in inference
        ln_compute.ComputeForward(x_data,
                                  ln_scale_data,
                                  ln_bias_data,
                                  input_workspace.data<int8_t>(),
                                  ln_mean_data,
                                  ln_var_data,
                                  nullptr,
                                  0,
                                  qkv_in_scale[i],
                                  quant_round_type,
                                  quant_max_bound,
                                  quant_min_bound);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step1";
#endif

      // step2. qkv
      const Tensor *qkv_bias = qkv_biases.size() > 0 ? qkv_biases[i] : nullptr;
      // NOTE: in decoder stage, bias is fused in fmha
      const Tensor *bias = time_step ? nullptr : qkv_bias;
      if (!pre_layer_norm && i == 0) {
        qkv_compute.ComputeForward(qkv_weights[i],
                                   input_x,
                                   &input_workspace,
                                   bias,
                                   &qkv_out,
                                   &output_workspace,
                                   &qkv_out,
                                   qkv_in_scale[i],
                                   qkv_out_scale,
                                   i * qkv_out_scale_n,
                                   quant_round_type,
                                   quant_max_bound,
                                   quant_min_bound);
      } else if (!pre_layer_norm) {
        qkv_compute.ComputeForward(qkv_weights[i],
                                   buf1,
                                   &input_workspace,
                                   bias,
                                   &qkv_out,
                                   &output_workspace,
                                   &qkv_out,
                                   qkv_in_scale[i],
                                   qkv_out_scale,
                                   i * qkv_out_scale_n,
                                   quant_round_type,
                                   quant_max_bound,
                                   quant_min_bound);
      } else {
        qkv_compute.ComputeForwardINT8ToT(qkv_weights[i],
                                          qkv_in_scale[i],
                                          &input_workspace,
                                          bias,
                                          &qkv_out,
                                          &output_workspace,
                                          &qkv_out,
                                          qkv_out_scale,
                                          i * qkv_out_scale_n);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step2";
#endif

      // step3. fmha
      const Tensor *cache_kv = cache_kvs.size() > 0 ? cache_kvs[i] : nullptr;
      Tensor *cache_kv_out = cache_kv ? cache_kv_outs[i] : nullptr;

      if (time_step) {  // generation decoder stage
        // [2, batch_size, num_head, max_seq_len, head_size]
        int max_seq_len = cache_kv->dims()[3];
        fmha<T>(dev_ctx,
                qkv_out,
                *qkv_bias,
                *src_mask,
                cache_kv_out,
                &fmha_out,
                bsz,
                max_seq_len,
                num_head,
                dim_head,
                time_step->data<int>()[0],
                1. / sqrt(dim_head));
      } else if (cache_kv_out) {  // generation context stage
        // TODO(wangxi): can remove dropout in inference
        fmha_compute.ComputeForward(qkv_out,
                                    nullptr,
                                    src_mask,
                                    &transpose_out_2,
                                    nullptr,
                                    &qk_out,
                                    nullptr,
                                    &softmax_out,
                                    &attn_dropout_mask_out,
                                    &attn_dropout_out,
                                    &qktv_out,
                                    &fmha_out);
        // [3, bsz, num_head, seq_len, head_dim]
        T *qkv_data = transpose_out_2_data;
        int64_t q_size = bsz * seq_len * num_head * dim_head;
        int64_t k_size = q_size;
        const T *q_ptr = qkv_data;
        const T *k_ptr = q_ptr + q_size;
        const T *v_ptr = k_ptr + k_size;

        // [2, bsz, num_head, max_seq_len, head_dim]
        int max_seq_len = cache_kv_out->dims()[3];
        T *cache_kv_data = cache_kv_out->data<T>();
        int64_t cache_k_size = bsz * num_head * max_seq_len * dim_head;

        T *cache_k_ptr = cache_kv_data;
        T *cache_v_ptr = cache_kv_data + cache_k_size;

        write_cache_kv<T>(dev_ctx,
                          cache_k_ptr,
                          cache_v_ptr,
                          k_ptr,
                          v_ptr,
                          bsz,
                          num_head,
                          seq_len,
                          max_seq_len,
                          dim_head);
      } else {  // not generation
        // TODO(wangxi): can remove dropout in inference
        fmha_compute.ComputeForward(qkv_out,
                                    cache_kv,
                                    src_mask,
                                    &transpose_out_2,
                                    cache_kv_out,
                                    &qk_out,
                                    nullptr,
                                    &softmax_out,
                                    &attn_dropout_mask_out,
                                    &attn_dropout_out,
                                    &qktv_out,
                                    &fmha_out);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step3";
#endif

      if (pre_layer_norm) {
        out_linear_compute.ComputeForwardTToINT8(out_linear_weights[i],
                                                 out_linear_in_scale[i],
                                                 &fmha_out,
                                                 &input_workspace,
                                                 nullptr,
                                                 &output_workspace,
                                                 nullptr,
                                                 quant_round_type,
                                                 quant_max_bound,
                                                 quant_min_bound);
        AllReduce<int32_t>(output_workspace,
                           ring_id,
                           bsz * seq_len * num_head * dim_head,
                           dev_ctx);
      } else {
        out_linear_compute.ComputeForward(out_linear_weights[i],
                                          &fmha_out,
                                          &input_workspace,
                                          nullptr,
                                          buf0,
                                          &output_workspace,
                                          nullptr,
                                          out_linear_in_scale[i],
                                          out_linear_out_scale,
                                          i * out_linear_out_scale_n,
                                          quant_round_type,
                                          quant_max_bound,
                                          quant_min_bound);
        AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step4";
#endif

      // step5. ln(residual + dropout(input + bias))
      if (pre_layer_norm) {
        auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
        auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
        auto *out_linear_bias_data = out_linear_biases[i]->data<T>();

        // inplace
        // non-inplace: buf1 -> input_workspace
        fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
            dev_ctx,
            output_workspace.data<int32_t>(),
            x_data,
            out_linear_bias_data,
            ln_scale_data,
            ln_bias_data,
            bias_dropout_residual_out_data,
            dropout_mask_out_data,
            input_workspace.data<int8_t>(),
            ln_mean_data,
            ln_var_data,
            out_linear_in_scale[i],
            out_linear_out_scale->data<float>(),
            i * out_linear_out_scale_n,
            ffn1_in_scale[i],
            quant_round_type,
            quant_max_bound,
            quant_min_bound);
      } else {
        auto *ln_scale_data = ln_scales[i]->data<U>();
        auto *ln_bias_data = ln_biases[i]->data<U>();
        auto *out_linear_bias_data = out_linear_biases[i]->data<T>();
        auto *residual_data = (i == 0 ? x_data : buf1->data<T>());
        fused_dropout_layernorm_helper_for_post_layernorm
            .LayernormResidualDropoutBias(dev_ctx,
                                          buf0->data<T>(),
                                          residual_data,
                                          out_linear_bias_data,
                                          ln_scale_data,
                                          ln_bias_data,
                                          buf0->data<T>(),
                                          dropout_mask_out_data,
                                          buf1->data<T>(),
                                          ln_mean_data,
                                          ln_var_data);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step5";
#endif

      // step6. ffn matmul1

      if (pre_layer_norm) {
        ffn1_linear_compute.ComputeForwardINT8ToINT8(ffn1_weights[i],
                                                     &input_workspace,
                                                     nullptr,
                                                     &output_workspace,
                                                     nullptr);
      } else {
        ffn1_linear_compute.ComputeForward(ffn1_weights[i],
                                           buf1,
                                           &input_workspace,
                                           nullptr,
                                           &ffn1_out,
                                           &output_workspace,
                                           nullptr,
                                           ffn1_in_scale[i],
                                           ffn1_out_scale,
                                           i * ffn1_out_scale_n,
                                           quant_round_type,
                                           quant_max_bound,
                                           quant_min_bound);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step6";
#endif

      // step7. act bias
      // TODO(wangxi): remove dropout mask in inference
      if (pre_layer_norm) {
        fused_act_dropout_helper.DropoutActBias(
            dev_ctx,
            output_workspace.data<int32_t>(),
            ffn1_biases[i]->data<T>(),
            "gelu",
            input_workspace.data<int8_t>(),
            ffn1_dropout_mask_data,
            ffn1_in_scale[i],
            ffn1_out_scale->data<float>(),
            i * ffn1_out_scale_n,
            ffn2_in_scale[i],
            quant_round_type,
            quant_max_bound,
            quant_min_bound);
      } else {
        fused_act_dropout_helper_for_post_layernorm.DropoutActBias(
            dev_ctx,
            ffn1_out_data,
            ffn1_biases[i]->data<T>(),
            "gelu",
            ffn1_dropout_out_data,
            ffn1_dropout_mask_data);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step7";
#endif

      // step8. ffn matmul2
      if (pre_layer_norm) {
        ffn2_linear_compute.ComputeForwardINT8ToINT8(ffn2_weights[i],
                                                     &input_workspace,
                                                     nullptr,
                                                     &output_workspace,
                                                     nullptr);
      } else {
        ffn2_linear_compute.ComputeForward(ffn2_weights[i],
                                           &ffn1_dropout_out,
                                           &input_workspace,
                                           nullptr,
                                           buf0,
                                           &output_workspace,
                                           nullptr,
                                           ffn2_in_scale[i],
                                           ffn2_out_scale,
                                           i * ffn2_out_scale_n,
                                           quant_round_type,
                                           quant_max_bound,
                                           quant_min_bound);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step8.0";
#endif

      if (pre_layer_norm) {
        AllReduce<int32_t>(output_workspace,
                           ring_id,
                           bsz * seq_len * num_head * dim_head,
                           dev_ctx);
      } else {
        AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step8.1";
#endif

      // step9. residual bias
      if (pre_layer_norm) {
        // TODO(wangxi): remove dropout mask in inference
        if (i < layers - 1) {
          auto *ln_scale_data = ln_scales[i + 1]->data<U>();
          auto *ln_bias_data = ln_biases[i + 1]->data<U>();

          ffn2_fused_dropout_helper.LayernormResidualDropoutBias(
              dev_ctx,
              output_workspace.data<int32_t>(),
              bias_dropout_residual_out_data,
              ffn2_biases[i]->data<T>(),
              ln_scale_data,
              ln_bias_data,
              buf1->data<T>(),
              dropout_mask_out_data,
              input_workspace.data<int8_t>(),
              ln_mean_data,
              ln_var_data,
              ffn2_in_scale[i],
              ffn2_out_scale->data<float>(),
              i * ffn2_out_scale_n,
              qkv_in_scale[i + 1],
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
        } else {
          ffn2_fused_dropout_dequant_helper.ResidualDropoutBias(
              dev_ctx,
              output_workspace.data<int32_t>(),
              bias_dropout_residual_out_data,
              ffn2_biases[i]->data<T>(),
              buf1->data<T>(),
              dropout_mask_out_data,
              ffn2_in_scale[i],
              ffn2_out_scale->data<float>(),
              i * ffn2_out_scale_n,
              1.0);
        }
      } else {
        auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
        auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
        ffn2_fused_dropout_helper_for_post_layernorm
            .LayernormResidualDropoutBias(dev_ctx,
                                          buf0->data<T>(),
                                          buf1->data<T>(),
                                          ffn2_biases[i]->data<T>(),
                                          ln_scale_data,
                                          ln_bias_data,
                                          buf0->data<T>(),
                                          dropout_mask_out_data,
                                          buf1->data<T>(),
                                          ln_mean_data,
                                          ln_var_data);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(0) << "step9";
#endif
      if (pre_layer_norm) {
        x_data = buf1->data<T>();
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_multi_transformer_int8,
                        ops::FusedMultiTransformerINT8OpKernel<plat::float16>,
                        ops::FusedMultiTransformerINT8OpKernel<float>);
