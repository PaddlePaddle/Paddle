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

#include "paddle/fluid/operators/custom_all_reduce.h"
#include "paddle/fluid/operators/fused/fused_multi_transformer_helper.cu.h"
#include "paddle/fluid/platform/device/gpu/gpu_resource_pool.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

// #define _DEBUG_FUSED_MULTI_TRANSFORMER
// #define _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR

namespace paddle {
namespace operators {

static phi::DenseTensor CustomAllReduce(const phi::DenseTensor &t) {
  auto *ctx = static_cast<phi::GPUContext *>(
      platform::DeviceContextPool::Instance().Get(t.place()));
  auto comm = GetCustomNCCLComm(*ctx, 0);
  PADDLE_ENFORCE_NOT_NULL(comm);
  phi::DenseTensor ret;
  ret.Resize(t.dims());
  ctx->Alloc(&ret, t.dtype());
  comm->SwapInput(&ret);
  phi::Copy(*ctx, t, t.place(), false, &ret);
  return comm->AllReduce();
}

template <typename T, typename DeviceContext>
class FusedMultiTransformerOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    auto &dev_ctx = ctx.cuda_device_context();

    auto *time_step = ctx.Input<phi::DenseTensor>("TimeStep");
    // 0. input
    auto *input_x = ctx.Input<phi::DenseTensor>("X");
    const auto input_x_dims = input_x->dims();
    int bsz = input_x_dims[0];
    int seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];
    int bsz_seq = bsz * seq_len;
    const std::string act_method = ctx.Attr<std::string>("act_method");
    bool use_glu = (act_method == "geglu" || act_method == "swiglu");
    const std::string norm_type = ctx.Attr<std::string>("norm_type");
    const bool use_neox_rotary_style = ctx.Attr<bool>("use_neox_rotary_style");
    bool remove_padding = false;
    auto *sequence_lengths = ctx.Input<phi::DenseTensor>("SeqLengths");
    if (sequence_lengths) {
      remove_padding = true;
    }
    auto gqa_group_size = ctx.Attr<int>("gqa_group_size");

    VLOG(1) << "gqa_group_size is " << gqa_group_size;

    auto *beam_cache_offset = ctx.Input<phi::DenseTensor>("BeamCacheOffset");
    int beam_size = 1;
    if (beam_cache_offset) {
      beam_size = beam_cache_offset->dims()[1];
    }

    phi::DenseTensor d_token_tensor;
    phi::DenseTensor padding_offset_tensor;
    phi::DenseTensor x_remove_padding;

    // cumulative seqlens [batch_size+1]
    phi::DenseTensor cu_seqlens_q, cu_seqlens_k;
    bool encoder_remove_padding = (remove_padding && !time_step);
    int token_num = 0;

    auto *out = ctx.Output<phi::DenseTensor>("Out");
    auto *from_data = dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));

    // Init out
    if (encoder_remove_padding) {
      InitValue(dev_ctx, from_data, out->numel(), static_cast<T>(0.));
    }

    // remove padding in encoder
    if (encoder_remove_padding) {
      // just for encoder
      d_token_tensor.Resize({{1}});
      auto *d_token_num = dev_ctx.Alloc<int>(
          &d_token_tensor, d_token_tensor.numel() * sizeof(int));
      // alloc the max size of padding_offset_tensor
      padding_offset_tensor.Resize({{bsz_seq}});
      dev_ctx.Alloc<int>(&padding_offset_tensor,
                         padding_offset_tensor.numel() * sizeof(int));
      cu_seqlens_q.Resize({{bsz + 1}});
      dev_ctx.Alloc<int32_t>(&cu_seqlens_q,
                             cu_seqlens_q.numel() * sizeof(int32_t));

      InvokeGetPaddingOffset(dev_ctx,
                             &token_num,
                             d_token_num,
                             padding_offset_tensor.data<int>(),
                             cu_seqlens_q.data<int>(),
                             sequence_lengths->data<int>(),
                             bsz,
                             seq_len);
      if (token_num == 0) return;
      padding_offset_tensor.Resize({{token_num}});
      x_remove_padding.Resize({{token_num, dim_embed}});
      dev_ctx.Alloc<T>(&x_remove_padding, x_remove_padding.numel() * sizeof(T));
      InvokeRemovePadding(dev_ctx,
                          x_remove_padding.data<T>(),
                          input_x->data<T>(),
                          padding_offset_tensor.data<int>(),
                          token_num,
                          dim_embed);
    } else {
      token_num = bsz_seq;
      if (token_num == 0) return;
    }

    auto *padding_offset_data =
        encoder_remove_padding ? padding_offset_tensor.data<int>() : nullptr;

    // 1. layer norm
    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const float epsilon = ctx.Attr<float>("epsilon");
    const float residual_alpha = ctx.Attr<float>("residual_alpha");
    auto ln_scales = ctx.MultiInput<phi::DenseTensor>("LnScale");
    auto ln_biases = ctx.MultiInput<phi::DenseTensor>("LnBias");
    NormHelper<T> norm_helper(
        dev_ctx, norm_type, token_num, dim_embed, epsilon, residual_alpha);
    phi::DenseTensor ln_mean, ln_var;
    ln_mean.Resize({{token_num}});
    auto *ln_mean_data =
        dev_ctx.Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
    ln_var.Resize({{token_num}});
    auto *ln_var_data = dev_ctx.Alloc<U>(&ln_var, ln_var.numel() * sizeof(U));

    // 2. qkv
    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed] if not GQA else
    // [num_head + 2 * gqa_group_size, dim_head, dim_embed]
    auto qkv_weights = ctx.MultiInput<phi::DenseTensor>("QKVW");
    auto qkv_biases = ctx.MultiInput<phi::DenseTensor>("QKVBias");
    const bool trans_qkvw = ctx.Attr<bool>("trans_qkvw");
    const auto qkv_w_dims = qkv_weights[0]->dims();
    int num_head, dim_head;
    if (gqa_group_size > 0) {
      num_head = trans_qkvw ? (qkv_w_dims[0] - 2 * gqa_group_size)
                            : (qkv_w_dims[1] - 2 * gqa_group_size);
      dim_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
    } else {
      num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
      dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
    }
    int hidden_size = num_head * dim_head;
    int output_size = gqa_group_size <= 0
                          ? 3 * hidden_size
                          : (num_head + 2 * gqa_group_size) * dim_head;
    int input_size = dim_embed;

    auto cache_k_scale = ctx.Attr<std::vector<float>>("cache_k_scale");
    auto cache_v_scale = ctx.Attr<std::vector<float>>("cache_v_scale");
    auto cache_k_out_scale = ctx.Attr<std::vector<float>>("cache_k_out_scale");
    auto cache_v_out_scale = ctx.Attr<std::vector<float>>("cache_v_out_scale");
    bool do_cachekv_quant = (cache_k_scale.size() != 0);

    auto quant_round_type = ctx.Attr<int>("quant_round_type");
    auto quant_max_bound = ctx.Attr<float>("quant_max_bound");
    auto quant_min_bound = ctx.Attr<float>("quant_min_bound");

    // Set a flag whether need to add Matmul / Layernorm bias.
    bool compute_bias = qkv_biases.size() > 0;
    bool compute_ln_bias = ln_biases.size() > 0;

    // (transA, transB, compute_bias) = (false, trans_qkvw, false)
    // Since we fused QKVBias into QKVBiasAddTransposeSplit kernel, here we
    // set compute_bias as false.

    // auto mixed_gemm_runner = paddle::operators::CutlassFpAIntBGemmRunner<
    //     typename PDDataTypeTraits<T>::DataType,
    //     uint8_t>();
    auto qkv_compute = GEMMHelper<T>(
        dev_ctx, token_num, output_size, input_size, "None", trans_qkvw);

    phi::DenseTensor qkv_out;
    if (gqa_group_size > 0) {
      qkv_out.Resize({{token_num, num_head + 2 * gqa_group_size, dim_head}});
    } else {
      qkv_out.Resize({{token_num, 3, num_head, dim_head}});
    }
    auto *qkv_out_data =
        dev_ctx.Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

    // 2.1 rotary
    auto *rotary_tensor = ctx.Input<phi::DenseTensor>("RotaryPosEmb");
    const int rotary_emb_dims = ctx.Attr<int>("rotary_emb_dims");

    // 3. fmha
    AttnDropoutParam attn_param(
        true, "upscale_in_train", 0.0, true, true, 0, nullptr);
    auto fmha_compute =
        FMHARef<T>(dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);
    auto *src_mask = ctx.Input<phi::DenseTensor>("SrcMask");
    auto cache_kvs = ctx.MultiInput<phi::DenseTensor>("CacheKV");
    auto cache_kv_outs = ctx.MultiOutput<phi::DenseTensor>("CacheKVOut");
    // auto *time_step = ctx.Input<phi::DenseTensor>("TimeStep");
    auto pre_caches = ctx.MultiInput<phi::DenseTensor>("PreCaches");
    int cache_offset = 0;
    if (pre_caches.size() > 0) {
      cache_offset = pre_caches[0]->dims()[3];
    }

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
    } else {
      out_seq_len += cache_offset;
    }

    // whether to broadcast 2nd dimension for src_mask, default true
    // if mask_broadcast_num_heads if False, which means src_mask shape
    // will be:
    // 1. [batch_size, num_head, seq_len, seq_len] for encoder
    // 2. [batch_size, num_heads, 1, time_step+1] for decoder
    // and do not need to broadcast num_heads dimension when calculating
    // attn_mask offset in MHA
    bool mask_broadcast_num_heads = true;
    if (src_mask) {
      if (src_mask->dims()[1] == 1) {
        mask_broadcast_num_heads = true;
      } else if (src_mask->dims()[1] == num_head) {
        mask_broadcast_num_heads = false;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unknow dimension for attn_mask, the num_head(2nd) "
            "dimension is invalid, it should be 1 or num_head(%d), "
            "but got %d",
            num_head,
            src_mask->dims()[1]));
      }
    }

    phi::DenseTensor q_transpose_out, kv_transpose_out, qk_out;
    q_transpose_out.Resize({{bsz, num_head, seq_len, dim_head}});
    auto *q_transpose_out_data =
        dev_ctx.Alloc<T>(&q_transpose_out, q_transpose_out.numel() * sizeof(T));

    kv_transpose_out.Resize({{2, bsz, num_head, seq_len, dim_head}});
    auto *kv_transpose_out_data = dev_ctx.Alloc<T>(
        &kv_transpose_out, kv_transpose_out.numel() * sizeof(T));

    if (encoder_remove_padding) {
      InitValue(dev_ctx,
                q_transpose_out_data,
                q_transpose_out.numel(),
                static_cast<T>(0.));
      InitValue(dev_ctx,
                kv_transpose_out_data,
                kv_transpose_out.numel(),
                static_cast<T>(0.));
    }

    if (FLAGS_fmha_mode == "naive") {
      qk_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
      auto *qk_out_data = dev_ctx.Alloc<T>(&qk_out, qk_out.numel() * sizeof(T));
    }

    phi::DenseTensor src_mask_out;
    if (FLAGS_fmha_mode == "naive") {
      if (cache_offset > 0) {
        src_mask_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
        auto *src_mask_out_data =
            dev_ctx.Alloc<T>(&src_mask_out, src_mask_out.numel() * sizeof(T));
      }
    }

    // [2, bs, num_head, cache_seq_len + seq_len, head_dim]
    phi::DenseTensor pre_cache_kv_out;
    if (cache_offset > 0) {
      pre_cache_kv_out.Resize(
          {{2, bsz, num_head, seq_len + cache_offset, dim_head}});
      auto *pre_cache_kv_out_data = dev_ctx.Alloc<T>(
          &pre_cache_kv_out, pre_cache_kv_out.numel() * sizeof(T));
    }

    phi::DenseTensor softmax_out;
    phi::DenseTensor attn_dropout_mask_out, attn_dropout_out;
    phi::DenseTensor qktv_out, fmha_out;
    if (FLAGS_fmha_mode == "naive") {
      softmax_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
      auto *softmax_out_data =
          dev_ctx.Alloc<T>(&softmax_out, softmax_out.numel() * sizeof(T));
    }

    // unpadding_q/unpadding_k/unpadding_v: [token_num, num_head, dim_head]
    phi::DenseTensor unpadding_q, unpadding_k, unpadding_v;
    phi::DenseTensor softmax_lse, seed_offset;
    if (FLAGS_fmha_mode == "flash_attention_v2" && encoder_remove_padding) {
      unpadding_q.Resize({{token_num, num_head, dim_head}});
      if (gqa_group_size > 0) {
        unpadding_k.Resize({{token_num, gqa_group_size, dim_head}});
        unpadding_v.Resize({{token_num, gqa_group_size, dim_head}});
      } else {
        unpadding_k.Resize({{token_num, num_head, dim_head}});
        unpadding_v.Resize({{token_num, num_head, dim_head}});
      }
      cu_seqlens_k.Resize(cu_seqlens_q.dims());

      dev_ctx.Alloc<T>(&unpadding_q, unpadding_q.numel() * sizeof(T));
      dev_ctx.Alloc<T>(&unpadding_k, unpadding_k.numel() * sizeof(T));
      dev_ctx.Alloc<T>(&unpadding_v, unpadding_v.numel() * sizeof(T));
      dev_ctx.Alloc<int32_t>(&cu_seqlens_k,
                             cu_seqlens_k.numel() * sizeof(int32_t));
    }

    T *attn_dropout_mask_out_data = nullptr;
    T *attn_dropout_data_data = nullptr;

    qktv_out.Resize({{bsz, num_head, seq_len, dim_head}});
    auto *qktv_out_data =
        dev_ctx.Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
    if (remove_padding) {
      fmha_out.Resize({{token_num, num_head, dim_head}});
    } else {
      fmha_out.Resize({{bsz, seq_len, num_head, dim_head}});
    }
    auto *fmha_out_data =
        dev_ctx.Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));
    if (FLAGS_fmha_mode != "flash_attention_v2") {
      if (remove_padding && time_step) {
        InitValue(dev_ctx, fmha_out_data, fmha_out.numel(), static_cast<T>(0.));
      }
    }

    // 4. out_linear
    auto out_linear_weights = ctx.MultiInput<phi::DenseTensor>("OutLinearW");
    auto out_linear_biases = ctx.MultiInput<phi::DenseTensor>("OutLinearBias");
    int ring_id = ctx.Attr<int>("ring_id");
    auto *custom_comm = GetCustomNCCLComm(dev_ctx, ring_id);
    // (transA, transB, compute_bias) = (false, false, false)

    auto out_linear_compute = GEMMHelper<T>(
        dev_ctx, token_num, dim_embed, hidden_size, "None", false);

    // 5. ln(residual + bias)
    auto ffn_ln_scales = ctx.MultiInput<phi::DenseTensor>("FFNLnScale");
    auto ffn_ln_biases = ctx.MultiInput<phi::DenseTensor>("FFNLnBias");
    phi::DenseTensor bias_dropout_residual_out, dropout_mask_out;
    T *bias_dropout_residual_out_data = nullptr;
    if (pre_layer_norm) {
      bias_dropout_residual_out.Resize({{token_num, dim_embed}});
      bias_dropout_residual_out_data =
          dev_ctx.Alloc<T>(&bias_dropout_residual_out,
                           bias_dropout_residual_out.numel() * sizeof(T));
    }
    uint8_t *dropout_mask_out_data = nullptr;

    // 6. ffn matmul1
    auto ffn1_weights = ctx.MultiInput<phi::DenseTensor>("FFN1Weight");
    auto ffn1_biases = ctx.MultiInput<phi::DenseTensor>("FFN1Bias");
    auto ffn1_weight_dim = ffn1_weights[0]->dims();
    // if quant weight,
    // matmul weight is transposed
    int dim_ffn = ffn1_weight_dim[1];
    FFNHelper<T> ffn1_helper(
        dev_ctx, act_method, token_num, dim_ffn, dim_embed, "None");

    phi::DenseTensor ffn1_out;
    ffn1_out.Resize({{token_num, dim_ffn}});
    auto *ffn1_out_data =
        dev_ctx.Alloc<T>(&ffn1_out, ffn1_out.numel() * sizeof(T));

    // Note(Zhengzekang): It is no need when using FP16 matmul.
    phi::DenseTensor mixgemm_workspace;
    char *mixgemm_workspace_data = nullptr;

    // 7. ffn act + bias
    DropoutParam ffn1_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutHelper<T, int8_t> fused_act_dropout_helper(
        dev_ctx, token_num, dim_ffn, ffn1_dropout_param);
    phi::DenseTensor ffn1_dropout_out, ffn1_dropout_mask;
    int tmp_dim_ffn = dim_ffn;
    if (use_glu) tmp_dim_ffn /= 2;
    int8_t *ffn1_dropout_mask_data = nullptr;
    ffn1_dropout_out.Resize({{token_num, tmp_dim_ffn}});
    auto *ffn1_dropout_out_data = dev_ctx.Alloc<T>(
        &ffn1_dropout_out, ffn1_dropout_out.numel() * sizeof(T));

    // 8. ffn2 matmul
    auto ffn2_weights = ctx.MultiInput<phi::DenseTensor>("FFN2Weight");
    auto ffn2_biases = ctx.MultiInput<phi::DenseTensor>("FFN2Bias");
    auto ffn2_linear_compute = GEMMHelper<T>(
        dev_ctx, token_num, dim_embed, tmp_dim_ffn, "None", false);

    // 9. ffn2 residual bias
    DropoutParam ffn2_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> ffn2_fused_dropout_helper(
        dev_ctx,
        token_num,
        dim_embed,
        ffn2_dropout_param,
        epsilon,
        residual_alpha);

    phi::DenseTensor tmp_out, tmp_out_rm_padding;
    tmp_out.Resize({{token_num, dim_embed}});
    if (encoder_remove_padding) {
      tmp_out_rm_padding.Resize({{token_num, dim_embed}});
      auto *tmp_out_rm_padding_data = dev_ctx.Alloc<T>(
          &tmp_out_rm_padding, tmp_out_rm_padding.numel() * sizeof(T));
    }
    auto *tmp_out_data =
        dev_ctx.Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

    const T *x_data;
    if (encoder_remove_padding) {
      x_data = x_remove_padding.data<T>();
    } else {
      x_data = input_x->data<T>();
    }
    phi::DenseTensor *buf0 = nullptr;
    phi::DenseTensor *buf1 = nullptr;

    // step0:  x   --> buf1
    // step1: buf1 --> buf0
    // step2: buf0 --> buf1
    int layers = qkv_weights.size();
    if (encoder_remove_padding) {
      // In the case of variable lengths, the padding needs to be rebuilt
      // eventually. So buf0 and buf1 do not need to be changed according to the
      // pre_layer_norm and the number of layers.
      buf0 = &tmp_out;
      buf1 = &tmp_out_rm_padding;
    } else {
      if (pre_layer_norm) {
        if (layers & 1) {
          // odd, set buf1 as out
          buf0 = &tmp_out;
          buf1 = out;
        } else {
          // even, set buf0 as out
          buf0 = out;
          buf1 = &tmp_out;
        }
      } else {
        buf0 = &tmp_out;
        buf1 = out;
      }
    }

    VLOG(1) << "input_x->" << input_x->dims();

    VLOG(1) << "input_x " << *input_x;

    for (int i = 0; i < layers; ++i) {
      // step1. layer_norm
      if (i == 0 && pre_layer_norm) {
        norm_helper.Norm(x_data,
                         ln_scales[i],
                         compute_ln_bias ? ln_biases[i] : nullptr, /*norm_bias*/
                         &ln_mean,                                 /*mean*/
                         &ln_var,                                  /*var*/
                         buf1);
      }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(2) << "step1";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "ln1_out:" << *buf1;
#endif
#endif

      // step2. qkv
      // NOTE: In decoder stage, bias is fused in fmha. In encoder stage, bias
      // is fused in QKVBiasAddTransposeSplit
      const phi::DenseTensor *qkv_bias =
          qkv_biases.size() > 0 ? qkv_biases[i] : nullptr;
      if (!pre_layer_norm && i == 0) {
        const phi::DenseTensor *tmp_input_x =
            (encoder_remove_padding) ? &x_remove_padding : input_x;
        VLOG(5) << "Doing !pre_layer_norm&&i==0, qkv gemm, mnk:" << token_num
                << ", " << output_size << ", " << input_size;
        qkv_compute.Compute(tmp_input_x,
                            qkv_weights[i],
                            /*weight_scale*/ nullptr,
                            /*bias*/ nullptr,
                            &mixgemm_workspace,
                            &qkv_out);
      } else {
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
        VLOG(2) << "qkv_weights:" << *(qkv_weights[i]);
#endif
#endif
        VLOG(5) << "Doing qkv gemm, mnk:" << token_num << ", " << output_size
                << ", " << input_size;
        qkv_compute.Compute(buf1,
                            qkv_weights[i],
                            /*weight_scale*/ nullptr,
                            /*bias*/ nullptr,
                            &mixgemm_workspace,
                            &qkv_out);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(2) << "step2";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "qkv_out:" << qkv_out;
#endif
#endif

      // 2. cache kv
      auto write_cache_kv_helper = WriteCacheKVHelper<T>(
          dev_ctx, quant_round_type, quant_max_bound, quant_min_bound);

      // step3. fmha
      const phi::DenseTensor *cache_kv =
          cache_kvs.size() > 0 ? cache_kvs[i] : nullptr;
      phi::DenseTensor *cache_kv_out = cache_kv ? cache_kv_outs[i] : nullptr;
      int cache_bsz = 0;
      if (cache_kv) {
        cache_bsz = cache_kv->dims()[1];
      }

      if (time_step) {  // generation decoder stage
        // [2, batch_size, num_head, max_seq_len, head_size]
        int max_seq_len = cache_kv->dims()[3];
        VLOG(1) << "sequence_lengths " << *sequence_lengths;
        fmha<T>(dev_ctx,
                qkv_out,
                *qkv_bias,
                src_mask,
                nullptr,
                sequence_lengths,
                rotary_tensor,
                beam_cache_offset,
                cache_kv_out,
                &fmha_out,
                bsz,
                cache_bsz,
                seq_len,
                max_seq_len,
                num_head,
                dim_head,
                src_mask->dims()[3] - 1,
                rotary_emb_dims,
                1. / sqrt(dim_head),
                mask_broadcast_num_heads,
                compute_bias,
                use_neox_rotary_style,
                nullptr,  // qkv_out_scale
                nullptr,  // out_linear_shift
                nullptr,  // out_smooth_shift
                (do_cachekv_quant) ? cache_k_scale[i] : -1.0,
                (do_cachekv_quant) ? cache_v_scale[i] : -1.0,
                (do_cachekv_quant) ? cache_k_out_scale[i] : -1.0,
                (do_cachekv_quant) ? cache_v_out_scale[i] : -1.0,
                -1,       // quant_fmha_out_scale
                1,        // quant_round_type
                127.0f,   // quant_max_bound
                -127.0f,  // quant_min_bound
                gqa_group_size);
      } else if (cache_kv_out) {  // generation context stage
        if (FLAGS_fmha_mode == "flash_attention_v2" && encoder_remove_padding) {
          if (rotary_emb_dims != 0) {
            VLOG(1) << "rotary_tensor " << *rotary_tensor;
            if (gqa_group_size <= 0) {
              rotary_qk_variable(dev_ctx,
                                 qkv_out_data,
                                 qkv_out_data,
                                 qkv_bias->data<T>(),
                                 rotary_tensor->data<float>(),
                                 padding_offset_data,
                                 sequence_lengths->data<int>(),
                                 token_num,
                                 num_head,
                                 seq_len,
                                 rotary_tensor->dims()[3],
                                 dim_head,
                                 rotary_tensor->dims()[1]);
            } else {
              gqa_rotary_qk_variable(dev_ctx,
                                     qkv_out_data,
                                     qkv_out_data,
                                     qkv_bias->data<T>(),
                                     rotary_tensor->data<float>(),
                                     padding_offset_data,
                                     sequence_lengths->data<int>(),
                                     token_num,
                                     num_head,
                                     seq_len,
                                     rotary_tensor->dims()[3],
                                     dim_head,
                                     gqa_group_size,
                                     rotary_tensor->dims()[1]);
            }
          }
          if (gqa_group_size <= 0) {
            qkv_transpose_split<T>(dev_ctx,
                                   unpadding_q.data<T>(),
                                   unpadding_k.data<T>(),
                                   unpadding_v.data<T>(),
                                   qkv_out_data,
                                   padding_offset_data,
                                   sequence_lengths->data<int>(),
                                   token_num,
                                   bsz,
                                   num_head,
                                   seq_len,
                                   dim_head);
          } else {
            gqa_qkv_transpose_split<T>(dev_ctx,
                                       unpadding_q.data<T>(),
                                       unpadding_k.data<T>(),
                                       unpadding_v.data<T>(),
                                       qkv_out_data,
                                       padding_offset_data,
                                       sequence_lengths->data<int>(),
                                       token_num,
                                       bsz,
                                       num_head,
                                       seq_len,
                                       dim_head,
                                       gqa_group_size);
          }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
          VLOG(2) << "TransposeSplit";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
          VLOG(2) << "unpadding_q:" << unpadding_q;
          VLOG(2) << "unpadding_k:" << unpadding_k;
          VLOG(2) << "unpadding_v:" << unpadding_v;
#endif
#endif
          phi::Copy(dev_ctx,
                    cu_seqlens_q,
                    cu_seqlens_k.place(),
                    false,
                    &cu_seqlens_k);
          VLOG(2) << "cu_seqlens_q:" << cu_seqlens_q;
          VLOG(2) << "cu_seqlens_k:" << cu_seqlens_k;

          // fmha_out[token_num, num_head, dim_head]
          phi::FlashAttnUnpaddedKernel<T>(
              dev_ctx,
              unpadding_q,
              unpadding_k,
              unpadding_v,
              cu_seqlens_q,
              cu_seqlens_k,
              none /*fixed_seed_offset*/,
              none /*attn_mask*/,
              seq_len,
              seq_len,
              1.0f / sqrt(static_cast<float>(dim_head)),
              0.0,
              true /*causal*/,
              false,
              true /* is_test*/,
              "" /*rng_name*/,
              &fmha_out,
              &softmax_out,
              &softmax_lse,
              &seed_offset);
          // Note(@RichardWooSJTU): gqa_write_cachekv do not support pre_cache
          // and cache quantization
          gqa_write_cachekv<T>(dev_ctx,
                               cache_kv_out,
                               unpadding_k,
                               unpadding_v,
                               padding_offset_tensor,
                               *sequence_lengths,
                               seq_len);
        } else {
          if (gqa_group_size > 0) {
            PADDLE_THROW(phi::errors::Unimplemented(
                "GQA is just supported when FLAGS_fmha_mode is "
                "flash_attention_v2"));
          }
          const phi::DenseTensor *pre_cache_kv_tensor =
              pre_caches.size() > 0 ? pre_caches[i] : nullptr;
          phi::DenseTensor *pre_cache_kv_out_tmp =
              cache_offset > 0 ? &pre_cache_kv_out : nullptr;
          phi::DenseTensor *src_mask_tmp =
              cache_offset > 0 ? &src_mask_out : nullptr;
          const int *sequence_lengths_data =
              encoder_remove_padding ? sequence_lengths->data<int>() : nullptr;
          qkv_bias_add_transpose_split<T>(
              dev_ctx,
              q_transpose_out_data,
              kv_transpose_out_data,
              qkv_out_data,
              qkv_bias ? qkv_bias->data<T>() : nullptr,
              padding_offset_data,
              token_num,
              bsz,
              num_head,
              seq_len,
              dim_head,
              compute_bias);

          // q_transpose_out_data [bs, head_num, seq_len, dim_head]
          // kv_transpose_out_data [2， bs, head_num, seq_len, dim_head]
          if (rotary_emb_dims != 0) {
            auto *rotary_emb_data = rotary_tensor->data<float>();
            const int *sequence_lengths_data =
                encoder_remove_padding ? sequence_lengths->data<int>()
                                       : nullptr;
            rotary_qk(dev_ctx,
                      q_transpose_out_data,
                      kv_transpose_out_data,
                      q_transpose_out_data,
                      kv_transpose_out_data,
                      rotary_emb_data,
                      sequence_lengths_data,
                      rotary_emb_dims,
                      rotary_tensor->dims()[1],
                      bsz,
                      num_head,
                      seq_len,
                      dim_head,
                      use_neox_rotary_style);
          }

          phi::DenseTensor *tmp_padding_offset_tensor =
              encoder_remove_padding ? &padding_offset_tensor : nullptr;
          fmha_compute.Compute(pre_cache_kv_tensor,
                               src_mask,
                               tmp_padding_offset_tensor,
                               sequence_lengths,
                               &q_transpose_out,
                               &kv_transpose_out,
                               pre_cache_kv_out_tmp,
                               &qk_out,
                               src_mask_tmp,
                               &softmax_out,
                               &attn_dropout_mask_out,
                               &attn_dropout_out,
                               &qktv_out,
                               &fmha_out,
                               token_num,
                               mask_broadcast_num_heads);
          write_cache_kv_helper.Compute(
              &pre_cache_kv_out,
              cache_kv_out,       // int8_t
              &kv_transpose_out,  // T
              sequence_lengths_data,
              cache_bsz,
              bsz,
              num_head,
              seq_len,
              dim_head,
              cache_offset,
              (do_cachekv_quant) ? cache_k_scale[i] : -1.0,
              (do_cachekv_quant) ? cache_v_scale[i] : -1.0);
        }

      } else {  // not generation
        // TODO(wangxi): can remove dropout in inference
        qkv_bias_add_transpose_split<T>(
            dev_ctx,
            q_transpose_out_data,
            kv_transpose_out_data,
            qkv_out_data,
            qkv_bias ? qkv_bias->data<T>() : nullptr,
            padding_offset_data,
            token_num,
            bsz,
            num_head,
            seq_len,
            dim_head,
            compute_bias);

        // q_transpose_out_data [bs, head_num, seq_len, dim_head]
        // kv_transpose_out_data [2， bs, head_num, seq_len, dim_head]
        if (rotary_emb_dims != 0) {
          auto *rotary_emb_data = rotary_tensor->data<float>();
          const int *sequence_lengths_data =
              encoder_remove_padding ? sequence_lengths->data<int>() : nullptr;
          rotary_qk(dev_ctx,
                    q_transpose_out_data,
                    kv_transpose_out_data,
                    q_transpose_out_data,
                    kv_transpose_out_data,
                    rotary_emb_data,
                    sequence_lengths_data,
                    rotary_emb_dims,
                    rotary_tensor->dims()[1],
                    bsz,
                    num_head,
                    seq_len,
                    dim_head,
                    use_neox_rotary_style);
        }
        phi::DenseTensor *tmp_padding_offset_tensor =
            encoder_remove_padding ? &padding_offset_tensor : nullptr;

        if (FLAGS_fmha_mode == "flash_attention_v2" && encoder_remove_padding) {
          TransposeSplit<T>(dev_ctx,
                            unpadding_q.data<T>(),
                            unpadding_k.data<T>(),
                            unpadding_v.data<T>(),
                            q_transpose_out.data<T>(),
                            kv_transpose_out.data<T>(),
                            padding_offset_data,
                            sequence_lengths->data<int>(),
                            token_num,
                            bsz,
                            num_head,
                            seq_len,
                            dim_head);
          phi::Copy(dev_ctx,
                    cu_seqlens_q,
                    cu_seqlens_k.place(),
                    false,
                    &cu_seqlens_k);

          // fmha_out[token_num, num_head, dim_head]
          phi::FlashAttnUnpaddedKernel<T>(
              dev_ctx,
              unpadding_q,
              unpadding_k,
              unpadding_v,
              cu_seqlens_q,
              cu_seqlens_k,
              none /*fixed_seed_offset*/,
              none /*attn_mask*/,
              seq_len,
              seq_len,
              1.0f / sqrt(static_cast<float>(dim_head)),
              0.0,
              true /*causal*/,
              false,
              true /* is_test*/,
              "" /*rng_name*/,
              &fmha_out,
              &softmax_out,
              &softmax_lse,
              &seed_offset);
        } else {
          fmha_compute.Compute(cache_kv,
                               src_mask,
                               tmp_padding_offset_tensor,
                               sequence_lengths,
                               &q_transpose_out,
                               &kv_transpose_out,
                               cache_kv_out,
                               &qk_out,
                               nullptr,
                               &softmax_out,
                               &attn_dropout_mask_out,
                               &attn_dropout_out,
                               &qktv_out,
                               &fmha_out,
                               token_num,
                               mask_broadcast_num_heads);
        }
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(2) << "step3";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "fmha_out:" << fmha_out;
#endif
#endif
      VLOG(5) << "Doing out_linear gemm, mnk:" << token_num << ", " << dim_embed
              << ", " << hidden_size;
      if (pre_layer_norm) {
        if (custom_comm) {
          custom_comm->SwapInput(buf1);
        }

        out_linear_compute.Compute(&fmha_out,
                                   out_linear_weights[i],
                                   /*weight_scale*/ nullptr,
                                   /*bias*/ nullptr,
                                   &mixgemm_workspace,
                                   buf1);

        if (custom_comm) {
          *buf1 = custom_comm->AllReduce();
        } else {
          VLOG(1) << "ALLREDUCE " << buf1->numel();
          AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
        }
      } else {
        if (custom_comm) {
          custom_comm->SwapInput(buf0);
        }
        out_linear_compute.Compute(&fmha_out,
                                   out_linear_weights[i],
                                   /*weight_scale*/ nullptr,
                                   /*bias*/ nullptr,
                                   &mixgemm_workspace,
                                   buf0);

        if (custom_comm) {
          *buf0 = custom_comm->AllReduce();
        } else {
          AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
        }
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(2) << "step4";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "out_linear_out:" << *buf1;
#endif
#endif

      // step5. ln(residual + dropout(input + bias))
      if (pre_layer_norm) {
        norm_helper.NormResidualBias(
            buf1->data<T>(),
            x_data,
            compute_bias ? out_linear_biases[i] : nullptr, /*skip_bias*/
            ffn_ln_scales[i],
            compute_ln_bias ? ffn_ln_biases[i] : nullptr, /*norm_bias*/
            &ln_mean,                                     /*mean*/
            &ln_var,                                      /*var*/
            &bias_dropout_residual_out,
            buf1);
      } else {
        auto *residual_data = (i == 0 ? x_data : buf1->data<T>());
        norm_helper.NormResidualBias(
            buf0->data<T>(),
            residual_data,
            compute_bias ? out_linear_biases[i] : nullptr, /*skip_bias*/
            ln_scales[i],
            compute_ln_bias ? ln_biases[i] : nullptr, /*norm_bias*/
            &ln_mean,                                 /*mean*/
            &ln_var,                                  /*var*/
            buf0,
            buf1);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(2) << "step5";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "ffn1_input:" << *buf1;
#endif
#endif
      // step6. ffn matmul1
      ffn1_helper.Compute(buf1,
                          ffn1_weights[i],
                          /*weight_scale*/ nullptr,
                          compute_bias ? ffn1_biases[i] : nullptr,
                          &mixgemm_workspace,
                          &ffn1_out,
                          &ffn1_dropout_out);
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(2) << "step6";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "ffn1_output:" << ffn1_out;
#endif
#endif

      // step7. ffn2 matmul
      if (pre_layer_norm) {
        if (custom_comm) {
          custom_comm->SwapInput(buf1);
        }
        ffn2_linear_compute.Compute(&ffn1_dropout_out,
                                    ffn2_weights[i],
                                    nullptr,
                                    /*bias*/ nullptr,
                                    &mixgemm_workspace,
                                    buf1);
      } else {
        if (custom_comm) {
          custom_comm->SwapInput(buf0);
        }
        ffn2_linear_compute.Compute(&ffn1_dropout_out,
                                    ffn2_weights[i],
                                    nullptr,
                                    /*bias*/ nullptr,
                                    &mixgemm_workspace,
                                    buf0);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(2) << "step8.0";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      if (pre_layer_norm) {
        VLOG(2) << "ffn2_out, buf1:" << *buf1;
      } else {
        VLOG(2) << "ffn2_out, buf0:" << *buf0;
      }
#endif
#endif

      if (pre_layer_norm) {
        VLOG(4) << "MPAllReduce 4: " << buf1->numel();
        if (custom_comm) {
          *buf1 = custom_comm->AllReduce();
        } else {
          VLOG(1) << "ALLREDUCE ffn" << buf1->numel();
          AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
        }
      } else {
        VLOG(4) << "MPAllReduce 4: " << buf0->numel();
        if (custom_comm) {
          *buf0 = custom_comm->AllReduce();
        } else {
          AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
        }
      }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(2) << "step8.1";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      if (pre_layer_norm) {
        VLOG(2) << "ffn2_out_rd:" << *buf1;
      } else {
        VLOG(2) << "ffn2_out_rd:" << *buf0;
      }
#endif
#endif

      // step8. residual bias
      // TODO(wangxi): remove dropout mask in inference
      if (pre_layer_norm) {
        // TODO(wangxi): remove dropout mask in inference
        if (i < layers - 1) {
          norm_helper.NormResidualBias(
              buf1->data<T>(),
              bias_dropout_residual_out_data,
              compute_bias ? ffn2_biases[i] : nullptr, /*skip_bias*/
              ln_scales[i + 1],
              compute_ln_bias ? ln_biases[i + 1] : nullptr, /*norm_bias*/
              &ln_mean,                                     /*mean*/
              &ln_var,                                      /*var*/
              buf1,
              buf0);
        } else {
          ffn2_fused_dropout_helper.ResidualDropoutBias(
              dev_ctx,
              buf1->data<T>(),
              bias_dropout_residual_out_data,
              compute_bias ? ffn2_biases[i]->data<T>() : nullptr,
              buf1->data<T>(),
              dropout_mask_out_data);
        }
      } else {
        norm_helper.NormResidualBias(
            buf0->data<T>(),
            buf1->data<T>(),
            compute_bias ? ffn2_biases[i] : nullptr, /*skip_bias*/
            ffn_ln_scales[i],
            compute_ln_bias ? ffn_ln_biases[i] : nullptr, /*norm_bias*/
            &ln_mean,                                     /*mean*/
            &ln_var,                                      /*var*/
            buf0,
            buf1);
      }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(2) << "step9";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "residual_out:" << *buf1;
#endif
#endif
      if (pre_layer_norm) {
        x_data = buf1->data<T>();
        std::swap(buf0, buf1);
      }
    }

    if (encoder_remove_padding) {
      if (pre_layer_norm) {
        InvokeRebuildPadding(dev_ctx,
                             from_data,
                             buf0->data<T>(),
                             padding_offset_data,
                             token_num,
                             dim_embed);
      } else {
        InvokeRebuildPadding(dev_ctx,
                             from_data,
                             buf1->data<T>(),
                             padding_offset_data,
                             token_num,
                             dim_embed);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#if CUDA_VERSION >= 11000
PD_REGISTER_STRUCT_KERNEL(fused_multi_transformer,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedMultiTransformerOpKernel,
                          float,
                          plat::float16,
                          plat::bfloat16) {}
#else
PD_REGISTER_STRUCT_KERNEL(fused_multi_transformer,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedMultiTransformerOpKernel,
                          float,
                          plat::float16) {}
#endif
