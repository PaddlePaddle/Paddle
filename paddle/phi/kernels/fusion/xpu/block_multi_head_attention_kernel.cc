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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "xpu/xdnn.h"

namespace phi {
namespace fusion {

template <typename Context>
int GetMaxLen(const Context& dev_ctx,
              const phi::DenseTensor& seq_lens_tensor,
              phi::DenseTensor* max_len_tensor,
              const int batch_size) {
  int max_len_cpu = 0;
  int r = baidu::xpu::api::reduce_max<int>(dev_ctx.x_context(),
                                           seq_lens_tensor.data<int>(),
                                           max_len_tensor->data<int>(),
                                           {batch_size},
                                           {0});
  PADDLE_ENFORCE_EQ(
      r, 0, common::errors::Fatal("baidu::xpu::api::reduce_max failed."));
  xpu_wait(dev_ctx.x_context()->xpu_stream);
  r = xpu_memcpy(&max_len_cpu,
                 max_len_tensor->data<int>(),
                 sizeof(int),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  PADDLE_ENFORCE_EQ(r, 0, common::errors::Fatal("xpu_memcpy failed."));
  return max_len_cpu;
}

template <typename T, typename Context>
void qkv_split_rope_kernel(
    const Context& xpu_ctx,
    const DenseTensor& qkv_input,
    const DenseTensor& rotary_emb,
    const DenseTensor& seq_lens,
    const baidu::xpu::api::VectorParam<int32_t>& lods,
    const baidu::xpu::api::VectorParam<int32_t>& pos_emb_offset,
    int bsz,
    int max_seq_len,
    int token_num,
    int num_head,
    int dim_head,
    DenseTensor* q_out,
    DenseTensor* k_out,
    DenseTensor* v_out) {
  xpu::ctx_guard RAII_GUARD(xpu_ctx.x_context());
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto q_data = reinterpret_cast<XPUType*>(q_out->data<T>());
  auto k_data = reinterpret_cast<XPUType*>(k_out->data<T>());
  auto v_data = reinterpret_cast<XPUType*>(v_out->data<T>());
  int r = baidu::xpu::api::split<XPUType>(
      xpu_ctx.x_context(),
      reinterpret_cast<const XPUType*>(qkv_input.data<T>()),
      {q_data, k_data, v_data},
      {token_num, 3, num_head * dim_head},
      {1, 1, 1},
      1);
  const_cast<DenseTensor*>(&qkv_input)->clear();
  PADDLE_ENFORCE_EQ(
      r, 0, common::errors::Fatal("baidu::xpu::api::split failed."));
  r = baidu::xpu::api::vsl_rotary_neox_embedding<XPUType, float, int32_t>(
      xpu_ctx.x_context(),
      q_data,
      k_data,
      rotary_emb.data<float>(),
      q_data,
      k_data,
      lods,
      1,
      max_seq_len,
      num_head,
      dim_head,
      "BLHD",
      pos_emb_offset,
      "NORMAL",
      -1);
  PADDLE_ENFORCE_EQ(r,
                    0,
                    common::errors::Fatal(
                        "baidu::xpu::api::vsl_rotary_neox_embedding failed."));
}

template <typename T, typename Context>
void BlockMultiheadAttentionXPUKernel(
    const Context& dev_ctx,
    const DenseTensor& qkv,
    const DenseTensor& key_cache,
    const DenseTensor& value_cache,
    const DenseTensor& seq_lens_encoder,
    const DenseTensor& seq_lens_decoder,
    const DenseTensor& seq_lens_this_time,
    const DenseTensor& padding_offsets,
    const DenseTensor& cum_offsets,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const DenseTensor& block_tables,
    const DenseTensor& cache_k_per_batch_maxs,
    const DenseTensor& cache_v_per_batch_maxs,
    const paddle::optional<DenseTensor>& pre_key_cache,
    const paddle::optional<DenseTensor>& pre_value_cache,
    const paddle::optional<DenseTensor>& rope_emb,
    const paddle::optional<DenseTensor>& mask,
    const paddle::optional<DenseTensor>& tgt_mask,
    const paddle::optional<DenseTensor>& cache_k_quant_scales,
    const paddle::optional<DenseTensor>& cache_v_quant_scales,
    const paddle::optional<DenseTensor>& cache_k_dequant_scales,
    const paddle::optional<DenseTensor>& cache_v_dequant_scales,
    const paddle::optional<DenseTensor>& qkv_out_scale,
    const paddle::optional<DenseTensor>& qkv_bias,
    const paddle::optional<DenseTensor>& out_shift,
    const paddle::optional<DenseTensor>& out_smooth,
    const paddle::optional<DenseTensor>& max_enc_len_this_time,
    const paddle::optional<DenseTensor>& max_dec_len_this_time,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    const bool dynamic_cachekv_quant,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_scale,
    const std::string& compute_dtype,
    DenseTensor* fmha_out,
    DenseTensor* qkv_out,
    DenseTensor* key_cache_out,
    DenseTensor* value_cache_out) {
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  auto xpu_context = dev_ctx.x_context();

  using XPUType = typename XPUTypeTrait<T>::Type;

  phi::DenseTensor qkv_buf;
  phi::DenseTensor fmha_buf;
  VLOG(3) << "fmha_out " << fmha_out->dims();
  if (out_scale <= 0) {
    dev_ctx.template Alloc<T>(fmha_out);
    fmha_buf = *fmha_out;
  } else {
    PADDLE_THROW(common::errors::Unimplemented("Not supports out_scale > 0."));
  }
  int r = xpu::constant<XPUType>(xpu_context,
                                 reinterpret_cast<XPUType*>(fmha_buf.data<T>()),
                                 fmha_buf.numel(),
                                 0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  const auto& input_dims = qkv.dims();
  const auto& key_cache_dims = key_cache.dims();
  const int token_num = input_dims[0];
  const int num_head = key_cache_dims[1];
  const int dim_head = key_cache_dims[3];
  const int bsz = cum_offsets.dims()[0];
  const int max_block_per_seq = block_tables.dims()[1];
  VLOG(3) << "bsz: " << bsz << " token_num: " << token_num
          << " num_head: " << num_head << " dim_head: " << dim_head
          << " max_block_per_seq: " << max_block_per_seq;
  VLOG(3) << "fmha_out_dims: " << fmha_out->dims();
  bool causual = true;
  if (mask) {
    causual = false;
  }
  bool use_pre_cache = false;
  int pre_cache_length = 0;
  if (pre_key_cache) {
    PADDLE_THROW(
        common::errors::Unimplemented("Not supports pre_key_cache now."));
  }
  VLOG(3) << "token_num: " << token_num
          << " pre_cache_length: " << pre_cache_length;

  int max_dec_len_this_time_data(0);
  if (!max_dec_len_this_time) {
    phi::DenseTensor max_dec_len_tensor;
    max_dec_len_tensor.Resize({{1}});
    dev_ctx.template Alloc<int>(&max_dec_len_tensor,
                                max_dec_len_tensor.numel() * sizeof(int));
    max_dec_len_this_time_data =
        GetMaxLen(dev_ctx, seq_lens_decoder, &max_dec_len_tensor, bsz);
  } else {
    PADDLE_ENFORCE_EQ(
        max_dec_len_this_time.get().place().GetType(),
        phi::AllocationType::CPU,
        errors::InvalidArgument(
            "The place of input max_dec_len_this_time must be CPU, but got %s.",
            max_dec_len_this_time.get().place()));
    max_dec_len_this_time_data = *max_dec_len_this_time.get().data<int>();
  }
  int max_enc_len_this_time_data(0);
  if (!max_enc_len_this_time) {
    phi::DenseTensor max_enc_len_tensor;
    max_enc_len_tensor.Resize({{1}});
    dev_ctx.template Alloc<int>(&max_enc_len_tensor,
                                max_enc_len_tensor.numel() * sizeof(int));
    max_enc_len_this_time_data =
        GetMaxLen(dev_ctx, seq_lens_encoder, &max_enc_len_tensor, bsz);
  } else {
    PADDLE_ENFORCE_EQ(
        max_enc_len_this_time.get().place().GetType(),
        phi::AllocationType::CPU,
        errors::InvalidArgument(
            "The place of input max_enc_len_this_time must be CPU, but got %s.",
            max_enc_len_this_time.get().place()));
    max_enc_len_this_time_data = *max_enc_len_this_time.get().data<int>();
  }

  const int MAXPTR_N = xpu_context->max_ptr_size();
  VLOG(3) << "max_len end";
  phi::DenseTensor unpadding_q, unpadding_k, unpadding_v;
  phi::DenseTensor softmax_out, softmax_lse, seed_offset;
  phi::DenseTensor q_trans, k_trans, v_trans, qktv_out;
  if (!use_pre_cache) {
    unpadding_q.Resize({{token_num, num_head, dim_head}});
    unpadding_k.Resize({{token_num, num_head, dim_head}});
    unpadding_v.Resize({{token_num, num_head, dim_head}});

    dev_ctx.template Alloc<T>(&unpadding_q, unpadding_q.numel() * sizeof(T));
    dev_ctx.template Alloc<T>(&unpadding_k, unpadding_k.numel() * sizeof(T));
    dev_ctx.template Alloc<T>(&unpadding_v, unpadding_v.numel() * sizeof(T));
  } else {
    PADDLE_THROW(
        common::errors::Unimplemented("Not supports pre_key_cache now."));
  }
  VLOG(3) << "encoder";
  VLOG(3) << "max_enc_len_this_time_data: " << max_enc_len_this_time_data;
  if (qkv_out_scale) {
    PADDLE_THROW(
        common::errors::Unimplemented("Not supports qkv_out_scale now."));
  } else {
    VLOG(1) << "qkv_out_scale is none";
    qkv_buf = qkv;
  }
  if (qkv_bias) {
    PADDLE_THROW(common::errors::Unimplemented("Not supports qkv_bias now."));
  }
  std::vector<int> lods_cpu(bsz + 1, 0);
  xpu_wait(xpu_context->xpu_stream);
  xpu_memcpy(lods_cpu.data() + 1,
             seq_lens_this_time.data<int>(),
             sizeof(int32_t) * bsz,
             XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  for (int i = 1; i < bsz + 1; i++) {
    lods_cpu[i] += lods_cpu[i - 1];
  }
  using XPUType = typename XPUTypeTrait<T>::Type;
  baidu::xpu::api::VectorParam<int32_t> lods =
      baidu::xpu::api::VectorParam<int32_t>{lods_cpu.data(), bsz + 1, nullptr}
          .to_xpu(RAII_GUARD);
  float* p_batch_max_ptrs = RAII_GUARD.alloc_l3_or_gm<float>(bsz);

  if (!rope_emb || !use_neox_style) {
    PADDLE_THROW(common::errors::Unimplemented(
        "only supports use_neox_style rope_emb now."));
  }
  if (max_enc_len_this_time_data > 0) {
    // const int* sequence_lengths_data = seq_lens_encoder.data<int>();
    xpu::VectorParam<int32_t> pos_emb_offset =
        xpu::VectorParam<int32_t>{nullptr, 0, nullptr};
    qkv_split_rope_kernel<T, Context>(dev_ctx,
                                      qkv,
                                      rope_emb.get(),
                                      seq_lens_encoder,
                                      lods,
                                      pos_emb_offset,
                                      bsz,
                                      rope_emb.get().dims()[2],
                                      token_num,
                                      num_head,
                                      dim_head,
                                      &unpadding_q,
                                      &unpadding_k,
                                      &unpadding_v);

    VLOG(3) << "rope end";
    VLOG(3) << "causual: " << causual;
    if (!use_pre_cache) {
      phi::FlashAttnUnpaddedKernel<T>(dev_ctx,
                                      unpadding_q,
                                      unpadding_k,
                                      unpadding_v,
                                      cu_seqlens_q,
                                      cu_seqlens_k,
                                      paddle::none /*fixed_seed_offset*/,
                                      causual ? paddle::none : mask,
                                      max_enc_len_this_time_data,
                                      max_enc_len_this_time_data,
                                      1.0f / sqrt(static_cast<float>(dim_head)),
                                      0.0,
                                      causual,
                                      false,
                                      true /* is_test*/,
                                      "" /*rng_name*/,
                                      &fmha_buf,
                                      &softmax_out,
                                      &softmax_lse,
                                      &seed_offset);
    } else {
      PADDLE_THROW(
          common::errors::Unimplemented("Not supports use_pre_cache now."));
    }
    VLOG(3) << "flash end";
    if (cache_k_quant_scales && dynamic_cachekv_quant) {
      PADDLE_THROW(common::errors::Unimplemented("Not supports quant now."));
    } else {
      std::vector<int32_t> start_token_ctx(bsz, 0);
      xpu::VectorParam<int32_t> start_token_ctx_VP =
          xpu::VectorParam<int32_t>{
              start_token_ctx.data(),
              static_cast<int64_t>(start_token_ctx.size()),
              nullptr}
              .to_xpu(RAII_GUARD);

      std::vector<int32_t> ordered_index_ctx(bsz, 0);
      std::iota(ordered_index_ctx.begin(), ordered_index_ctx.end(), 0);
      xpu::VectorParam<int32_t> ordered_index_ctx_VP =
          xpu::VectorParam<int32_t>{
              ordered_index_ctx.data(), static_cast<int64_t>(bsz), nullptr}
              .to_xpu(RAII_GUARD);
      int ret = xpu::reshape_cached_kv<XPUType, XPUType, int32_t>(
          xpu_context,
          reinterpret_cast<const XPUType*>(unpadding_k.data<T>()),
          reinterpret_cast<XPUType*>(const_cast<T*>(key_cache.data<T>())),
          block_tables.data<int>(),
          lods,
          start_token_ctx_VP,
          ordered_index_ctx_VP,
          bsz,
          num_head,
          dim_head,
          bsz,
          block_size,
          max_block_per_seq,
          "BLHD",
          "HLD");
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reshape_cached_kv");
      ret = xpu::batch_findmax<XPUType>(
          xpu_context,
          reinterpret_cast<XPUType*>(const_cast<T*>(key_cache.data<T>())),
          token_num,
          num_head * dim_head,
          bsz,
          lods.xpu,
          p_batch_max_ptrs);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "batch_findmax");
      ret = xpu::copy2d<float>(
          xpu_context,
          p_batch_max_ptrs,
          const_cast<float*>(cache_k_per_batch_maxs.data<float>()),
          bsz,
          1,
          MAXPTR_N,
          1);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy2d");
      ret = xpu::reshape_cached_kv<XPUType, XPUType, int32_t>(
          xpu_context,
          reinterpret_cast<const XPUType*>(unpadding_v.data<T>()),
          reinterpret_cast<XPUType*>(const_cast<T*>(value_cache.data<T>())),
          block_tables.data<int>(),
          lods,
          start_token_ctx_VP,
          ordered_index_ctx_VP,
          bsz,
          num_head,
          dim_head,
          bsz,
          block_size,
          max_block_per_seq,
          "BLHD",
          "HLD");
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reshape_cached_kv");
      ret = xpu::batch_findmax<XPUType>(
          xpu_context,
          reinterpret_cast<XPUType*>(const_cast<T*>(value_cache.data<T>())),
          token_num,
          num_head * dim_head,
          bsz,
          lods.xpu,
          p_batch_max_ptrs);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "batch_findmax");
      ret = xpu::copy2d<float>(
          xpu_context,
          p_batch_max_ptrs,
          const_cast<float*>(cache_v_per_batch_maxs.data<float>()),
          bsz,
          1,
          MAXPTR_N,
          1);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy2d");
    }
    VLOG(3) << "cache end";
  }
  VLOG(3) << "encoder done";
  VLOG(3) << "max_dec_len_this_time_data: " << max_dec_len_this_time_data;

  if (max_dec_len_this_time_data > 0) {
    int cachekv_quant_mode = 0;
    if (cache_k_quant_scales || cachekv_quant_mode) {
      PADDLE_THROW(common::errors::Unimplemented(
          "Not supports cache_k_quant_scales or cachekv_quant_mode now."));
    }
    std::vector<int> lods_decoder_cpu(bsz + 1, 0);
    xpu_wait(xpu_context->xpu_stream);
    xpu_memcpy(lods_decoder_cpu.data() + 1,
               seq_lens_decoder.data<int>(),
               sizeof(int32_t) * bsz,
               XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    for (int i = 1; i < bsz + 1; i++) {
      lods_decoder_cpu[i] += lods_decoder_cpu[i - 1];
    }
    std::vector<int32_t> kv_seq_lod_dec(bsz + 1, 0);
    std::iota(kv_seq_lod_dec.begin(), kv_seq_lod_dec.end(), 0);
    xpu::VectorParam<int32_t> kv_seq_lod_dec_VP =
        xpu::VectorParam<int32_t>{kv_seq_lod_dec.data(),
                                  static_cast<int64_t>(kv_seq_lod_dec.size()),
                                  nullptr}
            .to_xpu(RAII_GUARD);
    std::vector<int32_t> start_token_ctx(bsz, 0);
    for (int i = 0; i < bsz; i++) {
      start_token_ctx[i] = lods_decoder_cpu[i + 1] - lods_decoder_cpu[i];
    }
    xpu::VectorParam<int32_t> start_token_ctx_VP =
        xpu::VectorParam<int32_t>{start_token_ctx.data(),
                                  static_cast<int64_t>(start_token_ctx.size()),
                                  nullptr}
            .to_xpu(RAII_GUARD);
    qkv_split_rope_kernel<T, Context>(dev_ctx,
                                      qkv,
                                      rope_emb.get(),
                                      seq_lens_encoder,
                                      lods,
                                      start_token_ctx_VP,
                                      bsz,
                                      rope_emb.get().dims()[2],
                                      token_num,
                                      num_head,
                                      dim_head,
                                      &unpadding_q,
                                      &unpadding_k,
                                      &unpadding_v);

    std::vector<int32_t> ordered_index_ctx(bsz, 0);
    std::iota(ordered_index_ctx.begin(), ordered_index_ctx.end(), 0);
    xpu::VectorParam<int32_t> ordered_index_ctx_VP =
        xpu::VectorParam<int32_t>{
            ordered_index_ctx.data(), static_cast<int64_t>(bsz), nullptr}
            .to_xpu(RAII_GUARD);

    float* p_batch_max_ptrs_fill =
        RAII_GUARD.alloc_l3_or_gm<float>(bsz * MAXPTR_N);
    int ret = xpu::constant<float>(
        xpu_context, p_batch_max_ptrs_fill, bsz * MAXPTR_N, 0.0);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "constant");
    float* p_cache_k_max_data = RAII_GUARD.alloc_l3_or_gm<float>(MAXPTR_N);
    float* p_cache_v_max_data = RAII_GUARD.alloc_l3_or_gm<float>(MAXPTR_N);
    ret = xpu::reshape_cached_kv<XPUType, XPUType, int32_t>(
        xpu_context,
        reinterpret_cast<const XPUType*>(unpadding_k.data<T>()),
        reinterpret_cast<XPUType*>(const_cast<T*>(key_cache.data<T>())),
        block_tables.data<int>(),
        kv_seq_lod_dec_VP,
        start_token_ctx_VP,
        ordered_index_ctx_VP,
        bsz,
        num_head,
        dim_head,
        bsz,
        block_size,
        max_block_per_seq,
        "BLHD",
        "HLD");
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reshape_cached_kv");
    ret = xpu::batch_findmax<XPUType>(
        xpu_context,
        reinterpret_cast<XPUType*>(unpadding_k.data<T>()),
        bsz,
        num_head * dim_head,
        p_batch_max_ptrs);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "batch_findmax");
    unpadding_k.clear();
    ret = xpu::copy2d<float>(xpu_context,
                             p_batch_max_ptrs,
                             p_batch_max_ptrs_fill,
                             bsz,
                             1,
                             MAXPTR_N,
                             1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy2d");
    ret = xpu::max<float>(
        xpu_context,
        cache_k_per_batch_maxs.data<float>(),
        p_batch_max_ptrs_fill,
        const_cast<float*>(cache_k_per_batch_maxs.data<float>()),
        bsz * MAXPTR_N);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "max");
    ret = xpu::findmax<float>(
        xpu_context,
        const_cast<float*>(cache_k_per_batch_maxs.data<float>()),
        p_cache_k_max_data,
        bsz * MAXPTR_N);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "findmax");
    ret = xpu::reshape_cached_kv<XPUType, XPUType, int32_t>(
        xpu_context,
        reinterpret_cast<const XPUType*>(unpadding_v.data<T>()),
        reinterpret_cast<XPUType*>(const_cast<T*>(value_cache.data<T>())),
        block_tables.data<int>(),
        kv_seq_lod_dec_VP,
        start_token_ctx_VP,
        ordered_index_ctx_VP,
        bsz,
        num_head,
        dim_head,
        bsz,
        block_size,
        max_block_per_seq,
        "BLHD",
        "HLD");
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reshape_cached_kv");
    ret = xpu::batch_findmax<XPUType>(
        xpu_context,
        reinterpret_cast<XPUType*>(unpadding_v.data<T>()),
        bsz,
        num_head * dim_head,
        p_batch_max_ptrs);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "batch_findmax");
    unpadding_v.clear();
    ret = xpu::copy2d<float>(xpu_context,
                             p_batch_max_ptrs,
                             p_batch_max_ptrs_fill,
                             bsz,
                             1,
                             MAXPTR_N,
                             1);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy2d");
    ret = xpu::max<float>(
        xpu_context,
        cache_v_per_batch_maxs.data<float>(),
        p_batch_max_ptrs_fill,
        const_cast<float*>(cache_v_per_batch_maxs.data<float>()),
        bsz * MAXPTR_N);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "max");
    ret = xpu::findmax<float>(
        xpu_context,
        const_cast<float*>(cache_v_per_batch_maxs.data<float>()),
        p_cache_v_max_data,
        bsz * MAXPTR_N);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "findmax");

    VLOG(1) << "cachekv_quant_mode " << cachekv_quant_mode;
    std::vector<int32_t> qkvlod_dec(2 * (bsz + 1), 0);
    for (int bs = 0; bs < bsz; bs++) {
      qkvlod_dec[bs + 1] = bs + 1;
      qkvlod_dec[bsz + 1 + bs + 1] = lods_decoder_cpu[bs + 1] + bs + 1;
    }
    auto qkvlod_dec_vp =
        xpu::VectorParam<int32_t>{
            qkvlod_dec.data(), static_cast<int64_t>(qkvlod_dec.size()), nullptr}
            .to_xpu(RAII_GUARD);
    xpu::DecodeAttnParam decoder_attn_vsl_param(
        qkvlod_dec_vp, max_seq_len, num_head, dim_head, -1, 0, bsz, {});
    xpu::PageAttnParam<int> page_param(
        block_size, bsz, max_block_per_seq, ordered_index_ctx_VP, 0, "HLD");
    float* max_q_ptr = RAII_GUARD.alloc_l3_or_gm<float>(MAXPTR_N);
    ret = xpu::findmax<XPUType>(xpu_context,
                                reinterpret_cast<XPUType*>(unpadding_q.data()),
                                max_q_ptr,
                                token_num * num_head * dim_head);

    ret = xpu::qkv_paged_attention<XPUType,
                                   XPUType,
                                   XPUType,
                                   XPUType,
                                   int16_t,
                                   float,
                                   int>(
        xpu_context,
        reinterpret_cast<XPUType*>(unpadding_q.data()),
        reinterpret_cast<XPUType*>(const_cast<T*>(key_cache.data<T>())),
        reinterpret_cast<XPUType*>(const_cast<T*>(value_cache.data<T>())),
        block_tables.data<int>(),  // [pagep.max_batch_size,
                                   // pagep.max_num_blocks_per_seq]
        reinterpret_cast<XPUType*>(fmha_buf.data<T>()),
        max_q_ptr,
        p_cache_k_max_data,  // shape=[6], nullptr if pagep.quant_type == 1
        p_cache_v_max_data,  // shape=[6], nullptr if pagep.quant_type == 1
        nullptr,
        decoder_attn_vsl_param,  // attention 相关参数
        page_param);             // page attention 相关参数
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "qkv_paged_attention");
  }
  VLOG(3) << "decoder done";
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(block_multihead_attention_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::BlockMultiheadAttentionXPUKernel,
                   phi::dtype::float16) {
  kernel->InputAt(26).SetBackend(phi::Backend::CPU);
  kernel->InputAt(27).SetBackend(phi::Backend::CPU);
}
