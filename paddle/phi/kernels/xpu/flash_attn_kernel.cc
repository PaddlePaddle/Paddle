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

#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_XPU_XRE5
#include "xfa/flash_api.h"
#endif
namespace phi {

template <typename T, typename Context>
void FlashAttnUnpaddedKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float scale,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  // q, k, v [batch_size * seq_len, num_heads, head_dim]
  std::vector<int64_t> dims = common::vectorize(q.dims());

  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads = dims[1];
  const int head_size = dims[2];
  const int num_heads_k = k.dims()[1];

  // lod info, only support qlod == klod
  std::vector<int> qlod_vec(batch_size + 1, 0);
  int r = xpu_wait(ctx.x_context()->xpu_stream);
  PADDLE_ENFORCE_EQ(r, 0, "xpu_wait failed.");
  r = xpu_memcpy(qlod_vec.data(),
                 cu_seqlens_q.data<int>(),
                 sizeof(int32_t) * (batch_size + 1),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  PADDLE_ENFORCE_EQ(r, 0, "xpu_memcpy failed.");
  std::vector<int> klod_vec(batch_size + 1, 0);
  r = xpu_wait(ctx.x_context()->xpu_stream);
  PADDLE_ENFORCE_EQ(r, 0, "xpu_wait failed.");
  r = xpu_memcpy(klod_vec.data(),
                 cu_seqlens_k.data<int>(),
                 sizeof(int32_t) * (batch_size + 1),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  PADDLE_ENFORCE_EQ(r, 0, "xpu_memcpy failed.");
  // output: softmax_lse, 训练参数，给反向用于反向重计算的L
  bool is_cross_attn = false;
  for (int i = 0; i < batch_size + 1; ++i) {
    if (qlod_vec[i] != klod_vec[i]) {
      is_cross_attn = true;
      break;
    }
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  const XPUType* q_data = reinterpret_cast<const XPUType*>(q.data<T>());
  const XPUType* k_data = reinterpret_cast<const XPUType*>(k.data<T>());
  const XPUType* v_data = reinterpret_cast<const XPUType*>(v.data<T>());
  if (!is_cross_attn) {
    xpu::VectorParam<int32_t> lods{
        qlod_vec.data(), (int32_t)(qlod_vec.size()), nullptr};
    xpu::QKVAttnParam qkv_attn_param(
        lods,                     // only support qlods == kvlods
        num_heads,                // head_nums
        head_size,                // head_dim
        xpu::Activation_t::RELU,  // Activation_t
        -1,                       // last_slice_seq(unused param)
        false,                    // do_fc_qkv_fusion(unused param)
        -1,                       // pad_seqlen(unused param)
        -1,                       // hidden_dim(unused param)
        false,                    // is_pre_norm(unused param)
        false,                    // is_perchannel(unused param)
        0,                        // qkv_shape
        {},                       // z_shape
        AttnMacMaxPtrType_t::ATTN_WHOLE_BATCH,  // max_ptr_type
        -1,                                     // ldz(unused param)
        {},                                     // sqlod(unused param)
        scale);                                 // alpha
    qkv_attn_param.triangle_mask_autogen = causal;
    qkv_attn_param.key_value_head_num = num_heads_k;
    r = xpu::qkv_attention<XPUType,
                           XPUType,
                           XPUType,
                           XPUType,
                           int16_t,
                           float,
                           int,
                           float,
                           float>(ctx.x_context(),
                                  q_data,    // q
                                  k_data,    // k
                                  v_data,    // v
                                  out_data,  // out
                                  nullptr,   // max_q
                                  nullptr,   // max_k
                                  nullptr,   // max_v
                                  nullptr,   // max_ctx
                                  qkv_attn_param,
                                  nullptr,
                                  nullptr,
                                  nullptr);
    PADDLE_ENFORCE_EQ(r, 0, "xpu::qkv_attention failed.");
  } else {
    std::vector<int> lod;
    lod.reserve(2 * batch_size + 2);
    int real_max_len = 0;
    for (int i = 0; i < batch_size + 1; i++) {
      lod.push_back(qlod_vec[i]);
      if (i)
        real_max_len = std::max(qlod_vec[i] - qlod_vec[i - 1], real_max_len);
    }
    for (int i = 0; i < batch_size + 1; i++) {
      lod.push_back(klod_vec[i]);
      if (i)
        real_max_len = std::max(klod_vec[i] - klod_vec[i - 1], real_max_len);
    }
    xpu::DifSeqAttnParam dis_api_attn_param(
        {lod.data(), 2 * batch_size + 2, nullptr}, num_heads, head_size);
    XPUType* qk_buf = RAII_GUARD.alloc_l3_or_gm<XPUType>(
        batch_size * num_heads * real_max_len * real_max_len);
    float* qk_max_buf = RAII_GUARD.alloc_l3_or_gm<float>(6);
    r = xpu::qk_attention<XPUType, XPUType, XPUType, int16_t, float>(
        ctx.x_context(),
        q_data,
        k_data,
        qk_buf,
        nullptr,
        nullptr,
        qk_max_buf,
        dis_api_attn_param,
        nullptr);
    PADDLE_ENFORCE_EQ(r, 0, "xpu::qk_attention failed.");
    r = xpu::qk_v_attention<XPUType, XPUType, XPUType, int16_t, float>(
        ctx.x_context(),
        qk_buf,
        v_data,
        out_data,
        qk_max_buf,
        nullptr,
        nullptr,
        dis_api_attn_param,
        nullptr);
    PADDLE_ENFORCE_EQ(r, 0, "xpu::qk_v_attention failed.");
  }
}

template <typename T, typename Context>
void FlashAttnKernel(const Context& ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     const paddle::optional<DenseTensor>& fixed_seed_offset,
                     const paddle::optional<DenseTensor>& attn_mask,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     bool is_test,
                     const std::string& rng_name,
                     DenseTensor* out,
                     DenseTensor* softmax,
                     DenseTensor* softmax_lse,
                     DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_XPU_XRE5
  if (return_softmax == true) {
    PADDLE_THROW(phi::errors::Unimplemented("return_softmax should be false"));
  }

  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  // lod info
  std::vector<int> qlod_vec = {0};
  std::vector<int> kvlod_vec = {0};
  for (int batch_idx = 1; batch_idx <= batch_size; ++batch_idx) {
    qlod_vec.push_back(seqlen_q * batch_idx);
    kvlod_vec.push_back(seqlen_k * batch_idx);
  }
  api::VectorParam<int> qlod{
      qlod_vec.data(), static_cast<int64_t>(qlod_vec.size()), nullptr};
  api::VectorParam<int> kvlod{
      kvlod_vec.data(), static_cast<int64_t>(kvlod_vec.size()), nullptr};

  // output: softmax_lse
  std::vector<int64_t> softmax_lse_dims;
  softmax_lse_dims = {batch_size, num_heads, seqlen_q};
  softmax_lse->Resize(phi::make_ddim(softmax_lse_dims));
  ctx.template Alloc<float>(softmax_lse);

  // output: o
  ctx.template Alloc<T>(out);

  // generate seed offset
  seed_offset->Resize({2});
  int64_t* seed_offset_data = ctx.template HostAlloc<int64_t>(seed_offset);
  if (fixed_seed_offset.get_ptr()) {
    if ((fixed_seed_offset->place()).GetType() == phi::AllocationType::XPU) {
      memory_utils::Copy(phi::CPUPlace(),
                         seed_offset_data,
                         fixed_seed_offset->place(),
                         fixed_seed_offset->data<int64_t>(),
                         sizeof(int64_t) * 2);
    } else {
      const int64_t* fixed_seed_offset_data =
          fixed_seed_offset->data<int64_t>();
      seed_offset_data[0] = fixed_seed_offset_data[0];
      seed_offset_data[1] = fixed_seed_offset_data[1];
    }
  } else {
    std::pair<uint64_t, uint64_t> seed_offset_pair;
    uint64_t inc = batch_size * num_heads * 32;
    if (rng_name != "") {
      auto gen = phi::GetRandomSeedGenerator(rng_name);
      seed_offset_pair = gen->IncrementOffset(inc);
    } else {
      auto* gen = ctx.GetGenerator();
      seed_offset_pair = gen->IncrementOffset(inc);
    }
    seed_offset_data[0] = static_cast<int64_t>(seed_offset_pair.first);
    seed_offset_data[1] = static_cast<int64_t>(seed_offset_pair.second);
  }

  // raw pointers
  using XPUType = typename XPUTypeTrait<T>::Type;
  const XPUType* q_data = reinterpret_cast<const XPUType*>(q.data<T>());
  const XPUType* k_data = reinterpret_cast<const XPUType*>(k.data<T>());
  const XPUType* v_data = reinterpret_cast<const XPUType*>(v.data<T>());
  XPUType* out_data = reinterpret_cast<XPUType*>(out->data<T>());

  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  float* softmax_lse_data = softmax_lse->data<float>();
  const float* bias_data = nullptr;
  int64_t fa_layout = AttnQKVLayout_t::ATTN_BLHD;
  if (attn_mask.get_ptr() != nullptr) {
    const auto& mask_dims = attn_mask->dims();
    if (mask_dims.size() == 3 || (mask_dims[1] == 1 && mask_dims.size() == 4)) {
      fa_layout |= AttnQKVLayout_t::BIAS_BLL;
    } else {
      PADDLE_ENFORCE_EQ(
          mask_dims.size(),
          4,
          phi::errors::InvalidArgument("flash_attn_fwd requires mask's shape "
                                       "like [b,l,l] or [b, h, l, l]"));
    }
    if (attn_mask->dtype() == phi::DataType::FLOAT32) {
      bias_data = attn_mask->data<float>();
    } else if (attn_mask->dtype() == phi::DataType::FLOAT16 ||
               attn_mask->dtype() == phi::DataType::BFLOAT16) {
      float* bias_tmp = RAII_GUARD.alloc_l3_or_gm<float>(attn_mask->numel());
      int r = xpu::cast<XPUType, float>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(attn_mask->data<T>()),
          bias_tmp,
          attn_mask->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      bias_data = bias_tmp;
    } else {
      errors::Unimplemented(
          "Unsupported dtype for attention_mask in xpu flash attention, only "
          "float32, float16 and "
          "bfloat16 are supported.");
    }
  }
  // template <typename T, typename TACCUM, typename TGEMM, typename TID> int
  // mha_varlen_fwd(xdnn::Context* ctx, const T* q, const T* k, const T* v, T*
  // out, TACCUM* softmax_lse, const xdnn::VectorParam<TID>& lod_seqlens_q,
  // const xdnn::VectorParam<TID>& lod_seqlens_k, int64_t max_seqlen_q, int64_t
  // max_seqlen_k, int64_t head_num, int64_t head_num_k, int64_t head_dim, const
  // float softmax_scale = 0.0f, const float p_dropout = 0.0f, int seed =
  // 0x45678901, const bool is_causal = true, const TACCUM* attn_mask = nullptr,
  // const TACCUM* bias = nullptr, const float* q_maxptr = nullptr, const float*
  // k_maxptr = nullptr, const float* v_maxptr = nullptr, float* o_maxptr =
  // nullptr);
  int r = baidu::xpu::xfa::mha_varlen_fwd<XPUType, float, tfloat32, int>(
      ctx.x_context(),
      q_data,                                     // q
      k_data,                                     // k
      v_data,                                     // v
      out_data,                                   // out
      softmax_lse_data,                           // softmax_lse
      qlod,                                       // lod_seqlens_q
      kvlod,                                      // lod_seqlens_k
      seqlen_q,                                   // max_seqlen_q
      seqlen_k,                                   // max_seqlen_k
      num_heads,                                  // head_num
      num_heads_k,                                // head_num_k
      head_size,                                  // head_dim
      1.0f / std::sqrt(head_size),                // softmax_scale
      dropout,                                    // p_dropout
      static_cast<int32_t>(seed_offset_data[0]),  // seed
      causal,                                     // is_causal
      nullptr,                                    // attn_mask
      bias_data,                                  // bias
      nullptr,                                    // q_maxptr
      nullptr,                                    // k_maxptr
      nullptr,                                    // v_maxptr
      nullptr,                                    // o_maxptr
      false,                                      // is_qkv_fusion
      fa_layout                                   // qkv_layout
  );
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mha_varlen_fwd");
#else
  PADDLE_THROW(phi::errors::Unimplemented(
      "re-compile using -DWITH_XPU_XRE5=ON to use FlashAttnKernel"));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_unpadded,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(5).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_KERNEL(flash_attn,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlashAttnKernel,
                   phi::dtype::bfloat16,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}
