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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/xpu/flash_attn_utils.h"
#include "xfa/flash_api.h"
namespace phi {
#define MHA_VARLEN_FWD_TYPES_AND_ARGS(T, TACCUM, TGEMM, TID)                  \
  xpu::Context *ctx, const T *q, const T *k, const T *v, T *out,              \
      TACCUM *softmax_lse, const xpu::VectorParam<TID>&lod_seqlens_q,         \
      const xpu::VectorParam<TID>&lod_seqlens_k, int64_t max_seqlen_q,        \
      int64_t max_seqlen_k, int64_t head_num, int64_t head_num_k,             \
      int64_t head_dim, const float softmax_scale, const float p_dropout,     \
      int seed, const bool is_causal, const TACCUM *attn_mask,                \
      const TACCUM *bias, const float *q_maxptr, const float *k_maxptr,       \
      const float *v_maxptr, float *o_maxptr, const bool is_qkv_fusion,       \
      const int64_t qkv_layout, const float *alibi_slopes,                    \
      const std::vector<int64_t>&alibi_slopes_shape, int window_size_left,    \
      int window_size_right, int64_t v_head_dim,                              \
      const int *downstart_row_indices_data,                                  \
      const int *downend_row_indices_data,                                    \
      const int *upstart_row_indices_data, const int *upend_row_indices_data, \
      const int flash_mask_head_num, int *flashmask_maxmin,                   \
      XPUStream side_stream, int64_t fixlen_batch_num, bool unpadded_lse

#define MHA_VARLEN_FWD_ARGS                                                   \
  ctx, q, k, v, out, softmax_lse, lod_seqlens_q, lod_seqlens_k, max_seqlen_q, \
      max_seqlen_k, head_num, head_num_k, head_dim, softmax_scale, p_dropout, \
      seed, is_causal, attn_mask, bias, q_maxptr, k_maxptr, v_maxptr,         \
      o_maxptr, is_qkv_fusion, qkv_layout, alibi_slopes, alibi_slopes_shape,  \
      window_size_left, window_size_right, v_head_dim,                        \
      downstart_row_indices_data, downend_row_indices_data,                   \
      upstart_row_indices_data, upend_row_indices_data, flash_mask_head_num,  \
      flashmask_maxmin, side_stream, fixlen_batch_num, unpadded_lse

template <typename T, typename TACCUM, typename TGEMM, typename TID>
int mha_varlen_fwd_wrapper(
    MHA_VARLEN_FWD_TYPES_AND_ARGS(T, TACCUM, TGEMM, TID)) {
  PADDLE_THROW(
      "Unsupported template params combination for mha_varlen_fwd, should not "
      "reach here.");
}

#define DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(T, TACCUM, TGEMM, TID) \
  template <>                                                         \
  int mha_varlen_fwd_wrapper<T, TACCUM, TGEMM, TID>(                  \
      MHA_VARLEN_FWD_TYPES_AND_ARGS(T, TACCUM, TGEMM, TID)) {         \
    return baidu::xpu::xfa::mha_varlen_fwd<T, TACCUM, TGEMM, TID>(    \
        MHA_VARLEN_FWD_ARGS);                                         \
  }

DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(float, float, float, int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(float, float, tfloat32, int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeFP16,
                                       XPUTypeFP16,
                                       XPUTypeFP16,
                                       int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeFP16, float, XPUTypeFP16, int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeFP16, float, float, int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeFP16, float, tfloat32, int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeBF16, float, float, int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeBF16, float, tfloat32, int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeBF16, float, XPUTypeFP16, int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeBF16,
                                       XPUTypeFP16,
                                       XPUTypeFP16,
                                       int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeBF16,
                                       XPUTypeBF16,
                                       XPUTypeBF16,
                                       int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeBF16, float, XPUTypeBF16, int);
DECLARE_SUPPORTED_MHA_VARLEN_FWD_TYPES(XPUTypeBF16,
                                       XPUTypeFP16,
                                       XPUTypeBF16,
                                       int);

#define MHA_VARLEN_FWD(T1, T2, T3, T4)                    \
  mha_varlen_fwd_wrapper<T1, T2, T3, T4>(                 \
      dev_ctx.x_context(),                                \
      q_data,                                             \
      k_data,                                             \
      v_data,                                             \
      out_data,                                           \
      softmax_lse_data,                                   \
      lod_seqlen_q,                                       \
      lod_seqlen_k,                                       \
      max_seqlen_q,                                       \
      max_seqlen_k,                                       \
      num_heads,                                          \
      num_heads_k,                                        \
      head_size,                                          \
      real_scale,                                         \
      real_dropout,                                       \
      static_cast<int32_t>(seed_offset_data[0]),          \
      causal,                                             \
      nullptr,                                            \
      bias_data,                                          \
      nullptr,                                            \
      nullptr,                                            \
      nullptr,                                            \
      nullptr,                                            \
      false,                                              \
      fa_layout,                                          \
      nullptr,                                            \
      {},                                                 \
      -1,                                                 \
      -1,                                                 \
      head_size_v,                                        \
      (const int*)downstart_row_indices_data,             \
      (const int*)downend_row_indices_data,               \
      (const int*)upstart_row_indices_data,               \
      (const int*)upend_row_indices_data,                 \
      is_flashmask ? startend_row_indices->dims()[1] : 0, \
      nullptr,                                            \
      is_flashmask ? flashmask_stream : nullptr,          \
      0,                                                  \
      false)

template <typename T, typename Context>
void FlashAttnKernelBase(const Context& dev_ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const api::VectorParam<int>& lod_seqlen_q,
                         const api::VectorParam<int>& lod_seqlen_k,
                         const optional<DenseTensor>& fixed_seed_offset,
                         const optional<DenseTensor>& attn_mask,
                         const optional<DenseTensor>& startend_row_indices,
                         const int64_t batch_size,
                         const Scalar& max_seqlen_q_,
                         const Scalar& max_seqlen_k_,
                         const int64_t num_heads,
                         const int64_t num_heads_k,
                         const int64_t head_size,
                         const int64_t head_size_v,
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
  // Handle 0-size tensors: return zeros without calling XPU kernel
  // to avoid invalid memory access
  if (q.numel() == 0 || k.numel() == 0 || v.numel() == 0) {
    if (out) {
      Full<T, Context>(dev_ctx, out->dims(), 0, out);
    }
    if (softmax) {
      Full<T, Context>(dev_ctx, softmax->dims(), 0, softmax);
    }
    if (softmax_lse) {
      std::vector<int64_t> softmax_lse_dims = {
          batch_size, num_heads, max_seqlen_q_.to<int64_t>()};
      softmax_lse->Resize(softmax_lse_dims);
      Full<float, Context>(dev_ctx, softmax_lse->dims(), 0, softmax_lse);
    }
    if (seed_offset) {
      Full<int64_t, Context>(dev_ctx, seed_offset->dims(), 0, seed_offset);
    }
    return;
  }

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  float real_scale = scale == 0.0f ? 1.0f / std::sqrt(head_size) : scale;
  float real_dropout = is_test ? 0.0f : dropout;

  // output: softmax_lse, 训练参数，给反向用于反向重计算的L
  int64_t max_seqlen_q = max_seqlen_q_.to<int64_t>();
  int64_t max_seqlen_k = max_seqlen_k_.to<int64_t>();
  std::vector<int64_t> softmax_lse_dims = {batch_size, num_heads, max_seqlen_q};
  softmax_lse->Resize(softmax_lse_dims);
  dev_ctx.template Alloc<float>(softmax_lse);

  // output: o
  dev_ctx.template Alloc<T>(out);

  // output: seed_offset
  seed_offset->Resize({2});
  int64_t* seed_offset_data = dev_ctx.template HostAlloc<int64_t>(seed_offset);

  phi::GenerateRNGState(dev_ctx,
                        fixed_seed_offset,
                        seed_offset_data,
                        rng_name,
                        batch_size,
                        num_heads);

  // raw pointers
  using XPUType = typename XPUTypeTrait<T>::Type;
  const XPUType* q_data = reinterpret_cast<const XPUType*>(q.data<T>());
  const XPUType* k_data = reinterpret_cast<const XPUType*>(k.data<T>());
  const XPUType* v_data = reinterpret_cast<const XPUType*>(v.data<T>());
  XPUType* out_data = reinterpret_cast<XPUType*>(out->data<T>());
  int64_t fa_layout = AttnQKVLayout_t::ATTN_BLHD;
  DenseTensor downstart_row_indices, upend_row_indices, downend_row_indices,
      upstart_row_indices;
  void *downstart_row_indices_data = nullptr, *upend_row_indices_data = nullptr,
       *downend_row_indices_data = nullptr, *upstart_row_indices_data = nullptr;
  bool is_flashmask = startend_row_indices.get_ptr() != nullptr;
  XPUStream flashmask_stream;
  if (is_flashmask) {
    xpu_stream_create(&flashmask_stream);
    PADDLE_ENFORCE_EQ(
        startend_row_indices->dims().size(),
        4,
        common::errors::InvalidArgument(
            "flashmask_attention receive startend_row_indices with dim "
            "[batch_size, num_heads,seq_len, mask_bounds]"));
    PADDLE_ENFORCE_EQ(startend_row_indices->dims()[3] == 1 ||
                          startend_row_indices->dims()[3] == 2 ||
                          startend_row_indices->dims()[3] == 4,
                      true,
                      common::errors::InvalidArgument(
                          "flashmask_attention startend_row_indices "
                          "mask_bounds must in [1,2,4]"));
    downstart_row_indices =
        phi::Slice<int32_t>(dev_ctx, startend_row_indices.get(), {3}, {0}, {1});
    downstart_row_indices_data = downstart_row_indices.data();
    if (startend_row_indices->dims()[3] == 2) {
      if (!causal) {
        upend_row_indices = phi::Slice<int32_t>(
            dev_ctx, startend_row_indices.get(), {3}, {1}, {2});
        upend_row_indices_data = upend_row_indices.data();
      } else {
        downend_row_indices = phi::Slice<int32_t>(
            dev_ctx, startend_row_indices.get(), {3}, {1}, {2});
        downend_row_indices_data = downend_row_indices.data();
      }
    } else if (startend_row_indices->dims()[3] == 4) {
      upend_row_indices = phi::Slice<int32_t>(
          dev_ctx, startend_row_indices.get(), {3}, {3}, {4});
      upend_row_indices_data = upend_row_indices.data();
      downend_row_indices = phi::Slice<int32_t>(
          dev_ctx, startend_row_indices.get(), {3}, {1}, {2});
      downend_row_indices_data = downend_row_indices.data();
      upstart_row_indices = phi::Slice<int32_t>(
          dev_ctx, startend_row_indices.get(), {3}, {2}, {3});
      upstart_row_indices_data = upstart_row_indices.data();
    }
  } else {
    if (attn_mask.get_ptr() != nullptr) {
      const auto& mask_dims = attn_mask->dims();
      if (mask_dims.size() == 3 ||
          (mask_dims[1] == 1 && mask_dims.size() == 4)) {
        fa_layout |= AttnQKVLayout_t::BIAS_BLL;
      } else {
        PADDLE_ENFORCE_EQ(mask_dims.size(),
                          4,
                          common::errors::InvalidArgument(
                              "flash_attn_fwd requires mask's shape "
                              "like [b,l,l] or [b, h, l, l]"));
      }
      if (!(attn_mask->dtype() == DataType::FLOAT32 ||
            attn_mask->dtype() == DataType::FLOAT16 ||
            attn_mask->dtype() == DataType::BFLOAT16)) {
        errors::Unimplemented(
            "Unsupported dtype for attention_mask in xpu flash attention, only "
            "float32, float16 and bfloat16 are supported.");
      }
    }
  }

  XPU_FA_DTYPE tgemm_dtype = get_flash_attn_tgemm<T>();
  XPU_FA_DTYPE taccum_dtype = get_flash_attn_taccum<T>();

  int r = 0;
  float* softmax_lse_data_fp32 = softmax_lse->data<float>();
  if (tgemm_dtype == XPU_FA_DTYPE::FA_FLOAT16 &&
      taccum_dtype == XPU_FA_DTYPE::FA_FLOAT16) {
    // for taccum = fp16
    XPUTypeFP16* softmax_lse_data =
        RAII_GUARD.alloc_l3_or_gm<XPUTypeFP16>(softmax_lse->numel());

    const XPUTypeFP16* bias_data = nullptr;
    if (!is_flashmask && attn_mask.get_ptr() != nullptr) {
      if (attn_mask->dtype() == DataType::FLOAT16) {
        bias_data = reinterpret_cast<const XPUTypeFP16*>(
            attn_mask->data<phi::float16>());
      } else {  // DataType::BFLOAT16
        XPUTypeFP16* bias_tmp =
            RAII_GUARD.alloc_l3_or_gm<XPUTypeFP16>(attn_mask->numel());
        r = xpu::cast<XPUType, XPUTypeFP16>(
            dev_ctx.x_context(),
            reinterpret_cast<const XPUType*>(attn_mask->data<T>()),
            bias_tmp,
            attn_mask->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
        bias_data = bias_tmp;
      }
    }
    r = MHA_VARLEN_FWD(XPUType, XPUTypeFP16, XPUTypeFP16, int32_t);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mha_varlen_fwd");
    r = xpu::cast<XPUTypeFP16, float>(dev_ctx.x_context(),
                                      softmax_lse_data,
                                      softmax_lse_data_fp32,
                                      softmax_lse->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  } else {
    float* softmax_lse_data = softmax_lse_data_fp32;
    const float* bias_data = nullptr;

    if (!is_flashmask && attn_mask.get_ptr() != nullptr) {
      float* bias_tmp = RAII_GUARD.alloc_l3_or_gm<float>(attn_mask->numel());
      r = xpu::cast<XPUType, float>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(attn_mask->data<T>()),
          bias_tmp,
          attn_mask->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      bias_data = bias_tmp;
    }
    if (tgemm_dtype == XPU_FA_DTYPE::FA_FLOAT16) {
      r = MHA_VARLEN_FWD(XPUType, float, XPUTypeFP16, int32_t);
    } else if (tgemm_dtype == XPU_FA_DTYPE::FA_FLOAT) {
      r = MHA_VARLEN_FWD(XPUType, float, float, int32_t);
    } else {
      r = MHA_VARLEN_FWD(XPUType, float, tfloat32, int32_t);
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mha_varlen_fwd");
  }

  if (is_flashmask && flashmask_stream != nullptr) {
    int r = xpu_wait(flashmask_stream);
    PADDLE_ENFORCE_XPU_SUCCESS(r);
    xpu_stream_destroy(flashmask_stream);
  }
}

template <typename T, typename Context>
void FlashAttnUnpaddedKernel(const Context& dev_ctx,
                             const DenseTensor& q,
                             const DenseTensor& k,
                             const DenseTensor& v,
                             const DenseTensor& cu_seqlens_q,
                             const DenseTensor& cu_seqlens_k,
                             const optional<DenseTensor>& fixed_seed_offset,
                             const optional<DenseTensor>& attn_mask,
                             const Scalar& max_seqlen_q,
                             const Scalar& max_seqlen_k,
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
  // q, k, v [batch_size * seq_len, num_heads, head_dim]
  std::vector<int64_t> dims = vectorize(q.dims());

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size = dims[2];
  const int64_t num_heads_k = k.dims()[1];
  const int64_t head_size_v = v.dims()[2];
  api::VectorParam<int> qlod{cu_seqlens_q.data<int>(),
                             static_cast<int64_t>(cu_seqlens_q.numel()),
                             nullptr};
  api::VectorParam<int> kvlod{cu_seqlens_k.data<int>(),
                              static_cast<int64_t>(cu_seqlens_k.numel()),
                              nullptr};

  FlashAttnKernelBase<T>(dev_ctx,
                         q,
                         k,
                         v,
                         qlod,
                         kvlod,
                         fixed_seed_offset,
                         attn_mask,
                         paddle::none,
                         batch_size,
                         max_seqlen_q,
                         max_seqlen_k,
                         num_heads,
                         num_heads_k,
                         head_size,
                         head_size_v,
                         scale,
                         dropout,
                         causal,
                         return_softmax,
                         is_test,
                         rng_name,
                         out,
                         softmax,
                         softmax_lse,
                         seed_offset);
}

template <typename T, typename Context>
void FlashAttnKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     const optional<DenseTensor>& fixed_seed_offset,
                     const optional<DenseTensor>& attn_mask,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     bool is_test,
                     const std::string& rng_name,
                     DenseTensor* out,
                     DenseTensor* softmax,
                     DenseTensor* softmax_lse,
                     DenseTensor* seed_offset) {
  if (return_softmax == true) {
    PADDLE_THROW(
        common::errors::Unimplemented("return_softmax should be false"));
  }

  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(k.dims().size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(v.dims().size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(out->dims().size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];
  const int64_t head_size_v = v.dims()[3];
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

  FlashAttnKernelBase<T>(dev_ctx,
                         q,
                         k,
                         v,
                         qlod,
                         kvlod,
                         fixed_seed_offset,
                         attn_mask,
                         paddle::none,
                         batch_size,
                         seqlen_q,
                         seqlen_k,
                         num_heads,
                         num_heads_k,
                         head_size,
                         head_size_v,
                         0.0,  // scale
                         dropout,
                         causal,
                         return_softmax,
                         is_test,
                         rng_name,
                         out,
                         softmax,
                         softmax_lse,
                         seed_offset);
}

template <typename T, typename Context>
void FlashMaskKernel(const Context& dev_ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     const DenseTensor& startend_row_indices,
                     const optional<DenseTensor>& fixed_seed_offset,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     bool is_test,
                     const std::string& rng_name,
                     DenseTensor* out,
                     DenseTensor* softmax,
                     DenseTensor* softmax_lse,
                     DenseTensor* seed_offset) {
  if (return_softmax == true) {
    PADDLE_THROW(
        common::errors::Unimplemented("return_softmax should be false"));
  }

  // Handle 0-size tensors: return zeros without calling XPU kernel
  // to avoid invalid memory access
  if (q.numel() == 0 || k.numel() == 0 || v.numel() == 0) {
    if (out) {
      Full<T, Context>(dev_ctx, out->dims(), 0, out);
    }
    if (softmax) {
      Full<T, Context>(dev_ctx, softmax->dims(), 0, softmax);
    }
    if (softmax_lse) {
      Full<float, Context>(dev_ctx, softmax_lse->dims(), 0, softmax_lse);
    }
    if (seed_offset) {
      Full<int64_t, Context>(dev_ctx, seed_offset->dims(), 0, seed_offset);
    }
    return;
  }

  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    common::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];
  const int64_t head_size_v = v.dims()[3];
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

  FlashAttnKernelBase<T>(dev_ctx,
                         q,
                         k,
                         v,
                         qlod,
                         kvlod,
                         fixed_seed_offset,
                         paddle::none,
                         startend_row_indices,
                         batch_size,
                         seqlen_q,
                         seqlen_k,
                         num_heads,
                         num_heads_k,
                         head_size,
                         head_size_v,
                         0.0,  // scale
                         dropout,
                         causal,
                         return_softmax,
                         is_test,
                         rng_name,
                         out,
                         softmax,
                         softmax_lse,
                         seed_offset);
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_unpadded,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedKernel,
                   float,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(3).SetBackend(phi::Backend::CPU);  // cu_seqlens_q
  kernel->InputAt(4).SetBackend(phi::Backend::CPU);  // cu_seqlens_k
  kernel->InputAt(5).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_KERNEL(flash_attn,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlashAttnKernel,
                   phi::bfloat16,
                   float,
                   phi::float16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_KERNEL(flashmask_attention,
                   XPU,
                   ALL_LAYOUT,
                   phi::FlashMaskKernel,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(4).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}
