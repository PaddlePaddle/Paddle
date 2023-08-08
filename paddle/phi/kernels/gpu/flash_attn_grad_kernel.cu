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

#include "paddle/phi/kernels/flash_attn_grad_kernel.h"
#include "glog/logging.h"  // For VLOG()
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/reshape_kernel.h"

DECLARE_bool(cudnn_deterministic);

namespace phi {

template <typename T, typename Context>
void FlashAttnUnpaddedGradImpl(const Context& ctx,
                               const DenseTensor& q,
                               const DenseTensor& k,
                               const DenseTensor& v,
                               const DenseTensor& cu_seqlens_q,
                               const DenseTensor& cu_seqlens_k,
                               const DenseTensor& out,
                               const DenseTensor& softmax_lse,
                               const DenseTensor& seed_offset,
                               const paddle::optional<DenseTensor>& attn_mask,
                               const DenseTensor& dout,
                               int64_t max_seqlen_q,
                               int64_t max_seqlen_k,
                               float scale,
                               float dropout,
                               bool causal,
                               DenseTensor* dq,
                               DenseTensor* dk,
                               DenseTensor* dv) {
#ifdef PADDLE_WITH_FLASHATTN
  const cudaStream_t stream = ctx.stream();

  auto dims = q.dims();
  int64_t total_q = dims[0];
  int64_t num_heads = dims[1];
  int64_t head_size = dims[2];

  int64_t total_k = k.dims()[0];
  int64_t batch_size = cu_seqlens_q.numel() - 1;

  PADDLE_ENFORCE_NE(causal,
                    true,
                    phi::errors::InvalidArgument(
                        "attn_mask is not nullptr, causal can not be true"));

  PADDLE_ENFORCE_EQ(
      head_size == 32 || head_size == 64 || head_size == 128,
      true,
      phi::errors::InvalidArgument("The head_dim is expected to be either 32, "
                                   "64, or 128, but recieved %d.",
                                   head_size));

  const int64_t* seed_offset_data = seed_offset.data<int64_t>();
  uint64_t seed = static_cast<uint64_t>(seed_offset_data[0]);
  uint64_t offset = static_cast<uint64_t>(seed_offset_data[1]);
  VLOG(10) << "FlashAttn bwd seed: " << seed << ", offset: " << offset;

  int64_t seqlen_q = ((max_seqlen_q + 16 - 1) / 16) * 16;
  DenseTensor dsoftmax = Empty<float>(ctx, {batch_size, num_heads, seqlen_q});

  const DenseTensor* attn_mask_tensor = attn_mask.get_ptr();
  std::vector<int64_t> mask_dims = GetAttnMaskDims(attn_mask_tensor);

  int fa_num_splits = 0;
  bool fa_is_bf16 = q.dtype() == DataType::BFLOAT16;
  float fa_with_mask_scale = 1.0f;
  bool fa_zero_tensors = false;

  uint64_t workspace_size;
  bool succ = phi::dynload::flash_attn_bwd_with_bias_and_mask(
      static_cast<const void*>(q.data()),
      static_cast<const void*>(k.data()),
      static_cast<const void*>(v.data()),
      static_cast<void*>(dq->data()),
      static_cast<void*>(dk->data()),
      static_cast<void*>(dv->data()),
      nullptr,  // set out to nullptr to calculate workspace size
      dout.data(),
      static_cast<const int32_t*>(cu_seqlens_q.data()),
      static_cast<const int32_t*>(cu_seqlens_k.data()),
      total_q,
      total_k,
      batch_size,
      num_heads,
      head_size,
      max_seqlen_q,
      max_seqlen_k,
      dropout,
      fa_with_mask_scale,
      fa_zero_tensors,
      fa_is_bf16,
      fa_num_splits,
      static_cast<const void*>(softmax_lse.data()),
      static_cast<void*>(dsoftmax.data()),
      nullptr,
      nullptr,
      &workspace_size,
      stream,
      seed,
      offset,
      attn_mask_tensor ? attn_mask_tensor->data() : nullptr,
      nullptr,
      mask_dims.data() ? mask_dims.data() : nullptr,
      nullptr);
  CheckFlashAttnStatus(succ);

  DenseTensor workspace;
  if (workspace_size > 0) {
    workspace = Empty<float>(
        ctx, {static_cast<int64_t>(workspace_size / sizeof(float))});
  }

  succ = phi::dynload::flash_attn_bwd_with_bias_and_mask(
      static_cast<const void*>(q.data()),
      static_cast<const void*>(k.data()),
      static_cast<const void*>(v.data()),
      static_cast<void*>(dq->data()),
      static_cast<void*>(dk->data()),
      static_cast<void*>(dv->data()),
      out.data(),  // set out to nullptr to calculate workspace size
      dout.data(),
      static_cast<const int32_t*>(cu_seqlens_q.data()),
      static_cast<const int32_t*>(cu_seqlens_k.data()),
      total_q,
      total_k,
      batch_size,
      num_heads,
      head_size,
      max_seqlen_q,
      max_seqlen_k,
      dropout,
      fa_with_mask_scale,
      fa_zero_tensors,
      fa_is_bf16,
      fa_num_splits,
      static_cast<const void*>(softmax_lse.data()),
      static_cast<void*>(dsoftmax.data()),
      nullptr,
      workspace_size > 0 ? workspace.data() : nullptr,
      &workspace_size,
      stream,
      seed,
      offset,
      attn_mask_tensor ? attn_mask_tensor->data() : nullptr,
      nullptr,
      mask_dims.data() ? mask_dims.data() : nullptr,
      nullptr);
  CheckFlashAttnStatus(succ);

  int64_t q_size = total_q * num_heads * head_size;
  ComputeScaleQ(ctx, q_size, scale, dq->data<T>(), dq->data<T>());
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnUnpaddedGradKernel(const Context& ctx,
                                 const DenseTensor& q,
                                 const DenseTensor& k,
                                 const DenseTensor& v,
                                 const DenseTensor& cu_seqlens_q,
                                 const DenseTensor& cu_seqlens_k,
                                 const DenseTensor& out,
                                 const DenseTensor& softmax_lse,
                                 const DenseTensor& seed_offset,
                                 const paddle::optional<DenseTensor>& attn_mask,
                                 const DenseTensor& dout,
                                 int64_t max_seqlen_q,
                                 int64_t max_seqlen_k,
                                 float scale,
                                 float dropout,
                                 bool causal,
                                 DenseTensor* dq,
                                 DenseTensor* dk,
                                 DenseTensor* dv) {
#ifdef PADDLE_WITH_FLASHATTN
  ctx.template Alloc<T>(dq);
  ctx.template Alloc<T>(dk);
  ctx.template Alloc<T>(dv);

  const cudaStream_t stream = ctx.stream();

  // q,k,v [total_*, num_heads, head_dim]
  auto dims = q.dims();

  if (attn_mask.get_ptr()) {
    FlashAttnUnpaddedGradImpl<T, Context>(ctx,
                                          q,
                                          k,
                                          v,
                                          cu_seqlens_q,
                                          cu_seqlens_k,
                                          out,
                                          softmax_lse,
                                          seed_offset,
                                          attn_mask,
                                          dout,
                                          max_seqlen_q,
                                          max_seqlen_k,
                                          scale,
                                          dropout,
                                          causal,
                                          dq,
                                          dk,
                                          dv);
  } else {
    const int64_t total_q = dims[0];
    const int64_t batch_size = cu_seqlens_q.numel() - 1;
    const int64_t num_heads = dims[1];
    const int64_t head_size_og = dout.dims()[2];
    const int64_t head_size = dims[2];
    const int64_t total_k = k.dims()[0];
    const int64_t num_heads_k = k.dims()[1];

    // TODO(umiswing): add deterministic in fa2.
    // int num_splits = 0;  // 0 for an internal heuristic, which is optimal
    // if (FLAGS_cudnn_deterministic) {
    //   num_splits = 1;
    // }

    // TODO(umiswing): add shape check
    PADDLE_ENFORCE_EQ(
        head_size_og,
        head_size,
        phi::errors::InvalidArgument(
            "flash_attn_bwd receive input with head_size_og == head_size"));

    FlashAttnBwdParamsV2 params =
        FlashAttnBwdParamsV2(ctx,
                             batch_size,
                             max_seqlen_q,
                             max_seqlen_k,
                             num_heads,
                             num_heads_k,
                             head_size,
                             dropout,
                             scale,
                             causal,
                             q.dtype(),
                             seed_offset.data<int64_t>());

    VLOG(10) << "FlashAttn bwd seed: " << params.seed
             << ", offset: " << params.offset;

    bool succ =
        phi::dynload::flash_attn_varlen_bwd(dout.data(),
                                            q.data(),
                                            k.data(),
                                            v.data(),
                                            out.data(),
                                            params.softmax_d.data(),
                                            softmax_lse.data(),
                                            cu_seqlens_q.data<int32_t>(),
                                            cu_seqlens_k.data<int32_t>(),
                                            params.rng_state.data(),
                                            dq->data(),
                                            dk->data(),
                                            dv->data(),
                                            params.dq_accum.data(),
                                            params.batch_size,
                                            params.max_seqlen_q,
                                            params.max_seqlen_k,
                                            params.seqlen_q_rounded,
                                            params.seqlen_k_rounded,
                                            params.num_heads,
                                            params.num_heads_k,
                                            params.head_size,
                                            params.head_size_rounded,
                                            params.dropout,
                                            params.scale,
                                            params.causal,
                                            params.is_bf16,
                                            stream,
                                            params.seed,
                                            params.offset);
    CheckFlashAttnStatus(succ);
  }
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const DenseTensor& out,
                         const DenseTensor& softmax_lse,
                         const DenseTensor& seed_offset,
                         const paddle::optional<DenseTensor>& attn_mask,
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
#ifdef PADDLE_WITH_FLASHATTN
  // q,k,v [batch_size, seq_len, num_heads, head_dim]

  const auto& dims = q.dims();
  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size_og = dout.dims()[3];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  const int64_t total_q = batch_size * seqlen_q;
  const int64_t total_k = batch_size * seqlen_k;

  // TODO(umiswing): add shape check
  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      phi::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  VLOG(10) << "FlashAttn bwd dims q[" << q.dims() << "], k[" << k.dims()
           << "], v[" << v.dims() << "]";

  const float scale = 1.0f / std::sqrt(head_size);
  if (attn_mask.get_ptr()) {
    DenseTensor q_t_s, k_t_s, v_t_s;
    q_t_s.ShareDataWith(q).Resize({total_q, num_heads, head_size});
    k_t_s.ShareDataWith(k).Resize({total_k, num_heads, head_size});
    v_t_s.ShareDataWith(v).Resize({total_k, num_heads, head_size});

    DenseTensor cu_seqlens_q;
    DenseTensor cu_seqlens_k;
    ArangeNullaryKernel<int32_t, Context>(
        ctx, 0, (batch_size + 1) * seqlen_q, seqlen_q, &cu_seqlens_q);
    ArangeNullaryKernel<int32_t, Context>(
        ctx, 0, (batch_size + 1) * seqlen_k, seqlen_k, &cu_seqlens_k);

    FlashAttnUnpaddedGradKernel<T, Context>(ctx,
                                            q_t_s,
                                            k_t_s,
                                            v_t_s,
                                            cu_seqlens_q,
                                            cu_seqlens_k,
                                            out,
                                            softmax_lse,
                                            seed_offset,
                                            attn_mask,
                                            dout,
                                            seqlen_q,
                                            seqlen_k,
                                            scale,
                                            dropout,
                                            causal,
                                            dq,
                                            dk,
                                            dv);
  } else {
    FlashAttnBwdParamsV2 params =
        FlashAttnBwdParamsV2(ctx,
                             batch_size,
                             seqlen_q,
                             seqlen_k,
                             num_heads,
                             num_heads_k,
                             head_size,
                             dropout,
                             scale,
                             causal,
                             q.dtype(),
                             seed_offset.data<int64_t>());

    ctx.template Alloc<T>(dq);
    ctx.template Alloc<T>(dk);
    ctx.template Alloc<T>(dv);

    cudaStream_t stream = ctx.stream();

    VLOG(10) << "FlashAttn bwd seed: " << params.seed
             << ", offset: " << params.offset;

    bool succ = phi::dynload::flash_attn_bwd(dout.data(),
                                             q.data(),
                                             k.data(),
                                             v.data(),
                                             out.data(),
                                             params.softmax_d.data(),
                                             softmax_lse.data(),
                                             params.rng_state.data(),
                                             dq->data(),
                                             dk->data(),
                                             dv->data(),
                                             params.dq_accum.data(),
                                             params.batch_size,
                                             params.max_seqlen_q,
                                             params.max_seqlen_k,
                                             params.seqlen_q_rounded,
                                             params.seqlen_k_rounded,
                                             params.num_heads,
                                             params.num_heads_k,
                                             params.head_size,
                                             params.head_size_rounded,
                                             params.dropout,
                                             params.scale,
                                             params.causal,
                                             params.is_bf16,
                                             stream,
                                             params.seed,
                                             params.offset);
    CheckFlashAttnStatus(succ);
  }
#else
  RaiseNotSupportedError();
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_unpadded_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}

PD_REGISTER_KERNEL(flash_attn_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}
