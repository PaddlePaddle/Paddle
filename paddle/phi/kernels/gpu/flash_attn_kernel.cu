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

#include <cstddef>
#include "glog/logging.h"  // For VLOG()
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"

namespace phi {
template <typename OutT>
struct ZeroFunctor {
  __device__ __forceinline__ OutT operator()() const {
    return static_cast<OutT>(0);
  }
};

template <typename T, typename Context>
void FlashAttnUnpaddedBaseKernel(
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
    DenseTensor* seed_offset,
    bool varlen_padded) {
#ifdef PADDLE_WITH_FLASHATTN
  if (!out->IsInitialized()) ctx.template Alloc<T>(out);
  if (varlen_padded) {
    std::vector<const DenseTensor*> inputs{};
    std::vector<DenseTensor*> outputs{out};

    phi::funcs::ElementwiseKernel<T>(ctx, inputs, &outputs, ZeroFunctor<T>());
  }
  cudaStream_t stream = ctx.stream();

  // q, k, v [total_q/k/v, num_heads, head_dim]
  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      k.dims().size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      v.dims().size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      out->dims().size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size = dims[2];
  const int64_t num_heads_k = k.dims()[1];

  // TODO(umiswing): add shape check

  FlashAttnFwdParamsV2<T> params =
      FlashAttnFwdParamsV2<T>(ctx,
                              batch_size,
                              max_seqlen_q,
                              max_seqlen_k,
                              num_heads,
                              num_heads_k,
                              head_size,
                              dropout,
                              scale,
                              causal,
                              return_softmax,
                              q.dtype(),
                              is_test,
                              rng_name,
                              0,  // attn_mask_start_row
                              fixed_seed_offset,
                              attn_mask,
                              nullptr,  // attn_mask_start_row_indices
                              softmax,
                              softmax_lse,
                              seed_offset);

  VLOG(10) << "FlashAttn fwd seed: " << params.seed
           << ", offset: " << params.offset;

  bool succ = phi::dynload::flash_attn_varlen_fwd(
      q.data(),
      k.data(),
      v.data(),
      cu_seqlens_q.data<int32_t>(),
      cu_seqlens_k.data<int32_t>(),
      params.rng_state.data(),
      out->data(),
      params.return_softmax ? softmax->data() : nullptr,
      softmax_lse->data(),
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
      params.softmax_scale,
      1.0f / params.softmax_scale,
      params.causal,
      params.return_softmax,
      params.is_bf16,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr,
      q.strides()[0],
      k.strides()[0],
      v.strides()[0],
      q.strides()[1],
      k.strides()[1],
      v.strides()[1],
      out->strides()[0],
      out->strides()[1],
      max_seqlen_q * q.strides()[0],
      max_seqlen_k * k.strides()[0],
      max_seqlen_k * v.strides()[0],
      max_seqlen_q * out->strides()[0],
      varlen_padded);
  CheckFlashAttnStatus(succ);
#else
  RaiseNotSupportedError();
#endif
}

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
#ifdef PADDLE_WITH_FLASHATTN
  FlashAttnUnpaddedBaseKernel<T>(ctx,
                                 q,
                                 k,
                                 v,
                                 cu_seqlens_q,
                                 cu_seqlens_k,
                                 fixed_seed_offset,
                                 attn_mask,
                                 max_seqlen_q,
                                 max_seqlen_k,
                                 scale,
                                 dropout,
                                 causal,
                                 return_softmax,
                                 is_test,
                                 rng_name,
                                 out,
                                 softmax,
                                 softmax_lse,
                                 seed_offset,
                                 false /*varlen_padded*/);
#else
  RaiseNotSupportedError();
#endif
}

static void sliceFlattenView(const DenseTensor& in,
                             DenseTensor* out,
                             int axis,
                             int64_t offset,
                             int64_t sliceLength) {
  PADDLE_ENFORCE_LT(
      axis,
      in.dims().size(),
      phi::errors::InvalidArgument("sliceView receive axis out of bound"));
  std::array<int64_t, DDim::kMaxRank> dimArr;
  std::array<int64_t, DDim::kMaxRank> strideArr;
  auto id = dimArr.begin(), is = strideArr.begin();
  for (int i = 0; i < in.dims().size(); i++) {
    if (i == axis) continue;
    if (i == axis + 1)
      *id = in.dims()[i] * sliceLength;
    else
      *id = in.dims()[i];
    *is = in.strides()[i];
    id++;
    is++;
  }
  *out = DenseTensor{
      in.Holder(),
      DenseTensorMeta{in.dtype(),
                      DDim{dimArr.data(), in.dims().size() - 1},
                      DDim(strideArr.data(), in.dims().size() - 1)}};
  out->set_offset(in.offset() +
                  offset * in.strides()[axis] * SizeOf(out->dtype()));
}
template <typename T, typename Context>
void FlashAttnVarlenQKVPackedKernel(
    const Context& ctx,
    const DenseTensor& qkv,
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
    bool varlen_padded,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_FLASHATTN
  const auto head_groupnum = qkv.dims()[1];  // nheads/nheads_k + 1 + 1
  DenseTensor q, k, v;
  sliceFlattenView(qkv, &q, 1, 0, head_groupnum - 2);
  sliceFlattenView(qkv, &k, 1, head_groupnum - 2, 1);
  sliceFlattenView(qkv, &v, 1, head_groupnum - 1, 1);
  FlashAttnUnpaddedBaseKernel<T>(ctx,
                                 q,
                                 k,
                                 v,
                                 cu_seqlens_q,
                                 cu_seqlens_k,
                                 fixed_seed_offset,
                                 attn_mask,
                                 max_seqlen_q,
                                 max_seqlen_k,
                                 scale,
                                 dropout,
                                 causal,
                                 return_softmax,
                                 is_test,
                                 rng_name,
                                 out,
                                 softmax,
                                 softmax_lse,
                                 seed_offset,
                                 varlen_padded);
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnBaseKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const paddle::optional<DenseTensor>& attn_mask_start_row_indices,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    int attn_mask_start_row,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_FLASHATTN
  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(k.dims().size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(v.dims().size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(out->dims().size(),
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

  // TODO(umiswing): Add check shape

  const float softmax_scale = 1.0f / std::sqrt(head_size);
  const float softmax_unscale = std::sqrt(head_size);

  FlashAttnFwdParamsV2<T> params =
      FlashAttnFwdParamsV2<T>(ctx,
                              batch_size,
                              seqlen_q,
                              seqlen_k,
                              num_heads,
                              num_heads_k,
                              head_size,
                              dropout,
                              softmax_scale,
                              causal,
                              return_softmax,
                              q.dtype(),
                              is_test,
                              rng_name,
                              attn_mask_start_row,
                              fixed_seed_offset,
                              attn_mask,
                              attn_mask_start_row_indices,
                              softmax,
                              softmax_lse,
                              seed_offset);

  VLOG(10) << "[FlashAttn Forward] q.shape=[" << q.dims() << "], k.shape=["
           << k.dims() << "], v.shape=[" << v.dims() << "]";
  VLOG(10) << "[FlashAttn Forward] dropout=" << dropout
           << ", seed=" << params.seed << ", offset=" << params.offset;
  VLOG(10) << "[FlashAttn Forward] softmax_scale=" << softmax_scale
           << ", softmax_unscale=" << softmax_unscale;
  if (attn_mask.get_ptr()) {
    VLOG(10) << "[FlashAttn Forward] attn_mask.shape=["
             << (attn_mask.get_ptr())->dims() << "]";
  }
  if (!out->IsInitialized()) ctx.template Alloc<T>(out);

  cudaStream_t stream = ctx.stream();

  bool succ = phi::dynload::flash_attn_fwd(
      q.data(),
      k.data(),
      v.data(),
      params.rng_state.data(),
      out->data(),
      params.return_softmax ? params.softmax->data() : nullptr,
      params.softmax_lse->data(),
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
      params.softmax_scale,
      softmax_unscale,
      params.causal,
      params.return_softmax,
      params.is_bf16,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.mask_dims.data(),
      params.attn_mask_start_row_indices_tensor
          ? params.attn_mask_start_row_indices_tensor->data()
          : nullptr,
      params.attn_mask_start_row_indices_tensor
          ? params.attn_mask_start_row_indices_dims.data()
          : nullptr,
      params.attn_mask_start_row,
      q.strides()[1],
      k.strides()[1],
      v.strides()[1],
      q.strides()[2],
      k.strides()[2],
      v.strides()[2],
      out->strides()[1],
      out->strides()[2],
      q.strides()[0],
      k.strides()[0],
      v.strides()[0],
      out->strides()[0]);
  CheckFlashAttnStatus(succ);
#else
  RaiseNotSupportedError();
#endif
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
  FlashAttnBaseKernel<T, Context>(ctx,
                                  q,
                                  k,
                                  v,
                                  fixed_seed_offset,
                                  attn_mask,
                                  paddle::none,
                                  dropout,
                                  causal,
                                  return_softmax,
                                  is_test,
                                  rng_name,
                                  0,
                                  out,
                                  softmax,
                                  softmax_lse,
                                  seed_offset);
}

template <typename T, typename Context>
void FlashAttnQKVPackedKernel(
    const Context& ctx,
    const DenseTensor& qkv,
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
#ifdef PADDLE_WITH_FLASHATTN
  const auto head_groupnum = qkv.dims()[2];  // nheads/nheads_k + 1 + 1
  DenseTensor q, k, v;
  sliceFlattenView(qkv, &q, 2, 0, head_groupnum - 2);
  sliceFlattenView(qkv, &k, 2, head_groupnum - 2, 1);
  sliceFlattenView(qkv, &v, 2, head_groupnum - 1, 1);
  FlashAttnBaseKernel<T, Context>(ctx,
                                  q,
                                  k,
                                  v,
                                  fixed_seed_offset,
                                  attn_mask,
                                  paddle::none,
                                  dropout,
                                  causal,
                                  return_softmax,
                                  is_test,
                                  rng_name,
                                  0,
                                  out,
                                  softmax,
                                  softmax_lse,
                                  seed_offset);
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnWithSparseMaskKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& attn_mask_start_row_indices,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    float dropout,
    bool causal,
    int attn_mask_start_row,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
  FlashAttnBaseKernel<T, Context>(ctx,
                                  q,
                                  k,
                                  v,
                                  fixed_seed_offset,
                                  paddle::none,
                                  attn_mask_start_row_indices,
                                  dropout,
                                  causal,
                                  return_softmax,
                                  is_test,
                                  rng_name,
                                  attn_mask_start_row,
                                  out,
                                  softmax,
                                  softmax_lse,
                                  seed_offset);
}

template <typename T, typename Context>
void CalcReducedAttnScoresKernel(const Context& ctx,
                                 const DenseTensor& q,
                                 const DenseTensor& k,
                                 const DenseTensor& softmax_lse,
                                 const bool return_softmax,
                                 DenseTensor* reduced_scores,
                                 DenseTensor* softmax) {
#ifdef PADDLE_WITH_FLASHATTN
  if (!reduced_scores->IsInitialized())
    ctx.template Alloc<float>(reduced_scores);
  phi::funcs::SetConstant<Context, float> set_zero;
  set_zero(ctx, reduced_scores, 0.0f);
  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const int64_t batch_size = q.dims()[0];
  const int64_t seqlen_q = q.dims()[1];
  const int64_t num_heads = q.dims()[2];
  const int64_t head_size = q.dims()[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  const float softmax_scale = 1.0f / std::sqrt(head_size);
  const float softmax_unscale = std::sqrt(head_size);

  using Params = CalcReducedAttnScoresParams;

  Params params = Params(ctx,
                         batch_size,
                         seqlen_q,
                         seqlen_k,
                         num_heads,
                         num_heads_k,
                         head_size,
                         softmax_scale,
                         return_softmax,
                         q.dtype(),
                         softmax);

  cudaStream_t stream = ctx.stream();

  bool succ = phi::dynload::calc_reduced_attn_scores(
      q.data(),
      k.data(),
      softmax_lse.data(),
      reduced_scores->data(),
      params.return_softmax ? params.softmax->data() : nullptr,
      params.batch_size,
      params.max_seqlen_q,
      params.max_seqlen_k,
      params.num_heads,
      params.num_heads_k,
      params.head_size,
      params.softmax_scale,
      params.return_softmax,
      params.is_bf16,
      /*num_splits=*/0,
      stream,
      q.strides()[1],
      k.strides()[1],
      reduced_scores->strides()[1],
      q.strides()[2],
      k.strides()[2],
      reduced_scores->strides()[2],
      q.strides()[0],
      k.strides()[0],
      reduced_scores->strides()[0]);
  CheckFlashAttnStatus(succ);
#else
  RaiseNotSupportedError();
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_unpadded,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_KERNEL(flash_attn_varlen_qkvpacked,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnVarlenQKVPackedKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_KERNEL(flash_attn,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_KERNEL(flash_attn_qkvpacked,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnQKVPackedKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(1).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_KERNEL(flash_attn_with_sparse_mask,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnWithSparseMaskKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(4).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_KERNEL(calc_reduced_attn_scores,
                   GPU,
                   ALL_LAYOUT,
                   phi::CalcReducedAttnScoresKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
