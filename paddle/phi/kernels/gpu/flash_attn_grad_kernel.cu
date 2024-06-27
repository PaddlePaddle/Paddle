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
#include "paddle/phi/kernels/funcs/tensor_formatter.h"
#include "glog/logging.h"  // For VLOG()
#include "paddle/common/ddim.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/backends/gpu/musa/mudnn_helper.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/impl/tril_triu_kernel_impl.h"

PD_DECLARE_bool(cudnn_deterministic);

namespace phi {
using ScaledDotProductAttention =
    phi::backends::gpu::ScaledDotProductAttention;
using ScopedTensorDescriptor =
    phi::backends::gpu::ScopedTensorDescriptor;
using GPUDNNDataLayout = phi::backends::gpu::DataLayout;

inline bool is_pad_mask(const DenseTensor& mask, const DenseTensor& query) {
  return mask.dims().size() == 2 && mask.dims()[0] == query.dims()[0] &&
      mask.dims()[1] == query.dims()[2];
}

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      DenseTensor* out);

int get_num_split() {
  // 0 for an internal heuristic, which is optimal
  return FLAGS_cudnn_deterministic ? 1 : 0;
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
  void* dq_ptr = nullptr;
  void* dk_ptr = nullptr;
  void* dv_ptr = nullptr;

  ctx.template Alloc<T>(dq);
  dq_ptr = dq->data();

  DenseTensor dk_tmp;
  if (dk) {
    ctx.template Alloc<T>(dk);
    dk_ptr = dk->data();
  } else {
    dk_tmp = EmptyLike<T, Context>(ctx, k);
    dk_ptr = dk_tmp.data();
  }

  DenseTensor dv_tmp;
  if (dv) {
    ctx.template Alloc<T>(dv);
    dv_ptr = dv->data();
  } else {
    dv_tmp = EmptyLike<T, Context>(ctx, v);
    dv_ptr = dv_tmp.data();
  }

  const cudaStream_t stream = ctx.stream();

  // q,k,v [total_*, num_heads, head_dim]
  auto dims = q.dims();

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size_og = dout.dims()[2];
  const int64_t head_size = dims[2];
  const int64_t num_heads_k = k.dims()[1];

  int num_splits = get_num_split();

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
                           attn_mask,
                           seed_offset.data<int64_t>());

  VLOG(10) << "FlashAttn bwd seed: " << params.seed
           << ", offset: " << params.offset;

  bool succ = phi::dynload::flash_attn_varlen_bwd(
      dout.data(),
      q.data(),
      k.data(),
      v.data(),
      out.data(),
      params.softmax_d.data(),
      softmax_lse.data(),
      cu_seqlens_q.data<int32_t>(),
      cu_seqlens_k.data<int32_t>(),
      params.rng_state.data(),
      dq_ptr,
      dk_ptr,
      dv_ptr,
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
      params.softmax_scale,
      1.0f / params.softmax_scale,
      params.causal,
      params.is_bf16,
      num_splits,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr);
  CheckFlashAttnStatus(succ);
#elif defined(PADDLE_WITH_FLASHATTN_MUSA)
  RaiseNotSupportedError();
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
#if defined(PADDLE_WITH_FLASHATTN_MUSA)
  ctx.template Alloc<T>(dq);
  ctx.template Alloc<T>(dk);
  ctx.template Alloc<T>(dv);
  ScaledDotProductAttention sdpa;  
  
  DenseTensor trans_q(q);
  trans_q.transpose_dim1_and_dim2_for_4d_tensor();

  DenseTensor trans_k(k);
  trans_k.transpose_dim1_and_dim2_for_4d_tensor();

  DenseTensor trans_v(v);
  trans_v.transpose_dim1_and_dim2_for_4d_tensor();

  //[batch_size, seq_len, num_heads, head_dim] in paddle, but [bs, num, seq_len, hd] in mudnn and pytorch
  //dq_temp, dk_temp, dv_temp are used for mudnn, in the end they will be transposed and copy to dq, dk, dv
  DenseTensor dq_temp(*dq);
  dq_temp.transpose_dim1_and_dim2_for_4d_tensor();

  DenseTensor dk_temp(*dk);
  dk_temp.transpose_dim1_and_dim2_for_4d_tensor();

  DenseTensor dv_temp(*dv);
  dv_temp.transpose_dim1_and_dim2_for_4d_tensor();

  DenseTensor trans_dout(dout);
  trans_dout.transpose_dim1_and_dim2_for_4d_tensor();

  DenseTensor trans_out(out);
  trans_out.transpose_dim1_and_dim2_for_4d_tensor();

  ScopedTensorDescriptor con_mask_scoped_desc;
  DenseTensor con_mask;
  dynload::Tensor musa_mask;

  if(attn_mask.get_ptr() && !causal){
    sdpa.desc_.SetMaskMode(is_pad_mask(*attn_mask.get_ptr(), trans_q));
    con_mask.Resize(attn_mask.get_ptr()->dims());
    phi ::ContiguousKernel<T, Context>(
          ctx, *attn_mask.get_ptr(), &con_mask);
    musa_mask = con_mask_scoped_desc.descriptor<T>(
          con_mask, GPUDNNDataLayout::kNCHW, common::vectorize<int>(con_mask.dims()));  
  }else{
    con_mask.Resize(common::make_ddim({0}));
    ctx.template Alloc<float>(&con_mask);
    musa_mask = con_mask_scoped_desc.descriptor<float>(
          con_mask, GPUDNNDataLayout::kNCHW, common::vectorize<int>(con_mask.dims()));      
  }

  auto q_head_dim = trans_q.dims()[3]; // head_dim
  auto q_seq_len = trans_q.dims()[2]; // seq_len
  auto q_head_num = trans_q.dims()[1]; // head_num
  auto batch_size = trans_q.dims()[0]; // batch_size
  auto kv_seq_len = trans_k.dims()[2];
  sdpa.desc_.SetEmbedDim(q_head_num * q_head_dim);
  sdpa.desc_.SetHeadsNum(q_head_num);
  sdpa.desc_.SetTraining(true);
  sdpa.desc_.SetCausal(causal);

  DenseTensor dropout_mask;
  dropout_mask.Resize(common::make_ddim({0}));
  ctx.template Alloc<T>(&dropout_mask);

  ScopedTensorDescriptor out_scoped_desc;
  ScopedTensorDescriptor dq_temp_scoped_desc;
  ScopedTensorDescriptor dk_temp_scoped_desc;
  ScopedTensorDescriptor dv_temp_scoped_desc;
  ScopedTensorDescriptor trans_q_scoped_desc;
  ScopedTensorDescriptor trans_k_scoped_desc;
  ScopedTensorDescriptor trans_v_scoped_desc;
  ScopedTensorDescriptor logsumexp_scoped_desc;
  ScopedTensorDescriptor dout_scoped_desc;
  ScopedTensorDescriptor dropout_mask_scoped_desc;

  auto& musa_output = out_scoped_desc.descriptor_with_stride<T>(
      trans_out, GPUDNNDataLayout::kNCHW, common::vectorize<int>(trans_out.dims()));
  auto& musa_grad_query = dq_temp_scoped_desc.descriptor_with_stride<T>(
      dq_temp, GPUDNNDataLayout::kNCHW, common::vectorize<int>(dq_temp.dims()));
  auto& musa_grad_key = dk_temp_scoped_desc.descriptor_with_stride<T>(
      dk_temp, GPUDNNDataLayout::kNCHW, common::vectorize<int>(dk_temp.dims()));
  auto& musa_grad_value = dv_temp_scoped_desc.descriptor_with_stride<T>(
      dv_temp, GPUDNNDataLayout::kNCHW, common::vectorize<int>(dv_temp.dims()));
  auto& musa_q = trans_q_scoped_desc.descriptor_with_stride<T>(
      trans_q, GPUDNNDataLayout::kNCHW, common::vectorize<int>(trans_q.dims()));
  auto& musa_k = trans_k_scoped_desc.descriptor_with_stride<T>(
      trans_k, GPUDNNDataLayout::kNCHW, common::vectorize<int>(trans_k.dims()));
  auto& musa_v = trans_v_scoped_desc.descriptor_with_stride<T>(
      trans_v, GPUDNNDataLayout::kNCHW, common::vectorize<int>(trans_v.dims()));
  auto& musa_logsumexp = logsumexp_scoped_desc.descriptor_with_stride<float>(
      softmax_lse, GPUDNNDataLayout::kNCHW, common::vectorize<int>(softmax_lse.dims()));
  auto& musa_grad_output = dout_scoped_desc.descriptor_with_stride<T>(
      trans_dout, GPUDNNDataLayout::kNCHW, common::vectorize<int>(trans_dout.dims()));
  auto& musa_dropout_mask = dropout_mask_scoped_desc.descriptor_with_stride<T>(
      dropout_mask, GPUDNNDataLayout::kNCHW, common::vectorize<int>(dropout_mask.dims()));
  

  sdpa.desc_.RunFlashBwd(
      *ctx.cudnn_handle(),
      musa_grad_query,
      musa_grad_key,
      musa_grad_value,
      musa_grad_output,
      musa_q,
      musa_k,
      musa_v,
      musa_mask,
      musa_output,
      musa_logsumexp,
      musa_dropout_mask,
      phi::backends::gpu::InternalMemAlloc);


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
