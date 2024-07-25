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

#include "glog/logging.h"  // For VLOG()
#include "paddle/common/macros.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/tril_triu_kernel_impl.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/funcs/tensor_formatter.h"
#include "paddle/phi/backends/gpu/musa/mudnn_helper.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include <chrono>

namespace phi {

static void InternalMemFree_flash_attn_fwd(void* ptr) {

}



inline bool is_pad_mask(const DenseTensor& mask, const DenseTensor& query) {
  return mask.dims().size() == 2 && mask.dims()[0] == query.dims()[0] &&
      mask.dims()[1] == query.dims()[2];
}

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      DenseTensor* out);
using ScaledDotProductAttention =
    phi::backends::gpu::ScaledDotProductAttention;
using ScopedTensorDescriptor =
    phi::backends::gpu::ScopedTensorDescriptor;
using GPUDNNDataLayout = phi::backends::gpu::DataLayout;
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
  ctx.template Alloc<T>(out);

  cudaStream_t stream = ctx.stream();

  // q, k, v [total_q/k/v, num_heads, head_dim]
  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size = dims[2];
  const int64_t num_heads_k = k.dims()[1];

  // TODO(umiswing): add shape check

  FlashAttnFwdParamsV2<T> params = FlashAttnFwdParamsV2<T>(ctx,
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
                                                           fixed_seed_offset,
                                                           attn_mask,
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
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr);
  CheckFlashAttnStatus(succ);
#elif defined(PADDLE_WITH_FLASHATTN_MUSA)
  RaiseNotSupportedError();
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
#if defined(PADDLE_WITH_FLASHATTN_MUSA)
  if(UNLIKELY(return_softmax)){
    PADDLE_ENFORCE_EQ((dropout>0.0f),true,"return_softmax is only supported when dropout > 0.0");
    PADDLE_ENFORCE_EQ(0,1,"not support");
  }
  PADDLE_ENFORCE_EQ(
      q.dims().size() == 4 && k.dims().size() == 4 && v.dims().size() == 4,true,
      "Expect all query, key, value has 4D shape!");

  DenseTensor trans_q(q);
  trans_q.transpose_dim1_and_dim2_for_4d_tensor();

  DenseTensor trans_k(k);
  trans_k.transpose_dim1_and_dim2_for_4d_tensor();

  DenseTensor trans_v(v);
  trans_v.transpose_dim1_and_dim2_for_4d_tensor();

  PADDLE_ENFORCE_EQ(trans_q.dims()[3],
                    trans_k.dims()[3],
                    phi::errors::InvalidArgument(
                        "head_dim of q must be equal to head_dim of k"
                        "but they are %d and %d",trans_q.dims()[3],trans_k.dims()[3]));
  PADDLE_ENFORCE_EQ(trans_k.dims()[2],
                    trans_v.dims()[2],
                    phi::errors::InvalidArgument(
                        "seq_len of k must be equal to seq_len of v"
                        "but they are %d and %d",trans_k.dims()[2],trans_v.dims()[2]));  

  auto head_dim_q = trans_q.dims()[3]; // head_dim
  auto q_seq_len = trans_q.dims()[2]; // seq_len
  auto head_num_q = trans_q.dims()[1]; // head_num
  auto batch_size_ = trans_q.dims()[0]; // batch_size
  auto head_dim_v = trans_v.dims()[3];
  auto seqlen_k = trans_k.dims()[2];

  ctx.template Alloc<T>(out);
  DenseTensor out_temp(*out);
  out_temp.transpose_dim1_and_dim2_for_4d_tensor();

  DenseTensor con_mask;
  ScopedTensorDescriptor con_mask_scoped_desc;
  dynload::Tensor con_mask_desc;
  if(attn_mask.get_ptr() && !causal){
    const auto attn_mask_dim_size=attn_mask.get_ptr()->dims().size();
    PADDLE_ENFORCE_EQ(
        attn_mask_dim_size,
        4,
        phi::errors::InvalidArgument(
            "The number of dimensions of attn_mask is expected to be "
            "equal to 4, but recieved %d. The shape of attn_mask is {%s}",
            attn_mask_dim_size,
            attn_mask.get_ptr()->dims()));

    PADDLE_ENFORCE_EQ(attn_mask.get_ptr()->dims()[3]==seqlen_k && \
        attn_mask.get_ptr()->dims()[2]==q_seq_len,true, 
        "the last two dim of attn_mask(%d , %d) should be equal to seqlen_q(%d),seqlen_k(%d)",
        attn_mask.get_ptr()->dims()[2], attn_mask.get_ptr()->dims()[3],q_seq_len,seqlen_k);

    PADDLE_ENFORCE_EQ(attn_mask.get_ptr()->dims()[1]==head_num_q || attn_mask.get_ptr()->dims()[1]==1,true, 
        "mask_dims[1] == 1 || mask_dims[1] == num_heads, but mask_dims[1] is %d, and num_heads_q is %d",attn_mask.get_ptr()->dims()[1],head_num_q);

    PADDLE_ENFORCE_EQ(attn_mask.get_ptr()->dims()[0]==batch_size_,true, 
        "mask_dims[0] == batch_size, but mask_dims[0] is %d, and batch_size is %d",attn_mask.get_ptr()->dims()[0],batch_size_);
    con_mask.Resize(attn_mask.get_ptr()->dims());
    phi ::ContiguousKernel<T, Context>(
          ctx, *attn_mask.get_ptr(), &con_mask); 
    con_mask_desc = con_mask_scoped_desc.descriptor<T>(con_mask,
                                GPUDNNDataLayout::kNCHW,
                                common::vectorize<int>(con_mask.dims()));  
    
  }else{
    con_mask.Resize(common::make_ddim({0}));
    ctx.template Alloc<float>(&con_mask);   
    con_mask_desc = con_mask_scoped_desc.descriptor<float>(con_mask,
                            GPUDNNDataLayout::kNCHW,
                            common::vectorize<int>(con_mask.dims()));   
  }

  std::vector<int64_t> softmax_lse_dims={batch_size_, head_num_q, q_seq_len};
  softmax_lse->Resize(phi::make_ddim(softmax_lse_dims));
  ctx.template Alloc<float>(softmax_lse);

  ScaledDotProductAttention sdpa;
  if (causal) {
    sdpa.desc_.SetCausal(true);  
  }  
  sdpa.desc_.SetEmbedDim(head_dim_q * head_num_q);
  sdpa.desc_.SetHeadsNum(head_num_q);
  if(attn_mask.get_ptr()){
    sdpa.desc_.SetMaskMode(is_pad_mask(*attn_mask.get_ptr(), trans_q));
  }

  DenseTensor dropout_mask;


  if(UNLIKELY(dropout > 0.0)){
    PADDLE_ENFORCE_EQ(0,1,"Flash Attention 2 Not Support Dropout Now");
  }else{
    dropout_mask.Resize(common::make_ddim({0}));
    ctx.template Alloc<T>(&dropout_mask);
  }

  auto handle = ctx.cudnn_handle();

  ScopedTensorDescriptor out_scoped_desc;
  ScopedTensorDescriptor trans_q_scoped_desc;
  ScopedTensorDescriptor trans_k_scoped_desc;
  ScopedTensorDescriptor trans_v_scoped_desc;
  ScopedTensorDescriptor logsumexp_scoped_desc;
  ScopedTensorDescriptor dropout_mask_scoped_desc;


  auto& out_desc = out_scoped_desc.descriptor_with_stride<T>(
      out_temp, GPUDNNDataLayout::kNCHW, common::vectorize<int>(out_temp.dims()));
  auto& logsumexp_desc =
      logsumexp_scoped_desc.descriptor_with_stride<float>(*softmax_lse,
                                      GPUDNNDataLayout::kNCHW,
                                      common::vectorize<int>(softmax_lse->dims()));

  auto& trans_q_desc =
      trans_q_scoped_desc.descriptor_with_stride<T>(trans_q,
                                      GPUDNNDataLayout::kNCHW,
                                      common::vectorize<int>(trans_q.dims()));     
  auto& trans_k_desc =
      trans_k_scoped_desc.descriptor_with_stride<T>(trans_k,
                                      GPUDNNDataLayout::kNCHW,
                                      common::vectorize<int>(trans_k.dims()));     
  auto& trans_v_desc =
      trans_v_scoped_desc.descriptor_with_stride<T>(trans_v,
                                      GPUDNNDataLayout::kNCHW,
                                      common::vectorize<int>(trans_v.dims()));     
  auto& dropout_mask_desc =
      dropout_mask_scoped_desc.descriptor_with_stride<T>(dropout_mask,
                                      GPUDNNDataLayout::kNCHW,
                                      common::vectorize<int>(dropout_mask.dims()));    

  Allocator::AllocationPtr memory_for_mudnn; //this is a unique ptr so the memory it holds will be free when it is out of its scope

  auto InternalMemAlloc_flash_attn_fwd = [&memory_for_mudnn, &ctx](size_t s) {
    memory_for_mudnn = std::move(phi::memory_utils::Alloc(ctx.GetPlace(),s,phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream()))));
    return dynload::MemoryHandler(memory_for_mudnn->ptr(), InternalMemFree_flash_attn_fwd);
  };

  sdpa.desc_.RunFlash(*handle, out_desc, logsumexp_desc, trans_q_desc, trans_k_desc, trans_v_desc,
   con_mask_desc, dropout_mask_desc, InternalMemAlloc_flash_attn_fwd);
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

PD_REGISTER_KERNEL(flash_attn,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}
