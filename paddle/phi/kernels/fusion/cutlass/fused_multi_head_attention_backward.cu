// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/malloc.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/dropout_impl_util.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/kernels/cutlass_fmha_backward.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

struct LaunchParams {
  // meta params
  phi::DataType datatype;
  bool preload_mems;

  // Create Input tensors in BMHK format, where
  // B = batch_size
  // M = sequence length
  // H = num_heads
  // K = embedding size per head

  const void* query_ptr;
  const void* key_ptr;
  const void* value_ptr;
  const void* bias_ptr;
  void* grad_bias_ptr = nullptr;
  void* logsumexp_ptr;  // [num_batches, num_heads, num_queries] - can be null
  const void* output_ptr;        // [num_batches, query_seq_len, num_heads, head_size]
  const void* grad_output_ptr;  // [num_batches, query_seq_len, num_heads, head_size]
  float* delta_ptr; // [nH, Mq]
    // Output tensors
  void* grad_query_ptr; //  [Mq, nH, K]
  void* grad_key_ptr; //    [Mk, nH, K]
  void* grad_value_ptr; //  [Mk, nH, Kv]

  // Dimensions/strides
  int32_t num_heads;
  int32_t query_seq_len;
  int32_t key_value_seq_len;
  int32_t head_size;
  int32_t value_head_size;

  // Scale
  float scale;
  bool causal;

  int8_t gQKV_strideM_multiplier; // 3 for packed, 1 otherwise
  
  uint64_t* seed_and_offset;
  
  int32_t* cu_seqlens_q_ptr = nullptr;
  int32_t* cu_seqlens_k_ptr = nullptr;

  float dropout_prob;
  void* workspace = nullptr;

  //@TODO zhangdanyang : reorder strideM
  int32_t q_strideM;
  int32_t k_strideM;
  int32_t v_strideM;
  int32_t bias_strideM = 0;
  int32_t gO_strideM;
  int32_t gB_strideM;

  /*
  We can understand the computation of Fused Multihead Attention in this way:
  for Query matmul Key, we execute num_batches * num_heads times matmul,
  each matmul problem is: (M, K) (K, N) -> (M, N).
  Here M is: query_seq_len, K is: head_size, N is: key_value_seq_len.
  The stride concept is equals to torch's, it means the offset to move to next
  axis. For Q matrix(M, K), we need to move K(which equals to head_size) offset
  to next row(in M axis).
  */
    int64_t o_strideH;
    int32_t q_strideH;
    int32_t k_strideH;
    int32_t v_strideH;
    int32_t bias_strideH = 0;
    int64_t lse_strideH;
    int64_t delta_strideH;
    int32_t num_batches;

    int64_t gO_strideH;
    int64_t gQ_strideH;
    int64_t gK_strideH;
    int64_t gV_strideH;
    int64_t gB_strideH;
};

template <
    typename ArchTag_,
    typename T,
    bool kIsAligned_,
    bool kApplyDropout_,
    bool kPreloadMmas_,
    int kBlockSizeI_,
    int kBlockSizeJ_,
    // upperbound on `max(value.shape[-1], query.shape[-1])`
    int kMaxK_>
void LaunchMultiHeadAttentionBackwardKernel(LaunchParams params,
                                    const phi::GPUContext& ctx
                                    ){ 
  using AttentionBackward = AttentionBackwardKernel<ArchTag_,
                                    T,
                                    kIsAligned_,
                                    kApplyDropout_,
                                    kPreloadMmas_,
                                    kBlockSizeI_,
                                    kBlockSizeJ_,
                                    kMaxK_>;

  typename AttentionBackward::Params p;
  {  // set parameters
    p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.query_ptr));
    p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.key_ptr));
    p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.value_ptr));
    p.bias_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.bias_ptr));;
    p.grad_bias_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.grad_bias_ptr));
    p.logsumexp_ptr = nullptr;
    p.output_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.output_ptr));
    p.grad_output_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.grad_output_ptr));
    p.grad_query_ptr = reinterpret_cast<T*>(params.grad_query_ptr);
    p.grad_key_ptr = reinterpret_cast<T*>(params.grad_key_ptr);
    p.grad_value_ptr = reinterpret_cast<T*>(params.grad_value_ptr);
    p.delta_ptr = (params.delta_ptr);

    p.num_batches = params.num_batches;
    p.num_heads = params.num_heads;
    p.num_queries = params.query_seq_len;
    p.num_keys = params.key_value_seq_len;
    p.head_dim = params.head_size;
    p.head_dim_value = params.value_head_size;

    p.scale = params.scale;
    p.causal = params.causal;

    //@TODO zhangdanyang: double check the param
    p.gQKV_strideM_multiplier = 1;

    p.seed_and_offset = params.seed_and_offset;

    p.o_strideH = params.o_strideH;
    p.q_strideH = params.q_strideH;
    p.k_strideH = params.k_strideH;
    p.v_strideH = params.v_strideH;
    p.bias_strideH = params.bias_strideH;
    p.lse_strideH = params.lse_strideH;
    p.delta_strideH = params.delta_strideH;
    p.gO_strideH = params.gO_strideH;
    p.gQ_strideH = params.gQ_strideH;
    p.gK_strideH = params.gK_strideH;
    p.gV_strideH = params.gV_strideH;
    p.gB_strideH = params.gB_strideH;

    p.q_strideM = p.q_strideH * params.num_heads;
    p.k_strideM = p.k_strideH * params.num_heads;
    p.v_strideM = p.v_strideH * params.num_heads;
    p.bias_strideM = params.bias_strideH * params.num_heads;
    p.gO_strideM = p.gO_strideH * params.num_heads;
    p.gB_strideM = p.gB_strideH * params.num_heads;

    p.gO_strideB = p.gO_strideM * params.query_seq_len;
    p.gQ_strideB = p.gQ_strideH * params.num_heads * params.query_seq_len;
    p.gK_strideB = p.gK_strideH * params.num_heads * params.query_seq_len;
    p.gV_strideB = p.gV_strideH * params.num_heads * params.query_seq_len;
    p.o_strideB = p.o_strideH * params.num_heads * params.query_seq_len;
    p.q_strideB = p.q_strideM * params.query_seq_len;
    p.k_strideB = p.k_strideM * params.key_value_seq_len;
    p.v_strideB = p.v_strideM * params.key_value_seq_len;
    p.bias_strideB = p.bias_strideM * params.query_seq_len;
    p.lse_strideB = p.lse_strideH * params.num_heads * params.query_seq_len;
    p.delta_strideB = p.delta_strideH * params.num_heads * params.query_seq_len;
    p.gB_strideB = p.gB_strideM * params.query_seq_len;

    p.dropout_prob = params.dropout_prob;
    p.workspace = const_cast<float*>(reinterpret_cast<const float*>(params.workspace));
  }

  constexpr auto kernel_fn = attention_kernel_backward_batched<AttentionBackward>;
  int smem_bytes = sizeof(typename AttentionBackward::SharedStorage);
  if (smem_bytes > 0xc000) {
    cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }

  if (!AttentionBackward::check_supported(p)) {
    PADDLE_THROW(
        phi::errors::Unimplemented("The Params is not supported by cutlass "
                                   "fused multihead attention backward. "));
    return;
  }
  kernel_fn<<<p.getBlocksGrid(),
              p.getThreadsGrid(),
              smem_bytes,
              ctx.stream()>>>(p);
}

template <typename ArchTag_,
          typename scalar_t_,
          bool kIsAligned_,
          bool kApplyDropout_,
          bool kPreloadMmas_,
          int kBlockSizeI_,
          int kBlockSizeJ_>
void DispatchKMaxK(LaunchParams params, const phi::GPUContext& ctx) {
  // @TODO zhangdanyang: double check the KMaxK_
  LaunchMultiHeadAttentionBackwardKernel<ArchTag_,
                                scalar_t_,
                                kIsAligned_,
                                kApplyDropout_,
                                kPreloadMmas_,
                                kBlockSizeI_,
                                kBlockSizeJ_,
                                // upperbound on `max(value.shape[-1], query.shape[-1])`
                                128>(params, ctx);}

template <typename ArchTag_,
          typename scalar_t_,
          bool kIsAligned_,
          bool kApplyDropout_,
          bool kPreloadMmas_>
void DispatchFMHABlockSize(LaunchParams params, const phi::GPUContext& ctx) {
    DispatchKMaxK<ArchTag_, scalar_t_, kIsAligned_, kApplyDropout_, kPreloadMmas_, 128, 64>
                                                (params, ctx);
}

template <typename ArchTag_,
          typename scalar_t_,
          bool kIsAligned_,
          bool kApplyDropout_>
void DispatchFMHAPreload(LaunchParams params,
                                      const phi::GPUContext& ctx) {
  if (params.preload_mems) {
    DispatchFMHABlockSize<ArchTag_, scalar_t_, kIsAligned_, kApplyDropout_,
                        true>(params, ctx);
  } else {
    DispatchFMHABlockSize<ArchTag_, scalar_t_, kIsAligned_, kApplyDropout_,
                        false>(params, ctx);
    }
  }

template <typename ArchTag_,
          typename scalar_t_,
          bool kIsAligned_>
void DispatchFMHADropout(LaunchParams params,
                                      const phi::GPUContext& ctx) {
  if (params.dropout_prob>0) {
    DispatchFMHAPreload<ArchTag_, scalar_t_, kIsAligned_, true>(params, ctx);
  } else {
    DispatchFMHAPreload<ArchTag_, scalar_t_, kIsAligned_, false>(params, ctx);
    }
  }

template <typename ArchTag_, typename T>
void DispatchFMHAIsAligned(LaunchParams params, const phi::GPUContext& ctx) {
  if (reinterpret_cast<uintptr_t>(params.query_ptr) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(params.key_ptr) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(params.value_ptr) % 16 == 0 &&
      params.q_strideH % (16 / sizeof(T)) == 0 &&
      params.q_strideH % (16 / sizeof(T)) == 0 &&
      params.v_strideH % (16 / sizeof(T)) == 0) {
    DispatchFMHADropout<ArchTag_, T, true>(params, ctx);
  } else {
    // TODO(zhengzekang): Currently we assume that the input is aligned in
    // order to reduce templates num. DispatchFMHABlockSize<T, ArchTag,
    // false>(params, ctx);
    PADDLE_THROW(phi::errors::Unimplemented(
        "Currently cutlass fused multihead attention kernel only support "
        "aligned data. "
        "Please Check `query`, `key`, `value` is aligned with 128bit "
        "and final dim can be divide by %d. ",
        16 / sizeof(T)));
    return;
  }
}

template <typename ArchTag_>
void DispatchFusedMultiheadAttentionKernel(LaunchParams params,
                                           const phi::GPUContext& ctx) {
  if (params.datatype == DataType::FLOAT16) {
    return DispatchFMHAIsAligned<ArchTag_, cutlass::half_t>(params, ctx);
  } 
  else if (params.datatype == DataType::BFLOAT16) {
    return DispatchFMHAIsAligned<ArchTag_, cutlass::bfloat16_t>(params, ctx);
  } 
  else {
    return DispatchFMHAIsAligned<ArchTag_, float>(params, ctx);
  }
}

void DispatchFMHAArchTag(LaunchParams params, const phi::GPUContext& ctx) {
  const int compute_capability = ctx.GetComputeCapability();
  if (compute_capability == 80) {
    DispatchFusedMultiheadAttentionKernel<cutlass::arch::Sm80>(params, ctx);
  // } else if (compute_capability == 75) {
  //   DispatchFusedMultiheadAttentionKernel<cutlass::arch::Sm75>(params, ctx);
  // } else if (compute_capability == 70) {
  //   DispatchFusedMultiheadAttentionKernel<cutlass::arch::Sm70>(params, ctx);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Currently cutlass fused multihead attention kernel "
        "only support arch: SM80, SM75, SM70"));
    return;
  }
}

template <typename T, typename Context>
void MultiHeadAttentionBackwardKernel(const Context& ctx,
                                     const DenseTensor& query,
                                     const DenseTensor& key,
                                     const DenseTensor& value,
                                     const DenseTensor& seed_and_offset,
                                     const DenseTensor& out,
                                     const DenseTensor& out_grad,
                                     const bool causal,
                                     const float scale,
                                     const float dropout_p,
                                     DenseTensor* query_grad,
                                     DenseTensor* key_grad,
                                     DenseTensor* value_grad) {
  ctx.template Alloc<T>(query_grad);
  ctx.template Alloc<T>(key_grad);
  ctx.template Alloc<T>(value_grad);
  LaunchParams params{};

  bool is_half = query.dtype()==paddle::experimental::DataType::FLOAT16 
                                || query.dtype()==paddle::experimental::DataType::BFLOAT16;
  bool sm_range = ctx.GetComputeCapability() == 80;
  params.preload_mems = is_half && sm_range;

  params.datatype = query.dtype();
  params.query_ptr = query.data();
  params.key_ptr = key.data();
  params.value_ptr = value.data();
  params.bias_ptr = nullptr; //@TODO zhangdanyang: insert bias or not

  // TODO(zhengzekang): currently we only used in inference. Maybe add a bool
  // flag to save it ?
  params.logsumexp_ptr = nullptr;
  params.output_ptr = out.data();
  params.grad_output_ptr = out_grad.data();

  auto delta = phi::Empty<float, Context>(ctx, {query.dims()[0], query.dims()[2], query.dims()[1]});
  params.delta_ptr = reinterpret_cast<float*>(delta.data());

  params.cu_seqlens_q_ptr = nullptr;
  params.cu_seqlens_k_ptr = nullptr;

  params.grad_query_ptr = query_grad->data();
  params.grad_key_ptr = key_grad->data();
  params.grad_value_ptr = value_grad->data();
  params.grad_bias_ptr = nullptr;

  params.num_batches = query.dims()[0];
  params.head_size = query.dims()[3];
  params.value_head_size = value.dims()[3];
  params.query_seq_len = query.dims()[1];
  params.key_value_seq_len = key.dims()[0];
  params.num_heads = query.dims()[2];

  params.scale = scale;
  params.causal = causal;

  params.seed_and_offset = const_cast<uint64_t*>(reinterpret_cast<const uint64_t*>(seed_and_offset.data()));

  params.o_strideH = query.dims()[3];
  params.q_strideH = query.dims()[3];
  params.k_strideH = key.dims()[3];
  params.v_strideH = value.dims()[3];
  params.bias_strideH = query.dims()[3];
  params.lse_strideH = query.dims()[3];
  params.delta_strideH = query.dims()[3];
  params.gO_strideH = query.dims()[3];
  params.gQ_strideH = query.dims()[3];
  params.gK_strideH = key.dims()[3];
  params.gV_strideH = value.dims()[3];
  params.gB_strideH = query.dims()[3];
  
  params.dropout_prob = dropout_p;
  params.workspace = nullptr;

  DispatchFMHAArchTag(params, ctx);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

// PD_REGISTER_KERNEL(
//     fused_multihead_attention_grad,
//     GPU,
//     ALL_LAYOUT,
//     phi::fusion::cutlass_internal::MultiHeadAttentionBackwardKernel,
//     phi::dtype::float16) {}

PD_REGISTER_KERNEL(
    fused_multihead_attention_grad,
    GPU,
    ALL_LAYOUT,
    phi::fusion::cutlass_internal::MultiHeadAttentionBackwardKernel,
    float) {}

// PD_REGISTER_KERNEL(
//     fused_multihead_attention_grad,
//     GPU,
//     ALL_LAYOUT,
//     phi::fusion::cutlass_internal::MultiHeadAttentionBackwardKernel,
//     phi::dtype::bfloat16) {}