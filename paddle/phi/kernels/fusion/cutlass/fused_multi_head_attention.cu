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
#include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/kernels/cutlass_fmha_forward.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

struct LaunchParams {
  // meta params
  phi::DataType datatype;

  // Create Input tensors in BMHK format, where
  // B = batch_size
  // M = sequence length
  // H = num_heads
  // K = embedding size per head

  const void* query_ptr;
  const void* key_ptr;
  const void* value_ptr;
  // Mask Tensor format is BHMK
  // and it can be broadcasted in axis0, 1, 2.
  const void* mask_ptr = nullptr;

  int32_t* cu_seqlens_q_ptr = nullptr;
  int32_t* cu_seqlens_k_ptr = nullptr;

  // Output tensors
  void* output_ptr;        // [num_batches, query_seq_len, num_heads, head_size]
  void* output_accum_ptr;  // [num_batches, query_seq_len, num_heads, head_size]
  void* logsumexp_ptr;  // [num_batches, num_heads, num_queries] - can be null

  // Scale
  float scale;

  // Dimensions/strides
  int32_t num_batches;
  int32_t num_heads;
  int32_t query_seq_len;
  int32_t key_value_seq_len;
  int32_t head_size;
  int32_t value_head_size;
  bool causal;
  bool mask_broadcast_row;
  /*
  We can understand the computation of Fused Multihead Attention in this way:
  for Query matmul Key, we execute num_batches * num_heads times matmul,
  each matmul problem is: (M, K) (K, N) -> (M, N).
  Here M is: query_seq_len, K is: head_size, N is: key_value_seq_len.
  The stride concept is equals to torch's, it means the offset to move to next
  axis. For Q matrix(M, K), we need to move K(which equals to head_size) offset
  to next row(in M axis).
  */
  int32_t query_strideH;
  int32_t key_strideH;
  int32_t value_strideH;

  // Since mask can be broadcasted, we need to assign each stride
  int32_t mask_strideM;
  int64_t mask_strideH;  // stride for num_heads
  int64_t mask_strideB;  // stride for num_batches

  // dropout
  bool use_dropout;
  unsigned long long dropout_batch_head_rng_offset;
  float dropout_prob;

  uint64_t* seed_and_offset;
};

template <typename T,
          typename ArchTag,
          bool IsAligned,
          int QueriesPerBlock,
          int KeysPerBlock,
          bool SingleValueIteration,
          bool AddMask,
          bool MaskBroadcastRow,
          bool kSupportsDropout_>
void LaunchMultiHeadAttentionKernel(LaunchParams params,
                                    const phi::GPUContext& ctx) {
  using Attention = AttentionKernel<T,
                                    ArchTag,
                                    IsAligned,
                                    QueriesPerBlock,
                                    KeysPerBlock,
                                    SingleValueIteration,
                                    AddMask,
                                    MaskBroadcastRow,
                                    kSupportsDropout_>;

  typename Attention::Params p;
  {  // set parameters
    p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.query_ptr));
    p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.key_ptr));
    p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.value_ptr));
    p.mask_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.mask_ptr));

    // TODO(zhengzekang): Currently we only support inference, so here we set
    // `logsumexp_ptr` as nullptr, which is used for backward.
    p.logsumexp_ptr = nullptr;;

    p.output_accum_ptr = nullptr;
    if (Attention::kNeedsOutputAccumulatorBuffer) {
      const int64_t output_size = params.num_batches * params.num_heads *
                                  params.query_seq_len * params.head_size;
      paddle::memory::AllocationPtr tmp_output_accum_buffer_ptr{nullptr};
      tmp_output_accum_buffer_ptr = paddle::memory::Alloc(
          ctx.GetPlace(),
          output_size * sizeof(typename Attention::output_accum_t),
          phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
      p.output_accum_ptr =
          reinterpret_cast<typename Attention::output_accum_t*>(
              tmp_output_accum_buffer_ptr->ptr());
    }

    p.output_ptr = reinterpret_cast<T*>(params.output_ptr);

    // TODO(zhengzekang): support arbitrary seq lengths
    // if (cu_seqlens_q.has_value()) {
    //   p.cu_seqlens_q_ptr = (int32_t*)cu_seqlens_q->data_ptr();
    //   p.cu_seqlens_k_ptr = (int32_t*)cu_seqlens_k->data_ptr();
    // }

    p.num_batches = params.num_batches;
    p.num_heads = params.num_heads;
    p.num_queries = params.query_seq_len;
    p.num_keys = params.key_value_seq_len;
    p.head_dim = params.head_size;
    p.head_dim_value = params.value_head_size;

    p.scale = params.scale;
    p.causal = params.causal;
    p.mask_broadcast_row = params.mask_broadcast_row;

    p.seed_and_offset = params.seed_and_offset;

    // dropout
    p.use_dropout = params.use_dropout;
    p.dropout_batch_head_rng_offset = params.dropout_batch_head_rng_offset;
    p.dropout_prob = params.dropout_prob;

    // TODO(zhengzekang): This might overflow for big tensors
    p.q_strideH = params.query_strideH;
    p.k_strideH = params.key_strideH;
    p.v_strideH = params.value_strideH;
    p.mask_strideH = params.mask_strideH;

    p.q_strideM = p.q_strideH * params.num_heads;
    p.k_strideM = p.k_strideH * params.num_heads;
    p.v_strideM = p.v_strideH * params.num_heads;
    p.mask_strideM = params.mask_strideM;

    p.q_strideB = p.q_strideM * params.query_seq_len;
    p.k_strideB = p.k_strideM * params.key_value_seq_len;
    p.v_strideB = p.v_strideM * params.key_value_seq_len;
    p.mask_strideB = params.mask_strideB;
  }

  constexpr auto kernel_fn = attention_kernel_batched<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }

  if (!Attention::check_supported(p)) {
    PADDLE_THROW(
        phi::errors::Unimplemented("The Params is not supported by cutlass "
                                   "fused multihead attention. "));
    return;
  }
  kernel_fn<<<p.getBlocksGrid(),
              p.getThreadsGrid(),
              smem_bytes,
              ctx.stream()>>>(p);
}

template <typename T,
          typename ArchTag,
          bool IsAligned,
          int QueriesPerBlock,
          int KeysPerBlock,
          bool SingleValueIteration,
          bool AddMask,
          bool MaskBroadcastRow>
void DispatchFMHADropout(LaunchParams params,
                                  const phi::GPUContext& ctx) {
  if (params.use_dropout) {
    LaunchMultiHeadAttentionKernel<T,
                                   ArchTag,
                                   IsAligned,
                                   QueriesPerBlock,
                                   KeysPerBlock,
                                   SingleValueIteration,
                                   AddMask,
                                   MaskBroadcastRow,
                                   true>(params, ctx);
  } else {
    LaunchMultiHeadAttentionKernel<T,
                                   ArchTag,
                                   IsAligned,
                                   QueriesPerBlock,
                                   KeysPerBlock,
                                   SingleValueIteration,
                                   AddMask,
                                   MaskBroadcastRow,
                                   false>(params, ctx);
  }
}

template <typename T,
          typename ArchTag,
          bool IsAligned,
          int QueriesPerBlock,
          int KeysPerBlock,
          bool SingleValueIteration,
          bool AddMask>
void DispatchFMHAMaskBroadcastRow(LaunchParams params,
                                  const phi::GPUContext& ctx) {
  if (params.mask_broadcast_row) {
    DispatchFMHADropout<T,
                        ArchTag,
                        IsAligned,
                        QueriesPerBlock,
                        KeysPerBlock,
                        SingleValueIteration,
                        AddMask,
                        true>(params, ctx);
  } else {
    DispatchFMHADropout<T,
                        ArchTag,
                        IsAligned,
                        QueriesPerBlock,
                        KeysPerBlock,
                        SingleValueIteration,
                        AddMask,
                        false>(params, ctx);
  }
}

template <typename T,
          typename ArchTag,
          bool IsAligned,
          int QueriesPerBlock,
          int KeysPerBlock,
          bool SingleValueIteration>
void DispatchFMHAAddMask(LaunchParams params, const phi::GPUContext& ctx) {
  if (params.mask_ptr != nullptr) {
    DispatchFMHAMaskBroadcastRow<T,
                                 ArchTag,
                                 IsAligned,
                                 QueriesPerBlock,
                                 KeysPerBlock,
                                 SingleValueIteration,
                                 true>(params, ctx);
  } else {
    DispatchFMHAMaskBroadcastRow<T,
                                 ArchTag,
                                 IsAligned,
                                 QueriesPerBlock,
                                 KeysPerBlock,
                                 SingleValueIteration,
                                 false>(params, ctx);
  }
}

template <typename T,
          typename ArchTag,
          bool IsAligned,
          int QueriesPerBlock,
          int KeysPerBlock>
void DispatchFMHASingleValueIteration(LaunchParams params,
                                      const phi::GPUContext& ctx) {
  if (params.value_head_size <= KeysPerBlock) {
    DispatchFMHAAddMask<T,
                        ArchTag,
                        IsAligned,
                        QueriesPerBlock,
                        KeysPerBlock,
                        true>(params, ctx);
  } else {
    // When we only register singlevalueIteration=True kernel in
    // QueriesPerBlock, KeysPerBlock == 64.
    if ((QueriesPerBlock != 64) && (KeysPerBlock != 64)) {
      DispatchFMHAAddMask<T,
                          ArchTag,
                          IsAligned,
                          QueriesPerBlock,
                          KeysPerBlock,
                          false>(params, ctx);
    }
  }
}

template <typename T, typename ArchTag, bool IsAligned>
void DispatchFMHABlockSize(LaunchParams params, const phi::GPUContext& ctx) {
  if (params.value_head_size > 64) {
    DispatchFMHASingleValueIteration<T, ArchTag, IsAligned, 32, 128>(params,
                                                                     ctx);
  } else {
    DispatchFMHASingleValueIteration<T, ArchTag, IsAligned, 64, 64>(params,
                                                                    ctx);
  }
}

template <typename T, typename ArchTag>
void DispatchFMHAIsAligned(LaunchParams params, const phi::GPUContext& ctx) {
  if (reinterpret_cast<uintptr_t>(params.query_ptr) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(params.key_ptr) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(params.value_ptr) % 16 == 0 &&
      params.query_strideH % (16 / sizeof(T)) == 0 &&
      params.query_strideH % (16 / sizeof(T)) == 0 &&
      params.value_strideH % (16 / sizeof(T)) == 0) {
    DispatchFMHABlockSize<T, ArchTag, true>(params, ctx);
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

template <typename T>
void DispatchFMHAArchTag(LaunchParams params, const phi::GPUContext& ctx) {
  const int compute_capability = ctx.GetComputeCapability();
  if (compute_capability == 80) {
    DispatchFMHAIsAligned<T, cutlass::arch::Sm80>(params, ctx);
  } else if (compute_capability == 75) {
    DispatchFMHAIsAligned<T, cutlass::arch::Sm75>(params, ctx);
  } else 
  if (compute_capability == 70) {
    DispatchFMHAIsAligned<T, cutlass::arch::Sm70>(params, ctx);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Currently cutlass fused multihead attention kernel "
        "only support arch: SM80, SM75, SM70"));
    return;
  }
}

void DispatchFusedMultiheadAttentionKernel(LaunchParams params,
                                           const phi::GPUContext& ctx) {
  if (params.datatype == DataType::FLOAT16 || 
      params.datatype == DataType::FLOAT32) {
    return DispatchFMHAArchTag<cutlass::half_t>(params, ctx);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Currently cutlass fused multihead attention kernel "
        "only support datatype float16. "));
  }
}

template <typename T, typename Context>
void MultiHeadAttentionForwardKernel(const Context& ctx,
                                     const DenseTensor& query,
                                     const DenseTensor& key,
                                     const DenseTensor& value,
                                     const paddle::optional<DenseTensor>& mask,
                                     const float scale,
                                     const bool causal,
                                     const float dropout_p,
                                     // TODO(zhangdanyang) : currently we don't use
                                     // const bool logsumexp_p,
                                     DenseTensor* output,
                                     DenseTensor* seed_and_offset) {
  ctx.template Alloc<T>(output);
  ctx.template Alloc<T>(seed_and_offset);
  LaunchParams params{};

  params.datatype = query.dtype();
  params.query_ptr = query.data();
  params.key_ptr = key.data();

  params.value_ptr = value.data();
  params.output_ptr = output->data();

  params.seed_and_offset = const_cast<uint64_t*>(reinterpret_cast<const uint64_t*>(seed_and_offset->data()));

  VLOG(3) << "output"<<output;
  VLOG(3) << "params.output_ptr "<<params.output_ptr ;
  VLOG(3) << "gen is generating";
  auto gen = ctx.GetGenerator();
  uint64_t inc = query.dims()[0] * query.dims()[2] * 32;
  auto seed_offset_pair = gen->IncrementOffset(inc);
  auto seed = (uint64_t)(seed_offset_pair.first);
  auto seed_offset = (uint64_t)(seed_offset_pair.second);
  VLOG(3) << "seed and offset have been generated";
  params.seed_and_offset = (uint64_t*) std::malloc (2*sizeof(uint64_t));
  params.seed_and_offset[0] = seed;
  params.seed_and_offset[1] = seed_offset;
  VLOG(3) << "seed and offset have been set";

  params.output_accum_ptr = nullptr;

  // TODO(zhengzekang): currently we only used in inference. Maybe add a bool
  // flag to save it ?
  params.logsumexp_ptr = nullptr;

  params.num_batches = query.dims()[0];
  params.query_seq_len = query.dims()[1];
  params.num_heads = query.dims()[2];
  params.key_value_seq_len = key.dims()[1];
  params.head_size = query.dims()[3];
  params.value_head_size = value.dims()[3];

  params.scale = scale;
  params.causal = causal;

  params.query_strideH = query.dims()[3];
  params.key_strideH = key.dims()[3];
  params.value_strideH = value.dims()[3];

  // dropout
  params.use_dropout = dropout_p==NULL? false : true;
  params.dropout_batch_head_rng_offset=NULL;
  params.dropout_prob = dropout_p;

  if (mask) {
    auto mask_tensor = mask.get();
    params.mask_ptr = mask_tensor.data();
    params.mask_strideM =
        mask_tensor.dims()[2] == 1 ? 0 : mask_tensor.dims()[3];
    params.mask_strideH = mask_tensor.dims()[1] == 1
                              ? 0
                              : mask_tensor.dims()[2] * mask_tensor.dims()[3];
    params.mask_strideB = mask_tensor.dims()[0] == 1
                              ? 0
                              : mask_tensor.dims()[1] * mask_tensor.dims()[2] *
                                    mask_tensor.dims()[3];

    params.mask_broadcast_row = false;
    if (params.mask_strideM == 0) {
      params.mask_broadcast_row = true;
    }

  }

  VLOG(3) << "fused multihead attention kernel is dispatching";
  DispatchFusedMultiheadAttentionKernel(params, ctx);
  VLOG(3) << "fused multihead attention kernel has been dispatched";
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

// PD_REGISTER_KERNEL(
//     fused_multihead_attention,
//     GPU,
//     ALL_LAYOUT,
//     phi::fusion::cutlass_internal::MultiHeadAttentionForwardKernel,
//     phi::dtype::float16) {}

// PD_REGISTER_KERNEL(
//     fused_multihead_attention,
//     GPU,
//     ALL_LAYOUT,
//     phi::fusion::cutlass_internal::MultiHeadAttentionForwardKernel,
//     phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(
    fused_multihead_attention,
    GPU,
    ALL_LAYOUT,
    phi::fusion::cutlass_internal::MultiHeadAttentionForwardKernel,
    float) {}
