/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

#if FLASH_ATTN_WITH_TORCH

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAGraphsUtils.cuh>

#endif

#include "random_utils.h"

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
  using index_t = uint32_t;
  // The QKV matrices.
  void *__restrict__ q_ptr;
  void *__restrict__ k_ptr;
  void *__restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;

  // The number of heads.
  int h, h_k;
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k
  // could be different from nheads (query).
  int h_h_k_ratio;  // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
  // The O matrix (output).
  void *__restrict__ o_ptr;

  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The pointer to the P matrix.
  void *__restrict__ p_ptr;

  // The pointer to the softmax sum.
  void *__restrict__ softmax_lse_ptr;
  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b,
  // nheads, seqlen_q].
  bool unpadded_lse;

  // The dimensions.
  int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded,
      total_q;

  // The scaling factors for the kernel.
  float scale_softmax;
  float scale_softmax_log2;
  float unscale_softmax;

  // array of length b+1 holding starting offset of each sequence.
  int *__restrict__ cu_seqlens_q;
  int *__restrict__ cu_seqlens_k;

  int *__restrict__ blockmask;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  // uint32_t p_dropout_in_uint;
  // uint16_t p_dropout_in_uint16_t;
  uint8_t p_dropout_in_uint8_t;

  // Scale factor of 1 / (1 - p_dropout).
  float rp_dropout;
  float scale_softmax_rp_dropout;

  // Random state.
  at::PhiloxCudaState philox_args;

  // Pointer to the RNG seed (idx 0) and offset (idx 1).
  uint64_t *rng_state;

  bool is_bf16;
  bool is_causal;

  // The attn mask matrix
  void *__restrict__ attn_mask_ptr;
  int mask_head_mod_size;
  int mask_seq_q_mod_size;

  bool varlen_padded_input = false;
  void *__restrict__ flashmask_downstart_ptr = nullptr;
  void *__restrict__ flashmask_upend_ptr = nullptr;
  void *__restrict__ flashmask_downend_ptr = nullptr;
  void *__restrict__ flashmask_upstart_ptr = nullptr;
  int *__restrict__ flashmask_maxmin_ptr = nullptr;
  int *__restrict__ flashmask_upend_nblockmax = nullptr;
  int *__restrict__ flashmask_upend_nblockmin = nullptr;
  int *__restrict__ flashmask_downstart_nblockmax = nullptr;
  int *__restrict__ flashmask_downstart_nblockmin = nullptr;
  int *__restrict__ flashmask_downend_nblockmax = nullptr;
  int *__restrict__ flashmask_downend_nblockmin = nullptr;
  int *__restrict__ flashmask_upstart_nblockmax = nullptr;
  int *__restrict__ flashmask_upstart_nblockmin = nullptr;
  int h_sparsemask;
  int h_h_sparsemask_ratio;
  bool enable_mask_bypass;

  // AP variadic epilogue: POD copy of the generated functor Arguments. Kept
  // inside Flash_fwd_params so the FA kernel signature stays upstream-shaped.
  alignas(8) unsigned char variadic_args[64] = {};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {
  // The dO and dQKV matrices.
  void *__restrict__ do_ptr;
  void *__restrict__ dq_ptr;
  void *__restrict__ dk_ptr;
  void *__restrict__ dv_ptr;

  // To accumulate dQ
  void *__restrict__ dq_accum_ptr;
  void *__restrict__ dk_accum_ptr;
  void *__restrict__ dv_accum_ptr;

  // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
  // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
  // dv_accum_ptr;

  // The stride between rows of the dO, dQ, dK and dV matrices.
  // TD [2022-04-16]: We're using 32-bit indexing to save registers.
  // The code probably won't work for arrays larger than 2GB.
  index_t do_batch_stride;
  index_t do_row_stride;
  index_t do_head_stride;
  index_t dq_batch_stride;
  index_t dk_batch_stride;
  index_t dv_batch_stride;
  index_t dq_row_stride;
  index_t dk_row_stride;
  index_t dv_row_stride;
  index_t dq_head_stride;
  index_t dk_head_stride;
  index_t dv_head_stride;

  // The pointer to the softmax d sum.
  void *__restrict__ dsoftmax_sum;
  int num_splits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T,
          int Headdim,
          bool Is_causal,
          bool Is_densemask,
          bool Is_flashmask>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

template <typename T,
          int Headdim,
          bool Is_causal,
          bool Is_densemask,
          bool Is_flashmask>
void run_mha_bwd_(Flash_bwd_params &params,
                  cudaStream_t stream,
                  const bool configure);
