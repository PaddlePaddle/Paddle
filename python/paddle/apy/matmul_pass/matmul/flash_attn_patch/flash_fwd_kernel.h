/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "philox.cuh"
#include "softmax.h"
#include "utils.h"

// AP variadic epilogue: the functor type and its Arguments are supplied by the
// AP generated kernel via -DAP_ATTENTION_VARIADIC_FUNCTOR / -D..._ARGS, so no
// template parameter has to be threaded through the FA kernel.
#ifdef AP_ATTENTION_VARIADIC_FUNCTOR
#include "cutlass_patch/batched_matrix_coord.h"
#endif

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_M, class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto make_tiled_copy_A_warpcontiguousM(
    Copy_Atom<Args...> const &copy_atom, TiledMMA const &tiled_mma) {
  using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
  using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
  constexpr int AtomShape_M = decltype(size<0>(AtomShape_MNK{}))::value;
  constexpr int kNWarps =
      decltype(size<0>(TileShape_MNK{}))::value / AtomShape_M;
  constexpr int MMAStride_M = MMA_M * AtomShape_M;
  auto t = make_tile(Layout<Shape<Int<AtomShape_M>, Int<kNWarps>>,
                            Stride<_1, Int<MMAStride_M>>>{},
                     make_layout(size<2>(TileShape_MNK{})));
  // if (cute::thread0()) {printf("make_tiled_copy_A_warpcontiguousM ");
  // print(t); printf("\n");  }
  return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutA_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_M, class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto make_tiled_copy_C_warpcontiguousM(
    Copy_Atom<Args...> const &copy_atom, TiledMMA const &tiled_mma) {
  using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
  using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
  constexpr int AtomShape_M = decltype(size<0>(AtomShape_MNK{}))::value;
  constexpr int kNWarps =
      decltype(size<0>(TileShape_MNK{}))::value / AtomShape_M;
  constexpr int MMAStride_M = MMA_M * AtomShape_M;
  auto t = make_tile(Layout<Shape<Int<AtomShape_M>, Int<kNWarps>>,
                            Stride<_1, Int<MMAStride_M>>>{},
                     // TODO: Shouldn't this be size<1>?
                     make_layout(size<2>(TileShape_MNK{})));
  // if (cute::thread0()) {printf("make_tiled_copy_C_warpcontiguousM ");
  // print(t); printf("\n");  }
  return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_first,
          bool Check_inf = false,
          typename Tensor0,
          typename Tensor1,
          typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores,
                                         Tensor1 &scores_max,
                                         Tensor1 &scores_sum,
                                         Tensor2 &acc_o,
                                         float softmax_scale_log2) {
  if (Is_first) {
    flash::template reduce_max</*zero_init=*/true>(scores, scores_max);
    flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
    flash::reduce_sum(scores, scores_sum);
  } else {
    Tensor scores_max_prev = make_fragment_like(scores_max);
    cute::copy(scores_max, scores_max_prev);
    flash::template reduce_max</*zero_init=*/false>(scores, scores_max);
    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2,
    // MMA_K))
    Tensor acc_o_rowcol = make_tensor(
        acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
    for (int mi = 0; mi < size(scores_max); ++mi) {
      float scores_max_cur =
          !Check_inf ? scores_max(mi)
                     : (scores_max(mi) == -INFINITY ? 0.0f : scores_max(mi));
      float scores_scale =
          exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
      scores_sum(mi) *= scores_scale;
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale;
      }
    }
    flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
    Tensor scores_sum_cur = make_fragment_like(scores_sum);
    flash::reduce_sum(scores, scores_sum_cur);
#pragma unroll
    for (int mi = 0; mi < size(scores_sum); ++mi) {
      scores_sum(mi) += scores_sum_cur(mi);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1,
          typename TiledCopy>
inline __device__ void write_softmax_to_gmem(
    Tensor<Engine0, Layout0> const &tOrP,
    Tensor<Engine1, Layout1> &tPgP,
    TiledCopy gmem_tiled_copy_P) {
  // Reshape tOrP from (8, MMA_M, MMA_N) to (8, MMA_M * MMA_N)
  Layout l = tOrP.layout();
  Tensor tPrP = make_tensor(
      tOrP.data(), make_layout(get<0>(l), make_layout(get<1>(l), get<2>(l))));
  CUTE_STATIC_ASSERT_V(size<2>(tPgP) == _1{});
  CUTE_STATIC_ASSERT_V(size<1>(tPrP) == size<1>(tPgP));
#pragma unroll
  for (int mi = 0; mi < size<1>(tPrP); ++mi) {
    cute::copy(gmem_tiled_copy_P, tPrP(_, mi), tPgP(_, mi, 0));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits,
          bool Is_dropout,
          bool Is_causal,
          bool Is_even_N,
          bool Is_even_K,
          bool Return_softmax,
          bool Is_equal_seq_qk,
          typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params,
                                              const int bidb,
                                              const int bidh,
                                              const int m_block) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;
  // The global block index.
  const int block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int MMA_M =
      kBlockM / decltype(size<0>(
                    typename Kernel_traits::TiledMma::TiledShape_MNK{}))::value;

  const BlockInfo</*Varlen=*/!Is_even_N> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0)
    return;

  const int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  if (Is_causal) {
    n_block_max =
        std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM, kBlockN));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
    // }
  }

  // We iterate over the blocks in reverse order. This is because the last block
  // is the only one that needs masking when we read K and V from global memory.
  // Moreover, iterating in reverse might save us 1 register (we just need
  // n_block instead of both n_block and n_block_max).

  const index_t row_offset_q =
      binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
      m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
  // We move K and V to the last block.
  const index_t row_offset_k =
      binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
      (n_block_max - 1) * kBlockN * params.k_row_stride +
      (bidh / params.h_h_k_ratio) * params.k_head_stride;
  const index_t row_offset_v =
      binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
      (n_block_max - 1) * kBlockN * params.v_row_stride +
      (bidh / params.h_h_k_ratio) * params.v_head_stride;
  const index_t row_offset_p =
      ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) *
          params.seqlen_k_rounded +
      (n_block_max - 1) * kBlockN;

  Tensor gQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_stride(params.q_row_stride, _1{}));
  Tensor gK = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_stride(params.k_row_stride, _1{}));
  Tensor gV = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_stride(params.v_row_stride, _1{}));
  Tensor gP = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
      Shape<Int<kBlockM>, Int<kBlockN>>{},
      make_stride(params.seqlen_k_rounded, _1{}));

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
  Tensor sK =
      make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                  typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(
      sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  typename Kernel_traits::GmemTiledCopyP gmem_tiled_copy_P;
  auto gmem_thr_copy_P = gmem_tiled_copy_P.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
  Tensor tPgP = gmem_thr_copy_P.partition_D(gP);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)
  Tensor tOrVt =
      thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

  Tensor acc_o = partition_fragment_C(
      tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  //
  // Copy Atom retiling
  //

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  // auto smem_thr_copy_Q = make_tiled_copy_A_warpcontiguousM<MMA_M>(typename
  // Kernel_traits::SmemCopyAtom{}, tiled_mma).get_thread_slice(tidx); if
  // (cute::thread0()) {smem_thr_copy_Q.print_all();}
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
  // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  // TODO: this might need to change if we change the mma instruction in SM70
  Tensor scores_max =
      make_tensor<ElementAccum>(Shape<Int<2 * size<1>(acc_o)>>{});
  Tensor scores_sum = make_fragment_like(scores_max);

  //
  // PREDICATES
  //

  // // Allocate predicate tensors for m and n
  // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)),
  // Stride<_1,_0>{}); Tensor tKVpKV =
  // make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)),
  // Stride<_1,_0>{});

  // Construct identity layout for sQ and sK
  Tensor cQ = make_identity_tensor(
      make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cKV = make_identity_tensor(
      make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
  // Tensor tScQ = thr_mma.partition_A(cQ);                           //
  // (MMA,MMA_M,MMA_K) if (cute::thread0()) {
  //     print(tScQ.layout()); printf("\n");
  //     for (int i = 0; i < size(tScQ); ++i) {
  //         printf("%d ", get<0>(tScQ(i)));
  //     }
  //     printf("\n");
  //     for (int i = 0; i < size(tScQ); ++i) {
  //         printf("%d ", get<1>(tScQ(i)));
  //     }
  //     printf("\n");
  // }

  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(
      cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(
      cKV);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

  // Allocate predicate tensors for k
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

  // Set predicates for k bounds
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
#pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) {
      tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
    }
  }

  // Prologue

  Tensor tQrQ = make_fragment_like(tQgQ);
  // We don't need to clear the sQ smem tiles since we'll only write out the
  // valid outputs
  flash::copy</*Is_even_MN=*/false, Is_even_K>(
      gmem_tiled_copy_QKV,
      tQgQ,
      tQsQ,
      tQcQ,
      tQpQ,
      binfo.actual_seqlen_q - m_block * kBlockM);
  if (Kernel_traits::Is_Q_in_regs) {
    cute::cp_async_fence();
  }

  // // Copy rmem to smem
  // // copy(tQrQ, tQsQ);
  // flash::cp_async_wait<0>();
  // __syncthreads();
  // // if (cute::thread(1, 0)) { print(tQsQ); }
  // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element
  // *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
  // // if (cute::thread0()) { print(sQNoSwizzle); }

  if (Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<0>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    __syncthreads();
  }

  int n_block = n_block_max - 1;
  // We don't need to clear the sK smem tiles since we'll mask out the scores
  // anyway.
  flash::copy<Is_even_N, Is_even_K>(gmem_tiled_copy_QKV,
                                    tKgK,
                                    tKsK,
                                    tKVcKV,
                                    tKVpKV,
                                    binfo.actual_seqlen_k - n_block * kBlockN);
  cute::cp_async_fence();
  // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
  // __syncthreads();

  if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<1>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
  }

  auto seeds = at::cuda::philox::unpack(params.philox_args);
  unsigned long long seed = std::get<0>(seeds);
  unsigned long long offset =
      std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + tidx % 32;

  // Save seed and offset for backward.
  if (block_id == 0 && tidx == 0) {
    params.rng_state[0] = seed;
    params.rng_state[1] = std::get<1>(seeds);
  }

  clear(acc_o);

  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not
  // multiple of kBlockN. We also need masking on S if it's causal, for the last
  // ceil_div(kBlockM, kBlockN) blocks. We will have at least 1 "masking"
  // iteration.
  constexpr int n_masking_steps =
      Is_causal ? cute::ceil_div(kBlockM, kBlockN) : 1;
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps;
       ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    // Advance gV
    if (masking_step > 0) {
      tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
    } else {
      // Clear the smem tiles to account for predicated off loads
      flash::copy<Is_even_N, Is_even_K, /*Clear_OOB_MN=*/true>(
          gmem_tiled_copy_QKV,
          tVgV,
          tVsV,
          tKVcKV,
          tKVpKV,
          binfo.actual_seqlen_k - n_block * kBlockN);
    }
    cute::cp_async_fence();

    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(acc_s,
                                                           tSrQ,
                                                           tSrK,
                                                           tSsQ,
                                                           tSsK,
                                                           tiled_mma,
                                                           smem_tiled_copy_Q,
                                                           smem_tiled_copy_K,
                                                           smem_thr_copy_Q,
                                                           smem_thr_copy_K);
    // if (cute::thread0()) { print(acc_s); }

    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2,
    // MMA_N))
    Tensor scores = make_tensor(
        acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

    // if (cute::thread0()) { print(scores); }
    // We don't put the masking before the matmul S = Q K^T because we don't
    // clear sK for rows outside actual_seqlen_k. So those rows could have Inf /
    // NaN, and the matmul can produce Inf / NaN.
    if (!Is_causal) {
      if (!Is_even_N) {
        flash::apply_mask(scores, binfo.actual_seqlen_k - n_block * kBlockN);
      }
    } else {
      // Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>,
      // Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n) Tensor taccScS =
      // thr_mma.partition_C(caccS);                           //
      // (MMA,MMA_M,MMA_N) static_assert(decltype(size<0>(taccScS))::value ==
      // 4);
      // // Convert to ((2, 2), MMA_M, MMA_N) then take only the row indices.
      // Tensor idx_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _),
      // _, 0); Tensor idx_rowcol = make_tensor(taccScS.data(),
      // flash::convert_layout_acc_rowcol(taccScS.layout()));
      // flash::apply_mask_causal_w_idx(scores, idx_rowcol, n_block * kBlockN,
      // binfo.actual_seqlen_k,
      //                               m_block * kBlockM);
      // Idk why it's get<1> and not get<0> of the stride.
      // if (cute::thread0()) { print(idx_row.layout());
      // print(stride<1>(idx_row)); printf("stride = %d \n",
      // get<1>(stride<1>(idx_row))); } I can't get the stride from idx_row

      flash::apply_mask_causal(
          scores,
          n_block * kBlockN,
          binfo.actual_seqlen_k,
          // m_block * kBlockM + get<0>(idx_row(0)),
          m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
          kNWarps * 16);
      // m_block * kBlockM + (tidx / 32) * 16, kNWarps * 16);
      // m_block * kBlockM + (tidx / 32) * (kBlockM / kNWarps), 16);
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > 0) {
      // Advance gK
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // TODO: when we have key_padding_mask we'll need to Check_inf
    masking_step == 0
        ? softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal>(
              scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2)
        : softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(
              scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

    // Convert scores from fp32 to fp16/bf16
    Tensor rP = flash::convert_type<Element>(scores);
    // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M,
    // MMA_N / 2) if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using
    // m16n8k8.
    Tensor tOrP =
        make_tensor(rP.data(),
                    flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(
                        rP.layout()));
    uint32_t block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
    uint32_t block_col_idx = n_block * (kBlockN / 32);
    if (Return_softmax) {
      Tensor tOrP_copy = make_fragment_like(tOrP);
      cute::copy(tOrP, tOrP_copy);
      flash::apply_dropout</*encode_dropout_in_sign_bit=*/true>(
          tOrP_copy,
          params.p_dropout_in_uint8_t,
          seed,
          offset,
          block_row_idx,
          block_col_idx,
          kNWarps);
      flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_tiled_copy_P);
      tPgP.data() = tPgP.data() + (-kBlockN);
    }
    if (Is_dropout) {
      flash::apply_dropout(tOrP,
                           params.p_dropout_in_uint8_t,
                           seed,
                           offset,
                           block_row_idx,
                           block_col_idx,
                           kNWarps);
    }
    // if (cute::thread0()) { print(tOrP); }

    flash::gemm_A_in_regs(acc_o,
                          tOrP,
                          tOrVt,
                          tOsVt,
                          tiled_mma,
                          smem_tiled_copy_V,
                          smem_thr_copy_V);
    // if (cute::thread0()) { print(scores); }

    // This check is at the end of the loop since we always have at least 1
    // iteration
    if (n_masking_steps > 1 && n_block <= 0) {
      --n_block;
      break;
    }
  }

  // These are the iterations where we don't need masking on S
  for (; n_block >= 0; --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();
    // Advance gV
    tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
    flash::copy</*Is_even_MN=*/true, Is_even_K>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
    cute::cp_async_fence();

    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(acc_s,
                                                           tSrQ,
                                                           tSrK,
                                                           tSsQ,
                                                           tSsK,
                                                           tiled_mma,
                                                           smem_tiled_copy_Q,
                                                           smem_tiled_copy_K,
                                                           smem_thr_copy_Q,
                                                           smem_thr_copy_K);

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > 0) {
      // Advance gK
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2,
    // MMA_N))
    Tensor scores = make_tensor(
        acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

    if (Is_equal_seq_qk) {
      softmax_rescale_o</*Is_first=*/false>(
          scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);
    } else {
      softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(
          scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);
    }

    Tensor rP = flash::convert_type<Element>(scores);
    // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M,
    // MMA_N / 2) if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using
    // m16n8k8.
    Tensor tOrP =
        make_tensor(rP.data(),
                    flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(
                        rP.layout()));
    uint32_t block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
    uint32_t block_col_idx = n_block * (kBlockN / 32);
    if (Return_softmax) {
      Tensor tOrP_copy = make_fragment_like(tOrP);
      cute::copy(tOrP, tOrP_copy);
      flash::apply_dropout</*encode_dropout_in_sign_bit=*/true>(
          tOrP_copy,
          params.p_dropout_in_uint8_t,
          seed,
          offset,
          block_row_idx,
          block_col_idx,
          kNWarps);
      flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_tiled_copy_P);
      tPgP.data() = tPgP.data() + (-kBlockN);
    }
    if (Is_dropout) {
      flash::apply_dropout(tOrP,
                           params.p_dropout_in_uint8_t,
                           seed,
                           offset,
                           block_row_idx,
                           block_col_idx,
                           kNWarps);
    }

    flash::gemm_A_in_regs(acc_o,
                          tOrP,
                          tOrVt,
                          tOsVt,
                          tiled_mma,
                          smem_tiled_copy_V,
                          smem_thr_copy_V);
  }

  // Epilogue

  // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2,
  // MMA_K))
  Tensor acc_o_rowcol = make_tensor(
      acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
  Tensor lse = make_fragment_like(scores_sum);
#pragma unroll
  for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    float sum = scores_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    lse(mi) = (sum == 0.f || sum != sum)
                  ? INFINITY
                  : scores_max(mi) * params.scale_softmax + __logf(sum);
    float scale = !Is_dropout ? inv_sum : inv_sum * params.rp_dropout;
#pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
      acc_o_rowcol(mi, ni) *= scale;
    }
  }

  // if (cute::thread0()) { print(acc_o_rowcol); }

  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = flash::convert_type<Element>(acc_o);
  Tensor sO = make_tensor(
      sQ.data(), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  // auto smem_thr_copy_O = make_tiled_copy_C_warpcontiguousM<MMA_M>(typename
  // Kernel_traits::SmemCopyAtomO{}, tiled_mma).get_thread_slice(tidx);
  Tensor taccOrO =
      smem_thr_copy_O.retile_S(rO);  // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO =
      smem_thr_copy_O.partition_D(sO);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // sO has the same size as sQ, so we don't need to sync here.
  if (Kernel_traits::Share_Q_K_smem) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  const index_t row_offset_o =
      binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
      m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
  const index_t row_offset_lse =
      params.unpadded_lse
          ? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb) +
                m_block * kBlockM
          : (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
  Tensor gO = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_stride(params.o_row_stride, _1{}));
  Tensor gLSE = make_tensor(
      make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) +
                    row_offset_lse),
      Shape<Int<kBlockM>>{},
      Stride<_1>{});

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO =
      gmem_thr_copy_O.partition_S(sO);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  Tensor caccO = make_identity_tensor(
      Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO);  // (MMA,MMA_M,MMA_K)
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  Tensor taccOcO_row =
      logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M
  if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSE(row) = lse(mi);
      }
    }
  }

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(
      make_shape(size<0>(sO), size<1>(sO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  Tensor tOcO =
      gmem_thr_copy_O.partition_D(cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
  }
#ifdef AP_ATTENTION_VARIADIC_FUNCTOR
  // AP variadic epilogue (true fusion): apply the generated functor to the
  // normalized output tile in registers before it is written to gmem.
  {
    using ApVariadicFunctor = AP_ATTENTION_VARIADIC_FUNCTOR<float>;
    using ApVariadicArgs = AP_ATTENTION_VARIADIC_ARGS;
    const auto &variadic_args =
        *reinterpret_cast<const ApVariadicArgs *>(params.variadic_args);
    ApVariadicArgs local_args = variadic_args;
    local_args.b = bidb;
    local_args.h = bidh;
    ApVariadicFunctor variadic_functor;
    const int row_limit = binfo.actual_seqlen_q - m_block * kBlockM;
#pragma unroll
    for (int mi = 0; mi < size<1>(tOrO); ++mi) {
#pragma unroll
      for (int ni = 0; ni < size<2>(tOrO); ++ni) {
#pragma unroll
        for (int ai = 0; ai < size<0>(tOrO); ++ai) {
          const auto coord = tOcO(ai, mi, ni);  // (blk_m, blk_k)
          const int row = int(get<0>(coord));
          const int col = int(get<1>(coord));
          if (row < row_limit && col < params.d) {
            tOrO(ai, mi, ni) = Element(
                variadic_functor(static_cast<float>(tOrO(ai, mi, ni)),
                                 local_args,
                                 cutlass_patch::BatchedMatrixCoord(
                                     bidb, m_block * kBlockM + row, col)));
          }
        }
      }
    }
  }
#endif
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy</*Is_even_MN=*/false,
              Is_even_K,
              /*Clear_OOB_MN=*/false,
              /*Clear_OOB_K=*/false>(gmem_tiled_copy_O,
                                     tOrO,
                                     tOgO,
                                     tOcO,
                                     tOpO,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits,
          bool Is_dropout,
          bool Is_causal,
          bool Is_even_N,
          bool Is_even_K,
          bool Return_softmax,
          bool Is_equal_seq_qk,
          typename Params>
inline __device__ void compute_attn_1rowblock_densemask(const Params &params,
                                                        const int bidb,
                                                        const int bidh,
                                                        const int m_block) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;
  // The global block index.
  const int block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int MMA_M =
      kBlockM / decltype(size<0>(
                    typename Kernel_traits::TiledMma::TiledShape_MNK{}))::value;

  const BlockInfo</*Varlen=*/!Is_even_N> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0)
    return;

  // umiswing: residue is for predication of additional mask gmem access.
  // Additional mask for varlen qkv is supported, but a varlen mask is not
  // supported.
  const int m_residue =
      params.seqlen_q % kBlockM ? params.seqlen_q % kBlockM : kBlockM;
  const int n_residue =
      params.seqlen_k % kBlockN ? params.seqlen_k % kBlockN : kBlockN;

  const int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  const int oob_m_block_max = m_block_max;
  const int oob_n_block_max = n_block_max;
  if (Is_causal) {
    n_block_max =
        std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM, kBlockN));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
    // }
  }

  // We iterate over the blocks in reverse order. This is because the last block
  // is the only one that needs masking when we read K and V from global memory.
  // Moreover, iterating in reverse might save us 1 register (we just need
  // n_block instead of both n_block and n_block_max).

  const index_t row_offset_q =
      binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
      m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
  // We move K and V to the last block.
  const index_t row_offset_k =
      binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
      (n_block_max - 1) * kBlockN * params.k_row_stride +
      (bidh / params.h_h_k_ratio) * params.k_head_stride;
  const index_t row_offset_v =
      binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
      (n_block_max - 1) * kBlockN * params.v_row_stride +
      (bidh / params.h_h_k_ratio) * params.v_head_stride;
  const index_t row_offset_p =
      ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) *
          params.seqlen_k_rounded +
      (n_block_max - 1) * kBlockN;

  const uint64_t row_offset_mask =
      (uint64_t)((bidb * params.mask_head_mod_size +
                  (bidh % params.mask_head_mod_size)) *
                     params.mask_seq_q_mod_size +
                 (m_block * kBlockM % params.mask_seq_q_mod_size)) *
          params.seqlen_k +
      (n_block_max - 1) * kBlockN;

  Tensor gQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_stride(params.q_row_stride, _1{}));
  Tensor gK = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_stride(params.k_row_stride, _1{}));
  Tensor gV = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_stride(params.v_row_stride, _1{}));
  Tensor gP = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
      Shape<Int<kBlockM>, Int<kBlockN>>{},
      make_stride(params.seqlen_k_rounded, _1{}));

  Tensor gMask = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.attn_mask_ptr) +
                    row_offset_mask),
      Shape<Int<kBlockM>, Int<kBlockN>>{},
      make_stride(params.seqlen_k, _1{}));

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
  Tensor sK =
      make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                  typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(
      sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  typename Kernel_traits::GmemTiledCopyP gmem_tiled_copy_P;
  auto gmem_thr_copy_P = gmem_tiled_copy_P.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
  Tensor tPgP = gmem_thr_copy_P.partition_D(gP);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)
  Tensor tOrVt =
      thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

  Tensor acc_o = partition_fragment_C(
      tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  auto gmem_thr_copy_Mask =
      make_tiled_copy_C(typename Kernel_traits::GmemCopyAtomMask{}, tiled_mma)
          .get_thread_slice(tidx);

  Tensor tPgMask = gmem_thr_copy_Mask.partition_D(gMask);
  Tensor cMask = make_identity_tensor(shape(gMask));
  Tensor tPcMask = gmem_thr_copy_Mask.partition_D(cMask);

  //
  // Copy Atom retiling
  //

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  // auto smem_thr_copy_Q = make_tiled_copy_A_warpcontiguousM<MMA_M>(typename
  // Kernel_traits::SmemCopyAtom{}, tiled_mma).get_thread_slice(tidx); if
  // (cute::thread0()) {smem_thr_copy_Q.print_all();}
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
  // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  // TODO: this might need to change if we change the mma instruction in SM70
  Tensor scores_max =
      make_tensor<ElementAccum>(Shape<Int<2 * size<1>(acc_o)>>{});
  Tensor scores_sum = make_fragment_like(scores_max);

  //
  // PREDICATES
  //

  // // Allocate predicate tensors for m and n
  // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)),
  // Stride<_1,_0>{}); Tensor tKVpKV =
  // make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)),
  // Stride<_1,_0>{});

  // Construct identity layout for sQ and sK
  Tensor cQ = make_identity_tensor(
      make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cKV = make_identity_tensor(
      make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
  // Tensor tScQ = thr_mma.partition_A(cQ);                           //
  // (MMA,MMA_M,MMA_K) if (cute::thread0()) {
  //     print(tScQ.layout()); printf("\n");
  //     for (int i = 0; i < size(tScQ); ++i) {
  //         printf("%d ", get<0>(tScQ(i)));
  //     }
  //     printf("\n");
  //     for (int i = 0; i < size(tScQ); ++i) {
  //         printf("%d ", get<1>(tScQ(i)));
  //     }
  //     printf("\n");
  // }

  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(
      cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(
      cKV);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

  // Allocate predicate tensors for k
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

  // Set predicates for k bounds
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
#pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) {
      tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
    }
  }

  // Prologue

  Tensor tQrQ = make_fragment_like(tQgQ);
  // We don't need to clear the sQ smem tiles since we'll only write out the
  // valid outputs
  flash::copy</*Is_even_MN=*/false, Is_even_K>(
      gmem_tiled_copy_QKV,
      tQgQ,
      tQsQ,
      tQcQ,
      tQpQ,
      binfo.actual_seqlen_q - m_block * kBlockM);
  if (Kernel_traits::Is_Q_in_regs) {
    cute::cp_async_fence();
  }

  // // Copy rmem to smem
  // // copy(tQrQ, tQsQ);
  // flash::cp_async_wait<0>();
  // __syncthreads();
  // // if (cute::thread(1, 0)) { print(tQsQ); }
  // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element
  // *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
  // // if (cute::thread0()) { print(sQNoSwizzle); }

  if (Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<0>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    __syncthreads();
  }

  int n_block = n_block_max - 1;
  // We don't need to clear the sK smem tiles since we'll mask out the scores
  // anyway.
  flash::copy<Is_even_N, Is_even_K>(gmem_tiled_copy_QKV,
                                    tKgK,
                                    tKsK,
                                    tKVcKV,
                                    tKVpKV,
                                    binfo.actual_seqlen_k - n_block * kBlockN);
  cute::cp_async_fence();
  // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
  // __syncthreads();

  if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<1>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
  }

  auto seeds = at::cuda::philox::unpack(params.philox_args);
  unsigned long long seed = std::get<0>(seeds);
  unsigned long long offset =
      std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + tidx % 32;

  // Save seed and offset for backward.
  if (block_id == 0 && tidx == 0) {
    params.rng_state[0] = seed;
    params.rng_state[1] = std::get<1>(seeds);
  }

  clear(acc_o);

  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not
  // multiple of kBlockN. We also need masking on S if it's causal, for the last
  // ceil_div(kBlockM, kBlockN) blocks. We will have at least 1 "masking"
  // iteration.
  constexpr int n_masking_steps =
      Is_causal ? cute::ceil_div(kBlockM, kBlockN) : 1;
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps;
       ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    // Advance gV
    if (masking_step > 0) {
      tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
    } else {
      // Clear the smem tiles to account for predicated off loads
      flash::copy<Is_even_N, Is_even_K, /*Clear_OOB_MN=*/true>(
          gmem_tiled_copy_QKV,
          tVgV,
          tVsV,
          tKVcKV,
          tKVpKV,
          binfo.actual_seqlen_k - n_block * kBlockN);
    }
    cute::cp_async_fence();

    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(acc_s,
                                                           tSrQ,
                                                           tSrK,
                                                           tSsQ,
                                                           tSsK,
                                                           tiled_mma,
                                                           smem_tiled_copy_Q,
                                                           smem_tiled_copy_K,
                                                           smem_thr_copy_Q,
                                                           smem_thr_copy_K);
    // if (cute::thread0()) { print(acc_s); }

    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2,
    // MMA_N))
    Tensor scores = make_tensor(
        acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

    if (true /*Is_densemask*/) {
      flash::apply_attn_mask<Kernel_traits::TiledMma>(
          scores,
          tPgMask,
          tPcMask,
          m_block == oob_m_block_max - 1 ? m_residue : params.seqlen_q,
          n_block == oob_n_block_max - 1 ? n_residue : params.seqlen_k,
          params.unscale_softmax);
      tPgMask.data() = tPgMask.data() + (-kBlockN);
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > 0) {
      // Advance gK
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // TODO: when we have key_padding_mask we'll need to Check_inf
    {
      masking_step == 0
          ? softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/true>(
                scores,
                scores_max,
                scores_sum,
                acc_o,
                params.scale_softmax_log2)
          : softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
                scores,
                scores_max,
                scores_sum,
                acc_o,
                params.scale_softmax_log2);
    }
    // Convert scores from fp32 to fp16/bf16
    Tensor rP = flash::convert_type<Element>(scores);
    // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M,
    // MMA_N / 2) if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using
    // m16n8k8.
    Tensor tOrP =
        make_tensor(rP.data(),
                    flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(
                        rP.layout()));
    uint32_t block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
    uint32_t block_col_idx = n_block * (kBlockN / 32);
    if (Return_softmax) {
      Tensor tOrP_copy = make_fragment_like(tOrP);
      cute::copy(tOrP, tOrP_copy);
      flash::apply_dropout</*encode_dropout_in_sign_bit=*/true>(
          tOrP_copy,
          params.p_dropout_in_uint8_t,
          seed,
          offset,
          block_row_idx,
          block_col_idx,
          kNWarps);
      flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_tiled_copy_P);
      tPgP.data() = tPgP.data() + (-kBlockN);
    }
    if (Is_dropout) {
      flash::apply_dropout(tOrP,
                           params.p_dropout_in_uint8_t,
                           seed,
                           offset,
                           block_row_idx,
                           block_col_idx,
                           kNWarps);
    }
    // if (cute::thread0()) { print(tOrP); }

    flash::gemm_A_in_regs(acc_o,
                          tOrP,
                          tOrVt,
                          tOsVt,
                          tiled_mma,
                          smem_tiled_copy_V,
                          smem_thr_copy_V);
    // if (cute::thread0()) { print(scores); }

    // This check is at the end of the loop since we always have at least 1
    // iteration
    if (n_masking_steps > 1 && n_block <= 0) {
      --n_block;
      break;
    }
  }

  // These are the iterations where we don't need masking on S
  for (; n_block >= 0; --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();
    // Advance gV
    tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
    flash::copy</*Is_even_MN=*/true, Is_even_K>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
    cute::cp_async_fence();

    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(acc_s,
                                                           tSrQ,
                                                           tSrK,
                                                           tSsQ,
                                                           tSsK,
                                                           tiled_mma,
                                                           smem_tiled_copy_Q,
                                                           smem_tiled_copy_K,
                                                           smem_thr_copy_Q,
                                                           smem_thr_copy_K);

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > 0) {
      // Advance gK
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2,
    // MMA_N))
    Tensor scores = make_tensor(
        acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

    if (true /*Is_densemask*/) {
      flash::apply_attn_mask<Kernel_traits::TiledMma>(
          scores,
          tPgMask,
          tPcMask,
          m_block == oob_m_block_max - 1 ? m_residue : params.seqlen_q,
          n_block == oob_n_block_max - 1 ? n_residue : params.seqlen_k,
          params.unscale_softmax);
      tPgMask.data() = tPgMask.data() + (-kBlockN);
    }

    if (Is_equal_seq_qk) {
      softmax_rescale_o</*Is_first=*/false>(
          scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);
    } else {
      softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
          scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);
    }

    Tensor rP = flash::convert_type<Element>(scores);
    // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M,
    // MMA_N / 2) if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using
    // m16n8k8.
    Tensor tOrP =
        make_tensor(rP.data(),
                    flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(
                        rP.layout()));
    uint32_t block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
    uint32_t block_col_idx = n_block * (kBlockN / 32);
    if (Return_softmax) {
      Tensor tOrP_copy = make_fragment_like(tOrP);
      cute::copy(tOrP, tOrP_copy);
      flash::apply_dropout</*encode_dropout_in_sign_bit=*/true>(
          tOrP_copy,
          params.p_dropout_in_uint8_t,
          seed,
          offset,
          block_row_idx,
          block_col_idx,
          kNWarps);
      flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_tiled_copy_P);
      tPgP.data() = tPgP.data() + (-kBlockN);
    }
    if (Is_dropout) {
      flash::apply_dropout(tOrP,
                           params.p_dropout_in_uint8_t,
                           seed,
                           offset,
                           block_row_idx,
                           block_col_idx,
                           kNWarps);
    }

    flash::gemm_A_in_regs(acc_o,
                          tOrP,
                          tOrVt,
                          tOsVt,
                          tiled_mma,
                          smem_tiled_copy_V,
                          smem_thr_copy_V);
  }

  // Epilogue

  // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2,
  // MMA_K))
  Tensor acc_o_rowcol = make_tensor(
      acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
  Tensor lse = make_fragment_like(scores_sum);
#pragma unroll
  for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    float sum = scores_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    lse(mi) = (sum == 0.f || sum != sum)
                  ? INFINITY
                  : scores_max(mi) * params.scale_softmax + __logf(sum);
    float scale = !Is_dropout ? inv_sum : inv_sum * params.rp_dropout;
#pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
      acc_o_rowcol(mi, ni) *= scale;
    }
  }

  // if (cute::thread0()) { print(acc_o_rowcol); }

  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = flash::convert_type<Element>(acc_o);
  Tensor sO = make_tensor(
      sQ.data(), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  // auto smem_thr_copy_O = make_tiled_copy_C_warpcontiguousM<MMA_M>(typename
  // Kernel_traits::SmemCopyAtomO{}, tiled_mma).get_thread_slice(tidx);
  Tensor taccOrO =
      smem_thr_copy_O.retile_S(rO);  // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO =
      smem_thr_copy_O.partition_D(sO);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // sO has the same size as sQ, so we don't need to sync here.
  if (Kernel_traits::Share_Q_K_smem) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  const index_t row_offset_o =
      binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
      m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
  const index_t row_offset_lse =
      params.unpadded_lse
          ? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb) +
                m_block * kBlockM
          : (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
  Tensor gO = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_stride(params.o_row_stride, _1{}));
  Tensor gLSE = make_tensor(
      make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) +
                    row_offset_lse),
      Shape<Int<kBlockM>>{},
      Stride<_1>{});

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO =
      gmem_thr_copy_O.partition_S(sO);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  Tensor caccO = make_identity_tensor(
      Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO);  // (MMA,MMA_M,MMA_K)
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  Tensor taccOcO_row =
      logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M
  if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSE(row) = lse(mi);
      }
    }
  }

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(
      make_shape(size<0>(sO), size<1>(sO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  Tensor tOcO =
      gmem_thr_copy_O.partition_D(cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
  }
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy</*Is_even_MN=*/false,
              Is_even_K,
              /*Clear_OOB_MN=*/false,
              /*Clear_OOB_K=*/false>(gmem_tiled_copy_O,
                                     tOrO,
                                     tOgO,
                                     tOcO,
                                     tOpO,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits,
          bool Is_dropout,
          bool Is_causal,
          bool Is_even_N,
          bool Is_even_K,
          bool Return_softmax,
          bool Is_equal_seq_qk,
          typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_flashmask(
    const Params &params, const int bidb, const int bidh, const int m_block) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  __shared__ int32_t flash_mask_ltstart_smem_[Kernel_traits::kBlockN];
  __shared__ int32_t flash_mask_ltend_smem_[Kernel_traits::kBlockN];
  __shared__ int32_t flash_mask_utstart_smem_[Kernel_traits::kBlockN];
  __shared__ int32_t flash_mask_utend_smem_[Kernel_traits::kBlockN];
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;
  // The global block index.
  const int block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int MMA_M =
      kBlockM / decltype(size<0>(
                    typename Kernel_traits::TiledMma::TiledShape_MNK{}))::value;

  const BlockInfo</*Varlen=*/!Is_even_N> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0)
    return;

  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  if (Is_causal) {
    n_block_max =
        std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM, kBlockN));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
    // }
  }

  // We iterate over the blocks in reverse order. This is because the last block
  // is the only one that needs masking when we read K and V from global memory.
  // Moreover, iterating in reverse might save us 1 register (we just need
  // n_block instead of both n_block and n_block_max).

  const index_t row_offset_q =
      binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
      m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
  // We move K and V to the last block.
  const index_t row_offset_k =
      binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
      (n_block_max - 1) * kBlockN * params.k_row_stride +
      (bidh / params.h_h_k_ratio) * params.k_head_stride;
  const index_t row_offset_v =
      binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
      (n_block_max - 1) * kBlockN * params.v_row_stride +
      (bidh / params.h_h_k_ratio) * params.v_head_stride;
  const index_t row_offset_p =
      ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) *
          params.seqlen_k_rounded +
      (n_block_max - 1) * kBlockN;

  const uint64_t row_offset_mask =
      (uint64_t)((bidb * params.mask_head_mod_size +
                  (bidh % params.mask_head_mod_size)) *
                     params.mask_seq_q_mod_size +
                 (m_block * kBlockM % params.mask_seq_q_mod_size)) *
          params.seqlen_k +
      (n_block_max - 1) * kBlockN;

  const index_t row_offset_sparse_mask =
      (bidb * params.h_sparsemask + bidh / params.h_h_sparsemask_ratio) *
          params.seqlen_k +
      (n_block_max - 1) * kBlockN;
  const index_t row_offset_sparsemask_nblock =
      (bidb * params.h_sparsemask + bidh / params.h_h_sparsemask_ratio) *
      cute::ceil_div(params.seqlen_k, kBlockN);

  Tensor gQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_stride(params.q_row_stride, _1{}));
  Tensor gK = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_stride(params.k_row_stride, _1{}));
  Tensor gV = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_stride(params.v_row_stride, _1{}));
  Tensor gP = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
      Shape<Int<kBlockM>, Int<kBlockN>>{},
      make_stride(params.seqlen_k_rounded, _1{}));

  Tensor gFlashMaskLTStart = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<int32_t *>(params.flashmask_downstart_ptr) +
          row_offset_sparse_mask),
      Shape<Int<kBlockN>>{});
  Tensor gFlashMaskLTEnd = make_tensor(
      make_gmem_ptr(reinterpret_cast<int32_t *>(params.flashmask_downend_ptr) +
                    row_offset_sparse_mask),
      Shape<Int<kBlockN>>{});
  Tensor gFlashMaskUTStart = make_tensor(
      make_gmem_ptr(reinterpret_cast<int32_t *>(params.flashmask_upstart_ptr) +
                    row_offset_sparse_mask),
      Shape<Int<kBlockN>>{});
  Tensor gFlashMaskUTEnd = make_tensor(
      make_gmem_ptr(reinterpret_cast<int32_t *>(params.flashmask_upend_ptr) +
                    row_offset_sparse_mask),
      Shape<Int<kBlockN>>{});
  const int *gFlashMaskLTStartMax =
      reinterpret_cast<int32_t *>(params.flashmask_downstart_nblockmax) +
      row_offset_sparsemask_nblock;
  const int *gFlashMaskLTStartMin =
      reinterpret_cast<int32_t *>(params.flashmask_downstart_nblockmin) +
      row_offset_sparsemask_nblock;
  const int *gFlashMaskLTEndMax =
      reinterpret_cast<int32_t *>(params.flashmask_downend_nblockmax) +
      row_offset_sparsemask_nblock;
  const int *gFlashMaskLTEndMin =
      reinterpret_cast<int32_t *>(params.flashmask_downend_nblockmin) +
      row_offset_sparsemask_nblock;
  const int *gFlashMaskUTStartMax =
      reinterpret_cast<int32_t *>(params.flashmask_upstart_nblockmax) +
      row_offset_sparsemask_nblock;
  const int *gFlashMaskUTStartMin =
      reinterpret_cast<int32_t *>(params.flashmask_upstart_nblockmin) +
      row_offset_sparsemask_nblock;
  const int *gFlashMaskUTEndMax =
      reinterpret_cast<int32_t *>(params.flashmask_upend_nblockmax) +
      row_offset_sparsemask_nblock;
  const int *gFlashMaskUTEndMin =
      reinterpret_cast<int32_t *>(params.flashmask_upend_nblockmin) +
      row_offset_sparsemask_nblock;

  const bool enable_mask_bypass = params.enable_mask_bypass;
  const bool flashmask_lt_has_end = params.flashmask_downend_ptr != nullptr;
  const bool flashmask_ut_has_start = params.flashmask_upstart_ptr != nullptr;

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
  Tensor sK =
      make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                  typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(
      sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
  Tensor sFlashMaskLTStart = make_tensor(
      make_smem_ptr(reinterpret_cast<int32_t *>(flash_mask_ltstart_smem_)),
      Shape<Int<kBlockN>>{});
  Tensor sFlashMaskLTEnd = make_tensor(
      make_smem_ptr(reinterpret_cast<int32_t *>(flash_mask_ltend_smem_)),
      Shape<Int<kBlockN>>{});
  Tensor sFlashMaskUTStart = make_tensor(
      make_smem_ptr(reinterpret_cast<int32_t *>(flash_mask_utstart_smem_)),
      Shape<Int<kBlockN>>{});
  Tensor sFlashMaskUTEnd = make_tensor(
      make_smem_ptr(reinterpret_cast<int32_t *>(flash_mask_utend_smem_)),
      Shape<Int<kBlockN>>{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  typename Kernel_traits::GmemTiledCopyP gmem_tiled_copy_P;
  auto gmem_thr_copy_P = gmem_tiled_copy_P.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
  Tensor tPgP = gmem_thr_copy_P.partition_D(gP);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)
  Tensor tOrVt =
      thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)

  Tensor acc_o = partition_fragment_C(
      tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  //
  // Copy Atom retiling
  //

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  // auto smem_thr_copy_Q = make_tiled_copy_A_warpcontiguousM<MMA_M>(typename
  // Kernel_traits::SmemCopyAtom{}, tiled_mma).get_thread_slice(tidx); if
  // (cute::thread0()) {smem_thr_copy_Q.print_all();}
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
  // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  // TODO: this might need to change if we change the mma instruction in SM70
  Tensor scores_max =
      make_tensor<ElementAccum>(Shape<Int<2 * size<1>(acc_o)>>{});
  Tensor scores_sum = make_fragment_like(scores_max);

  //
  // PREDICATES
  //

  // // Allocate predicate tensors for m and n
  // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)),
  // Stride<_1,_0>{}); Tensor tKVpKV =
  // make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)),
  // Stride<_1,_0>{});

  // Construct identity layout for sQ and sK
  Tensor cQ = make_identity_tensor(
      make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cKV = make_identity_tensor(
      make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
  // Tensor tScQ = thr_mma.partition_A(cQ);                           //
  // (MMA,MMA_M,MMA_K) if (cute::thread0()) {
  //     print(tScQ.layout()); printf("\n");
  //     for (int i = 0; i < size(tScQ); ++i) {
  //         printf("%d ", get<0>(tScQ(i)));
  //     }
  //     printf("\n");
  //     for (int i = 0; i < size(tScQ); ++i) {
  //         printf("%d ", get<1>(tScQ(i)));
  //     }
  //     printf("\n");
  // }

  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(
      cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(
      cKV);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

  // Allocate predicate tensors for k
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

  // Set predicates for k bounds
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
#pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) {
      tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
    }
  }

  // Prologue

  Tensor tQrQ = make_fragment_like(tQgQ);
  // We don't need to clear the sQ smem tiles since we'll only write out the
  // valid outputs
  flash::copy</*Is_even_MN=*/false, Is_even_K>(
      gmem_tiled_copy_QKV,
      tQgQ,
      tQsQ,
      tQcQ,
      tQpQ,
      binfo.actual_seqlen_q - m_block * kBlockM);
  if (Kernel_traits::Is_Q_in_regs) {
    cute::cp_async_fence();
  }

  // // Copy rmem to smem
  // // copy(tQrQ, tQsQ);
  // flash::cp_async_wait<0>();
  // __syncthreads();
  // // if (cute::thread(1, 0)) { print(tQsQ); }
  // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element
  // *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
  // // if (cute::thread0()) { print(sQNoSwizzle); }

  if (Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<0>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    __syncthreads();
  }

  int n_block = n_block_max - 1;
  // We don't need to clear the sK smem tiles since we'll mask out the scores
  // anyway.
  flash::copy<Is_even_N, Is_even_K>(gmem_tiled_copy_QKV,
                                    tKgK,
                                    tKsK,
                                    tKVcKV,
                                    tKVpKV,
                                    binfo.actual_seqlen_k - n_block * kBlockN);
  cute::cp_async_fence();
  // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
  // __syncthreads();

  if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<1>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
  }

  auto seeds = at::cuda::philox::unpack(params.philox_args);
  unsigned long long seed = std::get<0>(seeds);
  unsigned long long offset =
      std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + tidx % 32;

  // Save seed and offset for backward.
  if (block_id == 0 && tidx == 0) {
    params.rng_state[0] = seed;
    params.rng_state[1] = std::get<1>(seeds);
  }

  clear(acc_o);

  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not
  // multiple of kBlockN. We also need masking on S if it's causal, for the last
  // ceil_div(kBlockM, kBlockN) blocks. We will have at least 1 "masking"
  // iteration.

#define SPARSE_MASKED_DOWN(N_BLOCK)                            \
  (((m_block * kBlockM) >= gFlashMaskLTStartMax[(N_BLOCK)]) && \
   (!flashmask_lt_has_end ||                                   \
    (m_block + 1) * kBlockM <= gFlashMaskLTEndMin[(N_BLOCK)]))

#define SPARSE_MASKED_UP(N_BLOCK)                                            \
  (!Is_causal && (m_block + 1) * kBlockM <= gFlashMaskUTEndMin[(N_BLOCK)] && \
   (!flashmask_ut_has_start ||                                               \
    m_block * kBlockM >= gFlashMaskUTStartMax[(N_BLOCK)]))

#define SPARSE_MASKED(N_BLOCK) \
  (SPARSE_MASKED_DOWN(N_BLOCK) || SPARSE_MASKED_UP(N_BLOCK))

  constexpr int n_masking_steps =
      Is_causal ? cute::ceil_div(kBlockM, kBlockN) : 1;
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps;
       ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();

    // Advance gV
    if (masking_step > 0) {
      tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
    } else {
      // Clear the smem tiles to account for predicated off loads
      flash::copy<Is_even_N, Is_even_K, /*Clear_OOB_MN=*/true>(
          gmem_tiled_copy_QKV,
          tVgV,
          tVsV,
          tKVcKV,
          tKVpKV,
          binfo.actual_seqlen_k - n_block * kBlockN);
    }
    cute::cp_async_fence();

    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(acc_s,
                                                           tSrQ,
                                                           tSrK,
                                                           tSsQ,
                                                           tSsK,
                                                           tiled_mma,
                                                           smem_tiled_copy_Q,
                                                           smem_tiled_copy_K,
                                                           smem_thr_copy_Q,
                                                           smem_thr_copy_K);
    // if (cute::thread0()) { print(acc_s); }

    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2,
    // MMA_N))
    Tensor scores = make_tensor(
        acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

    // if (cute::thread0()) { print(scores); }
    // We don't put the masking before the matmul S = Q K^T because we don't
    // clear sK for rows outside actual_seqlen_k. So those rows could have Inf /
    // NaN, and the matmul can produce Inf / NaN.
    if (!Is_causal) {
      if (true /*Is_flashmask*/ &&
          (!enable_mask_bypass ||
           (((m_block + 1) * kBlockM > gFlashMaskLTStartMin[n_block] &&
             (!flashmask_ut_has_start ||
              m_block * kBlockM < gFlashMaskLTEndMax[n_block])) ||
            (m_block * kBlockM < gFlashMaskUTEndMax[n_block] &&
             (!flashmask_ut_has_start ||
              (m_block + 1) * kBlockM > gFlashMaskUTStartMin[n_block]))))) {
        if (tidx < kBlockN &&
            tidx + n_block * kBlockN < binfo.actual_seqlen_k) {
          sFlashMaskLTStart(tidx) = gFlashMaskLTStart(tidx);
          sFlashMaskUTEnd(tidx) = gFlashMaskUTEnd(tidx);
          if (flashmask_ut_has_start) {
            sFlashMaskUTStart(tidx) = gFlashMaskUTStart(tidx);
            sFlashMaskLTEnd(tidx) = gFlashMaskLTEnd(tidx);
          }
        }
        __syncthreads();
        if (flashmask_ut_has_start) {
          flash::apply_sparse_mask(
              scores,
              sFlashMaskLTStart,
              sFlashMaskLTEnd,
              n_block * kBlockN,
              binfo.actual_seqlen_k,
              m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
              kNWarps * 16,
              n_block * kBlockN,
              /*pairwise*/ true);
          flash::apply_sparse_mask(
              scores,
              sFlashMaskUTStart,
              sFlashMaskUTEnd,
              n_block * kBlockN,
              binfo.actual_seqlen_k,
              m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
              kNWarps * 16,
              n_block * kBlockN,
              /*pairwise*/ true);
        } else {
          flash::apply_sparse_mask(
              scores,
              sFlashMaskLTStart,
              sFlashMaskUTEnd,
              n_block * kBlockN,
              binfo.actual_seqlen_k,
              m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
              kNWarps * 16,
              n_block * kBlockN,
              /*pairwise*/ false);
        }
        // m_block * kBlockM + (tidx / 32) * 16, kNWarps * 16);
        // m_block * kBlockM + (tidx / 32) * (kBlockM / kNWarps), 16);
      } else if (!Is_even_N) {
        flash::apply_mask(scores, binfo.actual_seqlen_k - n_block * kBlockN);
      }
    } else {
      // Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>,
      // Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n) Tensor taccScS =
      // thr_mma.partition_C(caccS);                           //
      // (MMA,MMA_M,MMA_N) static_assert(decltype(size<0>(taccScS))::value ==
      // 4);
      // // Convert to ((2, 2), MMA_M, MMA_N) then take only the row indices.
      // Tensor idx_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _),
      // _, 0); Tensor idx_rowcol = make_tensor(taccScS.data(),
      // flash::convert_layout_acc_rowcol(taccScS.layout()));
      // flash::apply_mask_causal_w_idx(scores, idx_rowcol, n_block * kBlockN,
      // binfo.actual_seqlen_k,
      //                               m_block * kBlockM);
      // Idk why it's get<1> and not get<0> of the stride.
      // if (cute::thread0()) { print(idx_row.layout());
      // print(stride<1>(idx_row)); printf("stride = %d \n",
      // get<1>(stride<1>(idx_row))); } I can't get the stride from idx_row
      if (true /*Is_flashmask*/ &&
          (!enable_mask_bypass ||
           (m_block + 1) * kBlockM > gFlashMaskLTStartMin[n_block] &&
               (!flashmask_lt_has_end ||
                m_block * kBlockM < gFlashMaskLTEndMax[n_block]))) {
        if (tidx < kBlockN &&
            tidx + n_block * kBlockN < binfo.actual_seqlen_k) {
          sFlashMaskLTStart(tidx) = gFlashMaskLTStart(tidx);
          if (flashmask_lt_has_end) {
            sFlashMaskLTEnd(tidx) = gFlashMaskLTEnd(tidx);
          }
        }
        __syncthreads();
        if (flashmask_lt_has_end) {
          flash::apply_sparse_mask_causal_withend(
              scores,
              sFlashMaskLTStart,
              sFlashMaskLTEnd,
              n_block * kBlockN,
              binfo.actual_seqlen_k,
              // m_block * kBlockM + get<0>(idx_row(0)),
              m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
              kNWarps * 16,
              n_block * kBlockN);
        } else {
          flash::apply_sparse_mask_causal(
              scores,
              sFlashMaskLTStart,
              n_block * kBlockN,
              binfo.actual_seqlen_k,
              // m_block * kBlockM + get<0>(idx_row(0)),
              m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
              kNWarps * 16,
              n_block * kBlockN);
        }
        // m_block * kBlockM + (tidx / 32) * 16, kNWarps * 16);
        // m_block * kBlockM + (tidx / 32) * (kBlockM / kNWarps), 16);
      } else {
        flash::apply_mask_causal(
            scores,
            n_block * kBlockN,
            binfo.actual_seqlen_k,
            // m_block * kBlockM + get<0>(idx_row(0)),
            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
            kNWarps * 16);
        // m_block * kBlockM + (tidx / 32) * 16, kNWarps * 16);
        // m_block * kBlockM + (tidx / 32) * (kBlockM / kNWarps), 16);
      }
    }

    if (true /*Is_flashmask*/) {
      gFlashMaskLTStart.data() = gFlashMaskLTStart.data() + (-kBlockN);
      if (flashmask_lt_has_end) {
        gFlashMaskLTEnd.data() = gFlashMaskLTEnd.data() + (-kBlockN);
      }
      if (!Is_causal) {
        gFlashMaskUTEnd.data() = gFlashMaskUTEnd.data() + (-kBlockN);
        if (flashmask_ut_has_start) {
          gFlashMaskUTStart.data() = gFlashMaskUTStart.data() + (-kBlockN);
        }
      }
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > 0) {
      // Advance gK
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      if (true /*Is_flashmask*/ && enable_mask_bypass &&
          masking_step == n_masking_steps - 1) {
        auto in_block = n_block - 1;
        for (; in_block > 0 && SPARSE_MASKED(in_block); --in_block) {
          tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
        }
        __syncwarp();
      }
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // TODO: when we have key_padding_mask we'll need to Check_inf
    if (true /*Is_flashmask*/) {
      // We must check inf if use sparse_attn_mask
      masking_step == 0
          ? softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/true>(
                scores,
                scores_max,
                scores_sum,
                acc_o,
                params.scale_softmax_log2)
          : softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
                scores,
                scores_max,
                scores_sum,
                acc_o,
                params.scale_softmax_log2);
    } else {
      masking_step == 0
          ? softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal>(
                scores,
                scores_max,
                scores_sum,
                acc_o,
                params.scale_softmax_log2)
          : softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(
                scores,
                scores_max,
                scores_sum,
                acc_o,
                params.scale_softmax_log2);
    }
    // Convert scores from fp32 to fp16/bf16
    Tensor rP = flash::convert_type<Element>(scores);
    // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M,
    // MMA_N / 2) if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using
    // m16n8k8.
    Tensor tOrP =
        make_tensor(rP.data(),
                    flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(
                        rP.layout()));
    uint32_t block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
    uint32_t block_col_idx = n_block * (kBlockN / 32);
    if (Return_softmax) {
      Tensor tOrP_copy = make_fragment_like(tOrP);
      cute::copy(tOrP, tOrP_copy);
      flash::apply_dropout</*encode_dropout_in_sign_bit=*/true>(
          tOrP_copy,
          params.p_dropout_in_uint8_t,
          seed,
          offset,
          block_row_idx,
          block_col_idx,
          kNWarps);
      flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_tiled_copy_P);
      tPgP.data() = tPgP.data() + (-kBlockN);
    }
    if (Is_dropout) {
      flash::apply_dropout(tOrP,
                           params.p_dropout_in_uint8_t,
                           seed,
                           offset,
                           block_row_idx,
                           block_col_idx,
                           kNWarps);
    }
    // if (cute::thread0()) { print(tOrP); }

    flash::gemm_A_in_regs(acc_o,
                          tOrP,
                          tOrVt,
                          tOsVt,
                          tiled_mma,
                          smem_tiled_copy_V,
                          smem_thr_copy_V);
    // if (cute::thread0()) { print(scores); }

    // This check is at the end of the loop since we always have at least 1
    // iteration
    if (n_masking_steps > 1 && n_block <= 0) {
      --n_block;
      break;
    }
  }

  // These are the iterations where we don't need masking on S
  for (; n_block >= 0; --n_block) {
    if (true /*Is_flashmask*/ && enable_mask_bypass && SPARSE_MASKED(n_block)) {
      if (n_block == 0) {
        flash::cp_async_wait<0>();
        __syncthreads();
      }
      tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
      gFlashMaskLTStart.data() = gFlashMaskLTStart.data() + (-kBlockN);
      gFlashMaskLTEnd.data() = gFlashMaskLTEnd.data() + (-kBlockN);
      if (!Is_causal) {
        gFlashMaskUTEnd.data() = gFlashMaskUTEnd.data() + (-kBlockN);
        gFlashMaskUTStart.data() = gFlashMaskUTStart.data() + (-kBlockN);
      }
      if (Return_softmax) tPgP.data() = tPgP.data() + (-kBlockN);
      continue;
    }
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    clear(acc_s);
    flash::cp_async_wait<0>();
    __syncthreads();
    // Advance gV
    tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
    flash::copy</*Is_even_MN=*/true, Is_even_K>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
    cute::cp_async_fence();

    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(acc_s,
                                                           tSrQ,
                                                           tSrK,
                                                           tSsQ,
                                                           tSsK,
                                                           tiled_mma,
                                                           smem_tiled_copy_Q,
                                                           smem_tiled_copy_K,
                                                           smem_thr_copy_Q,
                                                           smem_thr_copy_K);

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > 0) {
      // Advance gK
      tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
      if (true /*Is_flashmask*/ && enable_mask_bypass) {
        auto in_block = n_block - 1;
        for (; in_block > 0 && SPARSE_MASKED(in_block); --in_block) {
          tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
        }
        __syncwarp();
      }
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      cute::cp_async_fence();
    }

    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2,
    // MMA_N))
    Tensor scores = make_tensor(
        acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

    if (!Is_causal && true /*Is_flashmask*/ &&
        (!enable_mask_bypass ||
         (((m_block + 1) * kBlockM > gFlashMaskLTStartMin[n_block] &&
           (!flashmask_ut_has_start ||
            m_block * kBlockM < gFlashMaskLTEndMax[n_block])) ||
          (m_block * kBlockM < gFlashMaskUTEndMax[n_block] &&
           (!flashmask_ut_has_start ||
            (m_block + 1) * kBlockM > gFlashMaskUTStartMin[n_block]))))) {
      if (tidx < kBlockN && tidx + n_block * kBlockN < binfo.actual_seqlen_k) {
        sFlashMaskLTStart(tidx) = gFlashMaskLTStart(tidx);
        sFlashMaskUTEnd(tidx) = gFlashMaskUTEnd(tidx);
        if (flashmask_ut_has_start) {
          sFlashMaskUTStart(tidx) = gFlashMaskUTStart(tidx);
          sFlashMaskLTEnd(tidx) = gFlashMaskLTEnd(tidx);
        }
      }
      __syncthreads();
      if (flashmask_ut_has_start) {
        flash::apply_sparse_mask(
            scores,
            sFlashMaskLTStart,
            sFlashMaskLTEnd,
            n_block * kBlockN,
            binfo.actual_seqlen_k,
            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
            kNWarps * 16,
            n_block * kBlockN,
            /*pairwise*/ true);
        flash::apply_sparse_mask(
            scores,
            sFlashMaskUTStart,
            sFlashMaskUTEnd,
            n_block * kBlockN,
            binfo.actual_seqlen_k,
            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
            kNWarps * 16,
            n_block * kBlockN,
            /*pairwise*/ true);
      } else {
        flash::apply_sparse_mask(
            scores,
            sFlashMaskLTStart,
            sFlashMaskUTEnd,
            n_block * kBlockN,
            binfo.actual_seqlen_k,
            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
            kNWarps * 16,
            n_block * kBlockN,
            /*pairwise*/ false);
      }
      // m_block * kBlockM + (tidx / 32) * 16, kNWarps * 16);
      // m_block * kBlockM + (tidx / 32) * (kBlockM / kNWarps), 16);

    } else if (Is_causal && true /*Is_flashmask*/ &&
               (!enable_mask_bypass ||
                (m_block + 1) * kBlockM > gFlashMaskLTStartMin[n_block] &&
                    (!flashmask_lt_has_end ||
                     m_block * kBlockM < gFlashMaskLTEndMax[n_block]))) {
      if (tidx < kBlockN && tidx + n_block * kBlockN < binfo.actual_seqlen_k) {
        sFlashMaskLTStart(tidx) = gFlashMaskLTStart(tidx);
        if (flashmask_lt_has_end) {
          sFlashMaskLTEnd(tidx) = gFlashMaskLTEnd(tidx);
        }
      }
      __syncthreads();
      if (flashmask_lt_has_end) {
        flash::apply_sparse_mask_causal_withend(
            scores,
            sFlashMaskLTStart,
            sFlashMaskLTEnd,
            n_block * kBlockN,
            binfo.actual_seqlen_k,
            // m_block * kBlockM + get<0>(idx_row(0)),
            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
            kNWarps * 16,
            n_block * kBlockN);
      } else {
        flash::apply_sparse_mask_causal(
            scores,
            sFlashMaskLTStart,
            n_block * kBlockN,
            binfo.actual_seqlen_k,
            // m_block * kBlockM + get<0>(idx_row(0)),
            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
            kNWarps * 16,
            n_block * kBlockN);
      }
      // m_block * kBlockM + (tidx / 32) * 16, kNWarps * 16);
      // m_block * kBlockM + (tidx / 32) * (kBlockM / kNWarps), 16);
    }

    if (true /*Is_flashmask*/) {
      gFlashMaskLTStart.data() = gFlashMaskLTStart.data() + (-kBlockN);
      if (flashmask_lt_has_end) {
        gFlashMaskLTEnd.data() = gFlashMaskLTEnd.data() + (-kBlockN);
      }
      if (!Is_causal) {
        gFlashMaskUTEnd.data() = gFlashMaskUTEnd.data() + (-kBlockN);
        if (flashmask_ut_has_start) {
          gFlashMaskUTStart.data() = gFlashMaskUTStart.data() + (-kBlockN);
        }
      }
    }

    if (true /*Is_flashmask*/) {
      softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
          scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);
    } else if (Is_equal_seq_qk) {
      softmax_rescale_o</*Is_first=*/false>(
          scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);
    } else {
      softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(
          scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);
    }

    Tensor rP = flash::convert_type<Element>(scores);
    // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M,
    // MMA_N / 2) if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using
    // m16n8k8.
    Tensor tOrP =
        make_tensor(rP.data(),
                    flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(
                        rP.layout()));
    uint32_t block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
    uint32_t block_col_idx = n_block * (kBlockN / 32);
    if (Return_softmax) {
      Tensor tOrP_copy = make_fragment_like(tOrP);
      cute::copy(tOrP, tOrP_copy);
      flash::apply_dropout</*encode_dropout_in_sign_bit=*/true>(
          tOrP_copy,
          params.p_dropout_in_uint8_t,
          seed,
          offset,
          block_row_idx,
          block_col_idx,
          kNWarps);
      flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_tiled_copy_P);
      tPgP.data() = tPgP.data() + (-kBlockN);
    }
    if (Is_dropout) {
      flash::apply_dropout(tOrP,
                           params.p_dropout_in_uint8_t,
                           seed,
                           offset,
                           block_row_idx,
                           block_col_idx,
                           kNWarps);
    }

    flash::gemm_A_in_regs(acc_o,
                          tOrP,
                          tOrVt,
                          tOsVt,
                          tiled_mma,
                          smem_tiled_copy_V,
                          smem_thr_copy_V);
  }

  // Epilogue

  // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2,
  // MMA_K))
  Tensor acc_o_rowcol = make_tensor(
      acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
  Tensor lse = make_fragment_like(scores_sum);
#pragma unroll
  for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    float sum = scores_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    lse(mi) = (sum == 0.f || sum != sum)
                  ? INFINITY
                  : scores_max(mi) * params.scale_softmax + __logf(sum);
    float scale = !Is_dropout ? inv_sum : inv_sum * params.rp_dropout;
#pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
      acc_o_rowcol(mi, ni) *= scale;
    }
  }

  // if (cute::thread0()) { print(acc_o_rowcol); }

  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = flash::convert_type<Element>(acc_o);
  Tensor sO = make_tensor(
      sQ.data(), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  // auto smem_thr_copy_O = make_tiled_copy_C_warpcontiguousM<MMA_M>(typename
  // Kernel_traits::SmemCopyAtomO{}, tiled_mma).get_thread_slice(tidx);
  Tensor taccOrO =
      smem_thr_copy_O.retile_S(rO);  // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO =
      smem_thr_copy_O.partition_D(sO);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // sO has the same size as sQ, so we don't need to sync here.
  if (Kernel_traits::Share_Q_K_smem) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  const index_t row_offset_o =
      binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
      m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
  const index_t row_offset_lse =
      params.unpadded_lse
          ? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb) +
                m_block * kBlockM
          : (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
  Tensor gO = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_stride(params.o_row_stride, _1{}));
  Tensor gLSE = make_tensor(
      make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) +
                    row_offset_lse),
      Shape<Int<kBlockM>>{},
      Stride<_1>{});

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO =
      gmem_thr_copy_O.partition_S(sO);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  Tensor caccO = make_identity_tensor(
      Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor taccOcO = thr_mma.partition_C(caccO);  // (MMA,MMA_M,MMA_K)
  static_assert(decltype(size<0>(taccOcO))::value == 4);
  // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
  Tensor taccOcO_row =
      logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M
  if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSE(row) = lse(mi);
      }
    }
  }

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(
      make_shape(size<0>(sO), size<1>(sO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  Tensor tOcO =
      gmem_thr_copy_O.partition_D(cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
  }
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy</*Is_even_MN=*/false,
              Is_even_K,
              /*Clear_OOB_MN=*/false,
              /*Clear_OOB_K=*/false>(gmem_tiled_copy_O,
                                     tOrO,
                                     tOgO,
                                     tOcO,
                                     tOpO,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits,
          bool Is_dropout,
          bool Is_causal,
          bool Is_even_N,
          bool Is_even_K,
          bool Return_softmax,
          bool Is_densemask,
          bool Is_flashmask,
          bool Is_equal_seq_qk,
          typename Params>
inline __device__ void compute_attn(const Params &params) {
  const int m_block = blockIdx.x;
  // The block index for the batch.
  const int bidb = blockIdx.y;
  // The block index for the head.
  const int bidh = blockIdx.z;

  // We want the fwd and bwd to generate the same dropout pattern (RNG), without
  // restricting them to have the same number of threads or have to traverse the
  // attention matrix in the same order. In the Philox RNG, we use the offset to
  // store the batch, head, and the lane id (within a warp). We use the
  // subsequence to store the location of the 16 x 32 blocks within the
  // attention matrix. This way, as long as we have the batch, head, and the
  // location of the 16 x 32 block within the attention matrix, we can generate
  // the exact same dropout pattern.

  if (Is_densemask) {
    flash::compute_attn_1rowblock_densemask<Kernel_traits,
                                            Is_dropout,
                                            Is_causal,
                                            Is_even_N,
                                            Is_even_K,
                                            Return_softmax,
                                            Is_equal_seq_qk>(
        params, bidb, bidh, m_block);
  } else if (Is_flashmask) {
    flash::compute_attn_1rowblock_flashmask<Kernel_traits,
                                            Is_dropout,
                                            Is_causal,
                                            Is_even_N,
                                            Is_even_K,
                                            Return_softmax,
                                            Is_equal_seq_qk>(
        params, bidb, bidh, m_block);
  } else {
    flash::compute_attn_1rowblock<Kernel_traits,
                                  Is_dropout,
                                  Is_causal,
                                  Is_even_N,
                                  Is_even_K,
                                  Return_softmax,
                                  Is_equal_seq_qk>(params, bidb, bidh, m_block);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
