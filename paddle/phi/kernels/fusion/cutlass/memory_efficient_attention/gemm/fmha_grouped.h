/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Grouped FMHA kernel
*/

#pragma once

#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"

#include "../epilogue/epilogue_rescale_output.h"
#include "../gemm_kernel_utils.h"
#include "./attention_scaling_coefs_updater.h"
#include "./fmha_grouped_problem_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MM0_,  ///! Structure for computing P = Q @ K
          typename MM1_,  ///! Structure for computing O = P @ V
          typename scalar_t_,
          typename accum_t_,
          typename output_t_,
          typename output_accum_t_,
          bool kKeepOutputInRF,  ///! Whether the intermediate output from MM0_
                                 /// should be kept in the register file
          GroupScheduleMode GroupScheduleMode_,  ///! Type of scheduling to
                                                 /// perform
          bool kAddMask,
          // This is quite faster when mask need broadcast at row axis
          bool kMaskIsAligned>
struct FMHAGrouped {
 public:
  using MM0 = MM0_;
  using MM1 = MM1_;

  using scalar_t = scalar_t_;
  using accum_t = accum_t_;
  using output_t = output_t_;
  using output_accum_t = output_accum_t_;

  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;

  static constexpr bool kNeedsOutputAccumulatorBuffer =
      !kKeepOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  // Parameters to satisfy BaseGrouped
  using ElementA = scalar_t;
  using ElementB = scalar_t;
  using ElementC = accum_t;
  using LayoutA = typename MM0::LayoutA;
  using LayoutB = typename MM0::ElementB;
  using LayoutC = typename MM1::ElementC;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static int const kAlignmentA = MM0::kAlignmentA;
  static int const kAlignmentB = MM0::kAlignmentB;
  static int const kAlignmentC = 1;
  using Mma = typename MM1::Mma;
  using EpilogueOutputOp = typename MM1::EpilogueOutputOp;
  using ThreadblockSwizzle = void;
  using Operator = typename MM1::Operator;
  using WarpShape = typename MM1::WarpShape;
  using InstructionShape = typename MM1::InstructionShape;

  using ElementQ = scalar_t;
  using ElementK = scalar_t;
  using ElementP = accum_t;
  using ElementM = scalar_t;
  using ElementV = scalar_t;
  using ElementO = output_t;
  using ElementOAccum = output_accum_t;
  using ElementAccumulator = accum_t;

  using LayoutQ = typename MM0::LayoutA;
  using LayoutK = typename MM0::LayoutB;
  using LayoutP = typename MM0::LayoutC;
  using LayoutM = typename MM0::LayoutC;
  using LayoutV = typename MM1::LayoutB;
  using LayoutO = typename MM1::LayoutC;

  static bool const kPreloadV =
      (MM1::Mma::ArchTag::kMinComputeCapability >= 80 &&
       cutlass::sizeof_bits<ElementV>::value == 16);

  static int const kAlignmentQ = MM0::kAlignmentA;
  static int const kAlignmentK = MM0::kAlignmentB;
  static int const kAlignmentV = 1;
  static int64_t const kAlignmentM = kMaskIsAligned ? kAlignmentQ : 1;

  using ThreadblockShape = typename MM0::ThreadblockShape;

  static int const kQueriesPerBlock = ThreadblockShape::kM;
  static int const kKeysPerBlock = ThreadblockShape::kN;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename MM1::WarpCount;
  static int const kThreadsPerWarp = 32;
  static int const kThreadCount = kThreadsPerWarp * WarpCount::kCount;

  using ProblemVisitor = FMHAGroupedProblemVisitor<ThreadblockShape,
                                                   kGroupScheduleMode,
                                                   kThreadCount,
                                                   kThreadCount>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //

    GemmCoord *problem_sizes0;
    GemmCoord *problem_sizes1;

    int problem_count;
    int threadblock_count;
    int num_heads;
    int kv_num_heads;

    ElementQ *ptr_Q;
    ElementK *ptr_K;
    ElementP *ptr_P;
    ElementM *ptr_M;
    ElementV *ptr_V;
    ElementO *ptr_O;
    ElementOAccum *ptr_O_accum;

    typename LayoutQ::Stride::LongIndex ldq;
    typename LayoutK::Stride::LongIndex ldk;
    typename LayoutK::Stride::LongIndex ldm;
    typename LayoutP::Stride::LongIndex ldv;
    typename LayoutO::Stride::LongIndex ldo;

    int64_t kElementQ;
    int64_t kElementK;
    int64_t kElementM;
    int64_t kElementV;
    int64_t kElementO;

    // Scale
    ElementAccumulator scale;

    // Whether causal masking is to be performed
    bool causal;
    bool mask_broadcast_head;

    // Only used by device-level operator
    GemmCoord *host_problem_sizes;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments()
        : problem_count(0),
          threadblock_count(0),
          num_heads(0),
          kv_num_heads(0),
          ptr_Q(nullptr),
          ptr_K(nullptr),
          ptr_P(nullptr),
          ptr_M(nullptr),
          ptr_V(nullptr),
          ptr_O(nullptr),
          ptr_O_accum(nullptr),
          ldq(0),
          ldk(0),
          ldm(0),
          ldv(0),
          ldo(0),
          scale(0),
          kElementQ(0),
          kElementK(0),
          kElementM(0),
          kElementV(0),
          kElementO(0),
          causal(false),
          mask_broadcast_head(true),
          host_problem_sizes(nullptr) {}

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(GemmCoord *problem_sizes0,
              GemmCoord *problem_sizes1,
              int problem_count,
              int threadblock_count,
              int num_heads,
              int kv_num_heads,
              ElementQ *ptr_Q,
              ElementK *ptr_K,
              ElementM *ptr_M,
              ElementV *ptr_V,
              ElementO *ptr_O,
              ElementOAccum *ptr_O_accum,
              typename LayoutQ::Stride::LongIndex ldq,
              typename LayoutK::Stride::LongIndex ldk,
              typename LayoutM::Stride::LongIndex ldm,
              typename LayoutV::Stride::LongIndex ldv,
              typename LayoutO::Stride::LongIndex ldo,
              int64_t kElementQ,
              int64_t kElementK,
              int64_t kElementM,
              int64_t kElementV,
              int64_t kElementO,
              bool causal,
              bool mask_broadcast_head,
              ElementAccumulator scale,
              GemmCoord *host_problem_sizes = nullptr)
        : problem_sizes0(problem_sizes0),
          problem_sizes1(problem_sizes1),
          problem_count(problem_count),
          threadblock_count(threadblock_count),
          num_heads(num_heads),
          kv_num_heads(kv_num_heads),
          ptr_Q(ptr_Q),
          ptr_K(ptr_K),
          ptr_M(ptr_M),
          ptr_V(ptr_V),
          ptr_O(ptr_O),
          ptr_O_accum(kNeedsOutputAccumulatorBuffer
                          ? ptr_O_accum
                          : reinterpret_cast<accum_t *>(ptr_O)),  // wip
          ldq(ldq),
          ldk(ldk),
          ldm(ldm),
          ldv(ldv),
          ldo(ldo),
          kElementQ(kElementQ),
          kElementK(kElementK),
          kElementM(kElementM),
          kElementV(kElementV),
          kElementO(kElementO),
          causal(causal),
          mask_broadcast_head(mask_broadcast_head),
          scale(scale),
          host_problem_sizes(host_problem_sizes) {}

    bool __host__ check_supported() {
      CHECK_ALIGNED_PTR(ptr_Q, kAlignmentQ);
      CHECK_ALIGNED_PTR(ptr_K, kAlignmentK);
      CHECK_ALIGNED_PTR(ptr_V, kAlignmentV);
      if (ptr_M != nullptr) {
        CHECK_ALIGNED_PTR(ptr_M, kAlignmentM);
        XFORMERS_CHECK(ldm % kAlignmentM == 0,
                       "attn_mask is not correctly aligned");
      }
      XFORMERS_CHECK(ldq % kAlignmentQ == 0, "query is not correctly aligned");
      XFORMERS_CHECK(ldk % kAlignmentK == 0, "key is not correctly aligned");
      XFORMERS_CHECK(ldv % kAlignmentV == 0, "value is not correctly aligned");
      return true;
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {
    typename ProblemVisitor::Params problem_visitor;
    int threadblock_count;
    int num_heads;
    int kv_num_heads;

    ElementQ *ptr_Q;
    ElementK *ptr_K;
    ElementP *ptr_P;
    ElementM *ptr_M;
    ElementV *ptr_V;
    ElementO *ptr_O;
    ElementOAccum *ptr_O_accum;

    typename LayoutQ::Stride::LongIndex ldq;
    typename LayoutK::Stride::LongIndex ldk;
    typename LayoutM::Stride::LongIndex ldm;
    typename LayoutP::Stride::LongIndex ldv;
    typename LayoutO::Stride::LongIndex ldo;

    int64_t kElementQ;
    int64_t kElementK;
    int64_t kElementM;
    int64_t kElementV;
    int64_t kElementO;

    ElementAccumulator scale;
    bool causal;
    bool mask_broadcast_head;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : ptr_Q(nullptr),
          ptr_K(nullptr),
          ptr_P(nullptr),
          ptr_M(nullptr),
          ptr_V(nullptr),
          ptr_O(nullptr),
          ptr_O_accum(nullptr),
          ldq(0),
          ldk(0),
          ldm(0),
          ldv(0),
          ldo(0),
          kElementQ(0),
          kElementK(0),
          kElementM(0),
          kElementV(0),
          kElementO(0),
          causal(false),
          mask_broadcast_head(true),
          scale(0) {}

    explicit CUTLASS_HOST_DEVICE Params(Arguments const &args,
                                        void *workspace = nullptr,
                                        int tile_count = 0)
        : problem_visitor(args.problem_sizes0,
                          args.problem_sizes1,
                          args.problem_count,
                          workspace,
                          tile_count),
          threadblock_count(args.threadblock_count),
          num_heads(args.num_heads),
          kv_num_heads(args.kv_num_heads),
          ptr_Q(args.ptr_Q),
          ptr_K(args.ptr_K),
          ptr_P(args.ptr_P),
          ptr_M(args.ptr_M),
          ptr_V(args.ptr_V),
          ptr_O(args.ptr_O),
          ptr_O_accum(kNeedsOutputAccumulatorBuffer
                          ? args.ptr_O_accum
                          : reinterpret_cast<accum_t *>(args.ptr_O)),
          ldq(args.ldq),
          ldk(args.ldk),
          ldm(args.ldm),
          ldv(args.ldv),
          ldo(args.ldo),
          kElementQ(args.kElementQ),
          kElementK(args.kElementK),
          kElementM(args.kElementM),
          kElementV(args.kElementV),
          kElementO(args.kElementO),
          causal(args.causal),
          mask_broadcast_head(args.mask_broadcast_head),
          scale(args.scale) {}

    //   CUTLASS_HOST_DEVICE
    void update(Arguments const &args,
                void *workspace = nullptr,
                int tile_count = 0) {
      problem_visitor = typename ProblemVisitor::Params(args.problem_sizes0,
                                                        args.problem_sizes1,
                                                        args.problem_count,
                                                        workspace,
                                                        tile_count);
      threadblock_count = args.threadblock_count;
      num_heads = args.num_heads;
      kv_num_heads = args.kv_num_heads;
      ptr_Q = args.ptr_Q;
      ptr_K = args.ptr_K;
      ptr_P = args.ptr_P;
      ptr_M = args.ptr_M;
      ptr_V = args.ptr_V;
      ptr_O = args.ptr_O;
      ptr_O_accum = kNeedsOutputAccumulatorBuffer
                        ? args.ptr_O_accum
                        : reinterpret_cast<accum_t *>(args.ptr_O);
      ldq = args.ldq;
      ldk = args.ldk;
      ldm = args.ldm;
      ldv = args.ldv;
      ldo = args.ldo;
      kElementQ = args.kElementQ;
      kElementK = args.kElementK;
      kElementM = args.kElementM;
      kElementV = args.kElementV;
      kElementO = args.kElementO;
      causal = args.causal;
      scale = args.scale;
    }
  };

  // Shared storage - depends on kernel params
  struct ScalingCoefs {
    cutlass::Array<ElementAccumulator, kQueriesPerBlock> m_prime;
    cutlass::Array<ElementAccumulator, kQueriesPerBlock> s_prime;
    cutlass::Array<ElementAccumulator, kQueriesPerBlock> mi;
    cutlass::Array<ElementAccumulator,
                   kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>
        addition_storage;
  };

  struct SharedStorageEpilogueAtEnd : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      union {
        typename MM0::MaskLoader::SmemTile mask;
        typename MM0::AccumulatorSharedStorage si;
      };
      typename MM1::SharedStorageMM1 mm1;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage &
    epilogue_shared_storage() {
      return epilogue;
    }

    // ProblemVisitor shared storage can't be overlapped with others
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

  struct SharedStorageEpilogueInLoop : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      union {
        typename MM0::MaskLoader::SmemTile mask;
        typename MM0::AccumulatorSharedStorage si;
      };
      typename MM1::SharedStorageMM1 mm1;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage &
    epilogue_shared_storage() {
      return after_mm0.epilogue;
    }

    // ProblemVisitor shared storage can't be overlapped with others
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

  using SharedStorage = typename cutlass::platform::conditional<
      kKeepOutputInRF,
      SharedStorageEpilogueAtEnd,
      SharedStorageEpilogueInLoop>::type;

 private:
  // Parameters to be used by an individual tile
  struct TileParams {
    CUTLASS_HOST_DEVICE
    static int query_start(int threadblock_idx) {
      return threadblock_idx * kQueriesPerBlock;
    }

    // Returns whether this threadblock computes within the number of queries,
    // which is determined by the M dimension of problem 0
    CUTLASS_HOST_DEVICE
    static bool can_compute(int threadblock_idx,
                            const GemmCoord &problem_size0) {
      return query_start(threadblock_idx) < problem_size0.m();
    }

    CUTLASS_HOST_DEVICE
    static int num_queries(int threadblock_idx,
                           const GemmCoord &problem_size0) {
      return problem_size0.m() - query_start(threadblock_idx);
    }

    CUTLASS_HOST_DEVICE
    static int num_keys(int threadblock_idx,
                        const GemmCoord &problem_size0,
                        bool causal) {
      int nk = problem_size0.n();
      if (causal) {
        nk = cutlass::fast_min(
            int32_t(query_start(threadblock_idx) + kQueriesPerBlock), nk);
      }
      return nk;
    }
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  FMHAGrouped() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const &problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  static CUTLASS_DEVICE int16_t thread_id() { return threadIdx.x; }

  static CUTLASS_DEVICE int8_t warp_id() {
    return threadIdx.x / kThreadsPerWarp;
  }

  static CUTLASS_DEVICE int8_t lane_id() {
    return threadIdx.x % kThreadsPerWarp;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params,
                  SharedStorage &shared_storage) {  // NOLINT
    auto &m_prime = shared_storage.m_prime;
    auto &s_prime = shared_storage.s_prime;
    [[maybe_unused]] auto &si = shared_storage.after_mm0.si;
    auto &mi = shared_storage.mi;

    ProblemVisitor problem_visitor(
        params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

    // Outer 'persistent' loop to iterate over tiles
    while (problem_visitor.next_tile()) {
      GemmCoord problem_size0 = problem_visitor.problem_size0();
      GemmCoord problem_size1 = problem_visitor.problem_size1();
      const int32_t threadblock_idx =
          int32_t(problem_visitor.threadblock_idx());

      if (!TileParams::can_compute(threadblock_idx, problem_size0)) {
        problem_visitor.advance(gridDim.x);
        continue;
      }

      const int32_t problem_idx = problem_visitor.problem_index();
      const int32_t batch_idx = problem_idx / params.num_heads;
      // how many query head share a kv head?
      const int32_t qhead_per_kv_head = params.num_heads / params.kv_num_heads;

      if (thread_id() < kQueriesPerBlock) {
        s_prime[thread_id()] = ElementAccumulator(0);
        m_prime[thread_id()] =
            -cutlass::platform::numeric_limits<ElementAccumulator>::infinity();
        mi[thread_id()] =
            -cutlass::platform::numeric_limits<ElementAccumulator>::infinity();
      }

      ElementO *ptr_O = params.ptr_O + problem_idx * params.kElementQ +
                        TileParams::query_start(threadblock_idx) * params.ldo;
      ElementOAccum *ptr_O_accum =
          params.ptr_O_accum + problem_idx * params.kElementO +
          TileParams::query_start(threadblock_idx) * params.ldo;
      const int num_queries =
          TileParams::num_queries(threadblock_idx, problem_size0);

      auto createOutputIter = [&](int col) -> typename MM1::OutputTileIterator {
        using OutputTileIterator = typename MM1::OutputTileIterator;
        return OutputTileIterator(
            typename OutputTileIterator::Params{(int32_t)params.ldo},
            ptr_O,
            typename OutputTileIterator::TensorCoord{num_queries,
                                                     problem_size1.n()},
            thread_id(),
            {0, col});
      };

      auto createOutputAccumIter =
          [&](int col) -> typename MM1::OutputTileIteratorAccum {
        using OutputTileIteratorAccum = typename MM1::OutputTileIteratorAccum;
        return OutputTileIteratorAccum(
            typename OutputTileIteratorAccum::Params{(int32_t)params.ldo},
            ptr_O_accum,
            typename OutputTileIteratorAccum::TensorCoord{num_queries,
                                                          problem_size1.n()},
            thread_id(),
            {0, col});
      };

      typename MM1::Mma::FragmentC accum_o;
      accum_o.clear();

      const int num_keys =
          TileParams::num_keys(threadblock_idx, problem_size0, params.causal);

      for (int32_t iter_key_start = 0; iter_key_start < num_keys;
           iter_key_start += kKeysPerBlock) {
        int32_t problem_size_0_m =
            cutlass::fast_min((int32_t)kQueriesPerBlock, num_queries);
        int32_t problem_size_0_n = cutlass::fast_min((int32_t)kKeysPerBlock,
                                                     num_keys - iter_key_start);
        int32_t const &problem_size_0_k = problem_size0.k();
        int32_t const &problem_size_1_n = problem_size1.n();
        int32_t const &problem_size_1_k = problem_size_0_n;

        auto prologueV = [&](int blockN) {
          typename MM1::Mma::IteratorB iterator_V(
              typename MM1::IteratorB::Params{MM1::LayoutB(params.ldv)},
              params.ptr_V +
                  (problem_idx / qhead_per_kv_head) * params.kElementV +
                  iter_key_start * params.ldv,
              {problem_size_1_k, problem_size_1_n},
              thread_id(),
              cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});

          MM1::Mma::prologue(shared_storage.after_mm0.mm1.mm,
                             iterator_V,
                             thread_id(),
                             problem_size_1_k);
        };

        __syncthreads();  // Need to have shared memory initialized, and
                          // `m_prime` updated from end of prev iter

        //
        // MATMUL: Q.K_t
        //
        // Computes the block-matrix product of:
        // (a) query[query_start:query_end, :]
        // with
        // (b) key[iter_key_start:iter_key_start + kKeysPerBlock]
        // and stores that into `shared_storage.si`
        //

        ElementQ *ptr_Q = params.ptr_Q + problem_idx * params.kElementQ +
                          TileParams::query_start(threadblock_idx) * params.ldq;

        // Construct iterators to A and B operands
        typename MM0::IteratorA iterator_A(
            typename MM0::IteratorA::Params(
                typename MM0::MmaCore::LayoutA(params.ldq)),
            ptr_Q,
            {problem_size_0_m, problem_size_0_k},
            thread_id(),
            {0, 0});

        typename MM0::IteratorB iterator_B(
            typename MM0::IteratorB::Params(
                typename MM0::MmaCore::LayoutB(params.ldk)),
            params.ptr_K +
                (problem_idx / qhead_per_kv_head) * params.kElementK +
                iter_key_start * params.ldk,
            {problem_size_0_k, problem_size_0_n},
            thread_id(),
            {0, 0});

        // Construct thread-scoped matrix multiply
        typename MM0::Mma mma(
            shared_storage.mm0, thread_id(), warp_id(), lane_id());

        typename MM0::Mma::FragmentC accum;

        accum.clear();

        auto gemm_k_iterations =
            (problem_size_0_k + MM0::Mma::Shape::kK - 1) / MM0::Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
        __syncthreads();

        if (kPreloadV) {
          prologueV(0);
        }

        typename MM0::Mma::Operator::IteratorC::TensorCoord
            iteratorC_tile_offset = {(warp_id() % MM0::Mma::WarpCount::kM),
                                     (warp_id() / MM0::Mma::WarpCount::kM)};

        // apply attention mask if applicable
        if (kAddMask) {
          const int mask_id =
              params.mask_broadcast_head ? batch_idx : problem_idx;
          accum = cutlass::multiplies<typename MM0::Mma::FragmentC>()(
              params.scale, accum);
          // load mask tile Bij into shared memory
          typename MM0::MaskLoader::GmemTileIterator mask_iter(
              {cutlass::layout::RowMajor(params.ldm)},
              // attn_mask_pointer points to matrix of size (n_queries, n_keys)
              // for the relevant batch_id and head_id
              params.ptr_M + mask_id * params.kElementM +
                  TileParams::query_start(threadblock_idx) * params.ldm +
                  iter_key_start,
              {problem_size_0_m, problem_size_0_n},
              thread_id());
          cutlass::TensorRef<ElementM, cutlass::layout::RowMajor>
              mask_tensor_ref(
                  shared_storage.after_mm0.mask.data(),
                  cutlass::layout::RowMajor(MM0::ThreadblockShape::kN));
          typename MM0::MaskLoader::SmemTileIterator smem_tile_iter(
              mask_tensor_ref, thread_id());
          MM0::MaskLoader::load(mask_iter, smem_tile_iter);

          // Pij += Bij, Pij is in register fragment and Bij is in shared memory
          auto lane_offset = MM0::ScalingCoefsUpdater::get_lane_offset(
              lane_id(), warp_id(), iteratorC_tile_offset);

          MM0::ScalingCoefsUpdater::iterateRows(
              lane_offset,
              [&](int accum_m) {},
              [&](int accum_m, int accum_n, int idx) {
                if (accum_m < problem_size_0_m && accum_n < problem_size_0_n) {
                  accum[idx] += mask_tensor_ref.at({accum_m, accum_n});
                }
              },
              [&](int accum_m) {});
        }

        // Mask out last if causal
        if (params.causal && num_keys - iter_key_start <= kKeysPerBlock) {
          auto lane_offset = MM0::ScalingCoefsUpdater::get_lane_offset(
              lane_id(), warp_id(), iteratorC_tile_offset);
          int32_t last_col;
          MM0::ScalingCoefsUpdater::iterateRows(
              lane_offset,
              [&](int accum_m) {
                last_col = TileParams::query_start(threadblock_idx) + accum_m -
                           iter_key_start;
              },
              [&](int accum_m, int accum_n, int idx) {
                if (accum_n > last_col) {
                  accum[idx] =
                      -cutlass::platform::numeric_limits<accum_t>::infinity();
                }
              },
              [&](int accum_m) {});
        }
        DISPATCH_BOOL(iter_key_start == 0, kIsFirst, ([&] {
                        DISPATCH_BOOL(
                            num_keys - iter_key_start >= kKeysPerBlock,
                            kFullColumns,
                            ([&] {
                              // Update `mi` from accum stored in registers
                              // Also updates `accum` with accum[i] <-
                              // exp(accum[i] * scale
                              // - mi)
                              MM0::ScalingCoefsUpdater::update<
                                  kQueriesPerBlock,
                                  MM0::MmaCore::WarpCount::kCount,
                                  MM0::MmaCore::WarpCount::kN,
                                  kFullColumns,
                                  kIsFirst,
                                  kKeepOutputInRF>(
                                  accum_o,
                                  accum,
                                  mi,
                                  m_prime,
                                  s_prime,
                                  shared_storage.addition_storage,
                                  lane_id(),
                                  thread_id(),
                                  warp_id(),
                                  num_keys - iter_key_start,
                                  iteratorC_tile_offset,
                                  kAddMask ? 1.0f : params.scale);
                            }));
                      }));

        // Output results to shared-memory
        int warp_idx_mn_0 = warp_id() % (MM0::Mma::Base::WarpCount::kM *
                                         MM0::Mma::Base::WarpCount::kN);
        auto output_tile_coords =
            cutlass::MatrixCoord{warp_idx_mn_0 % MM0::Mma::Base::WarpCount::kM,
                                 warp_idx_mn_0 / MM0::Mma::Base::WarpCount::kM};

        MM0::B2bGemm::accumToSmem(
            shared_storage.after_mm0.si, accum, lane_id(), output_tile_coords);

        __syncthreads();

        //
        // MATMUL: Attn . V
        // Run the matmul `attn @ V` for a block of attn and V.
        // `attn` is read from shared memory (in `shared_storage_si`)
        // `V` is read from global memory (with iterator_B)
        //

        const int64_t nBlockN =
            kKeepOutputInRF ? 1
                            : ceil_div((int64_t)problem_size_1_n,
                                       int64_t(MM1::ThreadblockShape::kN));

        // Iterate over the N dimension of GEMM1
        for (int blockN = 0; blockN < nBlockN; ++blockN) {
          int gemm_k_iterations = (problem_size_1_k + MM1::Mma::Shape::kK - 1) /
                                  MM1::Mma::Shape::kK;

          // Compute threadblock-scoped matrix multiply-add and store it in
          // accum (in registers)
          if (!kPreloadV) {
            __syncthreads();  // we share shmem between mma and epilogue
          }

          typename MM1::Mma::IteratorB iterator_V(
              typename MM1::IteratorB::Params{MM1::LayoutB(params.ldv)},
              params.ptr_V +
                  (problem_idx / qhead_per_kv_head) * params.kElementV +
                  iter_key_start * params.ldv,
              {problem_size_1_k, problem_size_1_n},
              thread_id(),
              cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});

          typename MM1::Mma mma_pv(shared_storage.after_mm0.mm1.mm,
                                   shared_storage.after_mm0.si,
                                   static_cast<int>(thread_id()),
                                   static_cast<int>(warp_id()),
                                   static_cast<int>(lane_id()),
                                   static_cast<int>(problem_size_1_k));

          mma_pv.set_prologue_done(kPreloadV);
          if (!kKeepOutputInRF) {
            accum_o.clear();
          }

          mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);
          __syncthreads();

          if (kPreloadV && !kKeepOutputInRF && blockN + 1 < nBlockN) {
            prologueV(blockN + 1);
          }

          if (!kKeepOutputInRF) {
            DISPATCH_BOOL(
                iter_key_start == 0, kIsFirst, ([&] {
                  DISPATCH_BOOL(
                      (iter_key_start + kKeysPerBlock) >= num_keys,
                      kIsLast,
                      ([&] {
                        using DefaultEpilogue = typename MM1::DefaultEpilogue;
                        using DefaultOp =
                            typename MM1::DefaultConfig::EpilogueOutputOp;
                        using ElementCompute =
                            typename DefaultOp::ElementCompute;
                        using EpilogueOutputOp = typename cutlass::epilogue::
                            thread::MemoryEfficientAttentionNormalize<
                                typename cutlass::platform::conditional<
                                    kIsLast,
                                    output_t,
                                    output_accum_t>::type,
                                output_accum_t,
                                DefaultOp::kCount,
                                typename DefaultOp::ElementAccumulator,
                                output_accum_t,
                                kIsFirst,
                                kIsLast,
                                cutlass::Array<ElementCompute,
                                               kQueriesPerBlock>>;
                        using Epilogue = typename cutlass::epilogue::
                            threadblock::EpiloguePipelined<
                                typename DefaultEpilogue::Shape,
                                typename MM1::Mma::Operator,
                                DefaultEpilogue::kPartitionsK,
                                typename cutlass::platform::conditional<
                                    kIsLast,
                                    typename MM1::OutputTileIterator,
                                    typename MM1::OutputTileIteratorAccum>::
                                    type,
                                typename DefaultEpilogue::
                                    AccumulatorFragmentIterator,
                                typename DefaultEpilogue::WarpTileIterator,
                                typename DefaultEpilogue::SharedLoadIterator,
                                EpilogueOutputOp,
                                typename DefaultEpilogue::Padding,
                                DefaultEpilogue::kFragmentsPerIteration,
                                true,  // IterationsUnroll
                                typename MM1::
                                    OutputTileIteratorAccum  // Read
                                                             // iterator
                                >;

                        int col = blockN * MM1::Mma::Shape::kN;
                        auto source_iter = createOutputAccumIter(col);
                        auto dest_iter = gemm_kernel_utils::call_conditional<
                            kIsLast,
                            decltype(createOutputIter),
                            decltype(createOutputAccumIter)>::
                            apply(createOutputIter, createOutputAccumIter, col);
                        EpilogueOutputOp rescale(s_prime, m_prime);
                        Epilogue epilogue(
                            shared_storage.epilogue_shared_storage(),
                            thread_id(),
                            warp_id(),
                            lane_id());
                        epilogue(rescale, dest_iter, accum_o, source_iter);
                      }));
                }));
            if (!kKeepOutputInRF) {
              __syncthreads();
            }
          }
        }
        __syncthreads();  // we modify `m_prime` after
      }

      if (kKeepOutputInRF) {
        const bool kIsFirst = true;
        const bool kIsLast = true;
        using DefaultEpilogue = typename MM1::DefaultEpilogue;
        using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
        using ElementCompute = typename DefaultOp::ElementCompute;
        using EpilogueOutputOp = typename cutlass::epilogue::thread::
            MemoryEfficientAttentionNormalize<
                output_t,        // output
                output_accum_t,  // source
                DefaultOp::kCount,
                typename DefaultOp::ElementAccumulator,  // accum
                output_accum_t,                          // compute
                kIsFirst,
                kIsLast,
                cutlass::Array<ElementCompute, kQueriesPerBlock>>;
        using Epilogue =
            typename cutlass::epilogue::threadblock::EpiloguePipelined<
                typename DefaultEpilogue::Shape,
                typename MM1::Mma::Operator,
                DefaultEpilogue::kPartitionsK,
                typename MM1::OutputTileIterator,  // destination
                typename DefaultEpilogue::AccumulatorFragmentIterator,
                typename DefaultEpilogue::WarpTileIterator,
                typename DefaultEpilogue::SharedLoadIterator,
                EpilogueOutputOp,
                typename DefaultEpilogue::Padding,
                DefaultEpilogue::kFragmentsPerIteration,
                true,                                  // IterationsUnroll
                typename MM1::OutputTileIteratorAccum  // source tile
                >;
        auto dest_iter = createOutputIter(0);
        EpilogueOutputOp rescale(s_prime, m_prime);
        Epilogue epilogue(shared_storage.epilogue_shared_storage(),
                          thread_id(),
                          warp_id(),
                          lane_id());
        epilogue(rescale, dest_iter, accum_o);
      }

      // Next tile
      problem_visitor.advance(gridDim.x);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
