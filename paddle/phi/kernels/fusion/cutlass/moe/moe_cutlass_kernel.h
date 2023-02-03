/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief
*/

#pragma once

#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/grouped_problem_visitor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/// Visitor class to abstract away the algorithm for iterating over tiles
template <typename ProblemSizeHelper, typename ThreadblockShape_>
struct BaseMoeProblemVisitor {
  using ThreadblockShape = ThreadblockShape_;

  struct ProblemInfo {
    static int32_t const kNoPrefetchEntry = -1;
    int32_t problem_idx;
    int32_t problem_start;

    CUTLASS_DEVICE
    ProblemInfo()
        : problem_idx(kNoPrefetchEntry), problem_start(kNoPrefetchEntry) {}

    CUTLASS_DEVICE
    ProblemInfo(int32_t problem_idx_, int32_t problem_start_)
        : problem_idx(problem_idx_), problem_start(problem_start_) {}
  };

  struct Params {
    int64_t const *last_row_for_problem;
    int64_t gemm_n;
    int64_t gemm_k;
    int32_t problem_count;
    void const *workspace;
    int32_t tile_count;

    //
    // Methods
    //

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params()
        : last_row_for_problem(nullptr),
          gemm_n(0),
          gemm_k(0),
          problem_count(0),
          workspace(nullptr),
          tile_count(0) {}

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(int64_t const *last_row_for_problem,
           int64_t gemm_n,
           int64_t gemm_k,
           int32_t problem_count,
           void const *workspace = nullptr,
           int32_t tile_count = 0)
        : last_row_for_problem(last_row_for_problem),
          gemm_n(gemm_n),
          gemm_k(gemm_k),
          problem_count(problem_count),
          workspace(workspace),
          tile_count(tile_count) {}
  };

  Params const &params;
  int32_t tile_idx;
  int32_t problem_tile_start;
  int32_t problem_idx;

  //
  // Methods
  //
  CUTLASS_DEVICE
  BaseMoeProblemVisitor(Params const &params_, int32_t block_idx)
      : params(params_),
        tile_idx(block_idx),
        problem_tile_start(0),
        problem_idx(0) {}

  /// Get the grid shape
  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord grid_shape(
      const cutlass::gemm::GemmCoord &problem) {
    return cutlass::gemm::GemmCoord(
        ((problem.m() - 1 + ThreadblockShape::kM) / ThreadblockShape::kM),
        ((problem.n() - 1 + ThreadblockShape::kN) / ThreadblockShape::kN),
        1);
  }

  /// Gets the global tile index
  CUTLASS_HOST_DEVICE
  int32_t tile_index() const { return tile_idx; }

  /// Gets the index of the problem
  CUTLASS_HOST_DEVICE
  int32_t problem_index() const { return problem_idx; }

  CUTLASS_HOST_DEVICE
  int32_t threadblock_idx() const { return tile_idx - problem_tile_start; }

  CUTLASS_DEVICE
  void advance(int32_t grid_size) { tile_idx += grid_size; }

  CUTLASS_HOST_DEVICE
  static void possibly_transpose_problem(
      cutlass::gemm::GemmCoord &problem) {  // NOLINT
    ProblemSizeHelper::possibly_transpose_problem(problem);
  }

  /// Returns the problem size for the current problem
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size() const {
    return problem_size(problem_idx);
  }

  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size(int idx) const {
    const int64_t prev_problem_row =
        idx == 0 ? 0 : params.last_row_for_problem[idx - 1];
    const int64_t current_problem_row = params.last_row_for_problem[idx];
    const int64_t gemm_m = current_problem_row - prev_problem_row;
    GemmCoord problem(GemmCoord::Index(gemm_m),
                      GemmCoord::Index(params.gemm_n),
                      GemmCoord::Index(params.gemm_k));
    ProblemSizeHelper::possibly_transpose_problem(problem);
    return problem;
  }

  CUTLASS_HOST_DEVICE
  static int32_t tile_count(const cutlass::gemm::GemmCoord &grid) {
    return ProblemSizeHelper::tile_count(grid);
  }

  static int32_t group_tile_count(
      const cutlass::gemm::GemmCoord *host_problem_sizes_ptr,
      int32_t problem_count) {
    int32_t total_tiles = 0;
    for (int32_t i = 0; i < problem_count; ++i) {
      auto problem = host_problem_sizes_ptr[i];
      possibly_transpose_problem(problem);
      auto grid = grid_shape(problem);
      total_tiles += tile_count(grid);
    }

    return total_tiles;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          GroupScheduleMode GroupScheduleMode_,
          int PrefetchTileCount,
          int ThreadCount>
struct MoeProblemVisitor;

/////////////////////////////////////////////////////////////////////////////////////////////////
// ProblemVisitor that performs all scheduling on device
//
template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount>
struct MoeProblemVisitor<ProblemSizeHelper,
                         ThreadblockShape,
                         GroupScheduleMode::kDeviceOnly,
                         PrefetchTileCount,
                         ThreadCount>
    : public BaseMoeProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
  using Base = BaseMoeProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
  using Params = typename Base::Params;
  static int const kThreadCount = ThreadCount;
  static bool const kRequiresPrecomputation = false;
  static int const kThreadsPerWarp = 32;

  struct SharedStorage {};

  // Final tile of the problem loaded by this thread. Each thread will hold
  // a separate value.
  int32_t problem_ending_tile;

  SharedStorage &shared_storage;

  //
  // Methods
  //
  CUTLASS_DEVICE
  MoeProblemVisitor(Params const &params_,
                    SharedStorage &shared_storage_,  // NOLINT
                    int32_t block_idx)
      : Base(params_, block_idx),
        problem_ending_tile(0),
        shared_storage(shared_storage_) {
    this->problem_idx = -1 * kThreadsPerWarp;
    this->problem_tile_start = 0;
  }

  CUTLASS_DEVICE
  bool next_tile() {
    // Check whether the tile to compute is within the range of the current
    // problem.
    int32_t problem_tile_end = __shfl_sync(
        0xffffffff, problem_ending_tile, this->problem_idx % kThreadsPerWarp);
    if (this->tile_idx < problem_tile_end) {
      return true;
    }

    // Check whether the tile to compute is within the current group of problems
    // fetched by the warp. The last tile for this group is the final tile of
    // the problem held by the final thread in the warp.
    int32_t group_tile_end =
        __shfl_sync(0xffffffff, problem_ending_tile, kThreadsPerWarp - 1);

    // Keep the starting problem for this group in `problem_idx`. This is done
    // to reduce register pressure. The starting problem for this group is
    // simply the first problem in the group most recently fetched by the warp.
    int32_t &group_problem_start = this->problem_idx;
    group_problem_start =
        (this->problem_idx / kThreadsPerWarp) * kThreadsPerWarp;

    // Keep the starting tile for this group in `problem_tile_start`. This is
    // done to reduce register pressure.
    int32_t &group_tile_start = this->problem_tile_start;

    // Each thread in the warp processes a separate problem to advance until
    // reaching a problem whose starting tile is less less than tile_idx.
    while (group_tile_end <= this->tile_idx) {
      group_problem_start += kThreadsPerWarp;
      if (group_problem_start > this->params.problem_count) {
        return false;
      }

      // Since `group_tile_start` is a reference to `this->problem_tile_start`,
      // this also sets `this->problem_tile_start`. The fact that
      // `this->problem_tile_start` is also set here is used later in
      // `next_tile`.
      group_tile_start = group_tile_end;

      int lane_idx = threadIdx.x % kThreadsPerWarp;
      int32_t lane_problem = group_problem_start + lane_idx;

      // Compute the number of tiles in the problem assigned to each thread.
      problem_ending_tile = 0;
      if (lane_problem < this->params.problem_count) {
        cutlass::gemm::GemmCoord problem = this->problem_size(lane_problem);
        cutlass::gemm::GemmCoord grid = this->grid_shape(problem);
        problem_ending_tile = this->tile_count(grid);
      }

      // Compute a warp-wide inclusive prefix sum to compute the ending tile
      // index of each thread's problem.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kThreadsPerWarp; i <<= 1) {
        int32_t val = __shfl_up_sync(0xffffffff, problem_ending_tile, i);
        if (lane_idx >= i) {
          problem_ending_tile += val;
        }
      }

      // The total tile count for this group is now in the final position of the
      // prefix sum
      int32_t tiles_in_group =
          __shfl_sync(0xffffffff, problem_ending_tile, kThreadsPerWarp - 1);

      problem_ending_tile += group_tile_start;
      group_tile_end += tiles_in_group;
    }

    // The next problem to process is the first one that does not have ending
    // tile position that is greater than or equal to tile index.
    int32_t problem_idx_in_group = __popc(
        __ballot_sync(0xffffffff, problem_ending_tile <= this->tile_idx));

    this->problem_idx = group_problem_start + problem_idx_in_group;

    // The starting tile for this problem is the ending tile of the previous
    // problem. In cases where `problem_idx_in_group` is the first problem in
    // the group, we do not need to reset `problem_tile_start`, because it is
    // set to the previous group's ending tile in the while loop above.
    if (problem_idx_in_group > 0) {
      this->problem_tile_start = __shfl_sync(
          0xffffffff, problem_ending_tile, problem_idx_in_group - 1);
    }

    return true;
  }

  static size_t get_workspace_size(
      const cutlass::gemm::GemmCoord *host_problem_sizes_ptr,
      int32_t problem_count,
      int32_t block_count) {
    return 0;
  }

  static void host_precompute(
      const cutlass::gemm::GemmCoord *host_problem_sizes_ptr,
      int32_t problem_count,
      int32_t block_count,
      void *host_workspace_ptr) {}
};

/// Visitor class to abstract away the algorithm for iterating over tiles
template <typename ThreadblockShape,
          GroupScheduleMode GroupScheduleMode_,
          int PrefetchTileCount,
          int ThreadCount,
          bool Transposed = false>
struct GemmMoeProblemVisitor
    : public MoeProblemVisitor<
          detail::GemmGroupedProblemSizeHelper<ThreadblockShape, Transposed>,
          ThreadblockShape,
          GroupScheduleMode_,
          PrefetchTileCount,
          ThreadCount> {
  static bool const kTransposed = Transposed;

  using ProblemSizeHelper =
      detail::GemmGroupedProblemSizeHelper<ThreadblockShape, Transposed>;
  using Base = MoeProblemVisitor<ProblemSizeHelper,
                                 ThreadblockShape,
                                 GroupScheduleMode_,
                                 PrefetchTileCount,
                                 ThreadCount>;
  using Params = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GemmMoeProblemVisitor(Params const &params_,
                        SharedStorage &shared_storage_,  // NOLINT
                        int32_t block_idx)
      : Base(params_, shared_storage_, block_idx) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// This section exists to that we can use the same kernel code for regular gemm
// and dequantizing gemms. It will dispatch to the dequantizing gemm if the Mma
// type has an Iterator for scales in global.
template <typename...>
using void_t = void;

template <typename Mma, typename = void>
struct use_dq_gemm : platform::false_type {};

template <typename Mma>
struct use_dq_gemm<Mma, void_t<typename Mma::IteratorScale>>
    : platform::true_type {};

// SFINAE overload for dequantizing gemm
template <
    typename Mma,
    typename ElementScale,
    typename platform::enable_if<use_dq_gemm<Mma>::value, bool>::type = true>
CUTLASS_DEVICE static void run_mma(Mma mma,
                                   int gemm_k_iterations,
                                   typename Mma::FragmentC &accum,  // NOLINT
                                   typename Mma::IteratorA iterator_A,
                                   typename Mma::IteratorB iterator_B,
                                   typename Mma::FragmentC const &src_accum,
                                   ElementScale *weight_scale_ptr,
                                   MatrixCoord scale_extent,
                                   const int thread_idx,
                                   MatrixCoord tb_offset_scale) {
  typename Mma::IteratorScale iterator_scale(
      Mma::IteratorScale::Layout(scale_extent.column()),
      weight_scale_ptr,
      scale_extent,
      thread_idx,
      tb_offset_scale);

  mma(gemm_k_iterations,
      accum,
      iterator_A,
      iterator_B,
      iterator_scale,
      src_accum);
}

// SFINAE overload for normal gemm. This completely ignores the scale parameters
template <
    typename Mma,
    typename ElementScale,
    typename platform::enable_if<!use_dq_gemm<Mma>::value, bool>::type = true>
CUTLASS_DEVICE static void run_mma(Mma mma,
                                   int gemm_k_iterations,
                                   typename Mma::FragmentC &accum,  // NOLINT
                                   typename Mma::IteratorA iterator_A,
                                   typename Mma::IteratorB iterator_B,
                                   typename Mma::FragmentC const &src_accum,
                                   ElementScale *weight_scale_ptr,
                                   MatrixCoord scale_extent,
                                   const int thread_idx,
                                   MatrixCoord tb_offset_scale) {
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, src_accum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          GroupScheduleMode GroupScheduleMode_  ///! Type of scheduling to
                                                /// perform
          >
struct MoeFCGemm {
 public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed = false;

  // Optional transpose
  using MapArguments =
      kernel::detail::MapArguments<typename Mma::IteratorA::Element,
                                   typename Mma::IteratorA::Layout,
                                   Mma::kTransformA,
                                   Mma::IteratorA::AccessType::kElements,
                                   typename Mma::IteratorB::Element,
                                   typename Mma::IteratorB::Layout,
                                   Mma::kTransformB,
                                   Mma::IteratorB::AccessType::kElements,
                                   typename Mma::LayoutC,
                                   kTransposed>;

  // Public-facing type definitions related to operand element type, layout, and
  // complex conjugate operation. Must interact with the 'kTransposed' notion.
  static_assert(!kTransposed, "Transpose problem not supported");
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;
  using ElementScale = ElementC;

  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC =
      Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor = GemmMoeProblemVisitor<ThreadblockShape,
                                               kGroupScheduleMode,
                                               kThreadCount,
                                               kThreadCount,
                                               kTransposed>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //

    int problem_count;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    ElementA *ptr_A;
    ElementB *ptr_B;
    ElementScale *weight_scales;
    ElementC *ptr_C;
    ElementC *ptr_D;

    int64_t *total_rows_before_expert;
    int64_t gemm_n;
    int64_t gemm_k;

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
          ptr_A(nullptr),
          ptr_B(nullptr),
          weight_scales(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          total_rows_before_expert(nullptr),
          gemm_n(0),
          gemm_k(0),
          host_problem_sizes(nullptr) {}

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(int problem_count,
              int threadblock_count,
              typename EpilogueOutputOp::Params output_op,
              const ElementA *ptr_A,
              const ElementB *ptr_B,
              const ElementScale *weight_scales,
              const ElementC *ptr_C,
              ElementC *ptr_D,
              int64_t *total_rows_before_expert,
              int64_t gemm_n,
              int64_t gemm_k,
              GemmCoord *host_problem_sizes = nullptr)
        : problem_count(problem_count),
          threadblock_count(threadblock_count),
          output_op(output_op),
          ptr_A(const_cast<ElementA *>(ptr_A)),
          ptr_B(const_cast<ElementB *>(ptr_B)),
          weight_scales(const_cast<ElementScale *>(weight_scales)),
          ptr_C(const_cast<ElementC *>(ptr_C)),
          ptr_D(ptr_D),
          total_rows_before_expert(total_rows_before_expert),
          gemm_n(gemm_n),
          gemm_k(gemm_k),
          host_problem_sizes(nullptr) {
      if (platform::is_same<uint8_t, ElementB>::value ||
          platform::is_same<uint4b_t, ElementB>::value) {
        assert(weight_scales);
      }
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {
    typename ProblemVisitor::Params problem_visitor;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    ElementA *ptr_A;
    ElementB *ptr_B;
    ElementScale *weight_scales;
    ElementC *ptr_C;
    ElementC *ptr_D;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : ptr_A(nullptr),
          ptr_B(nullptr),
          weight_scales(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args,
           void *workspace = nullptr,
           int tile_count = 0)  // NOLINT
        : problem_visitor(args.total_rows_before_expert,
                          args.gemm_n,
                          args.gemm_k,
                          args.problem_count,
                          workspace,
                          tile_count),
          threadblock_count(args.threadblock_count),
          output_op(args.output_op),
          ptr_A(args.ptr_A),
          ptr_B(args.ptr_B),
          weight_scales(args.weight_scales),
          ptr_C(args.ptr_C),
          ptr_D(args.ptr_D) {}

    CUTLASS_HOST_DEVICE
    void update(Arguments const &args,
                void *workspace = nullptr,
                int tile_count = 0) {
      problem_visitor =
          typename ProblemVisitor::Params(args.total_rows_before_expert,
                                          args.gemm_n,
                                          args.gemm_k,
                                          args.problem_count,
                                          workspace,
                                          tile_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B = args.ptr_B;
      weight_scales = args.weight_scales;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename ProblemVisitor::SharedStorage problem_visitor;
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  MoeFCGemm() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const &problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    if (platform::is_same<uint8_t, ElementB>::value ||
        platform::is_same<uint4b_t, ElementB>::value) {
      if (args.weight_scales == nullptr) {
        CUTLASS_TRACE_HOST(
            "MoeFCGemm::can_implement() - weight scales are required for "
            "uint8_t and uint4b_t");
        return Status::kInvalid;
      }
    } else if (args.weight_scales != nullptr) {
      CUTLASS_TRACE_HOST(
          "MoeFCGemm::can_implement() - weight scales are ignored for all "
          "types except uint8_t and uint4b_t");
      return Status::kInvalid;
    }
    return Status::kSuccess;
  }

  static size_t get_extra_workspace_size(
      Arguments const &args, cutlass::gemm::GemmCoord const &grid_tiled_shape) {
    return 0;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params,
                  SharedStorage &shared_storage) {  // NOLINT
    //
    // These types shadow the type-level definitions and support the ability to
    // implement a 'transposed' GEMM that computes the transposed problems.
    //
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;
    static constexpr int kInterleave =
        Mma::IteratorB::Shape::kRow / Mma::SmemIteratorB::Shape::kRow;
    static_assert(platform::is_same<LayoutB, layout::RowMajor>::value &&
                          kInterleave == 1 ||
                      platform::is_same<LayoutB, layout::ColumnMajor>::value &&
                          kInterleave >= 1,
                  "B must be row major/col major OR col major interleaved.");

    //
    // Problem visitor.
    //
    ProblemVisitor problem_visitor(
        params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

    const int64_t gemm_k = params.problem_visitor.gemm_k;
    const int64_t gemm_n = params.problem_visitor.gemm_n;
    int64_t bytes_per_expert_matrix =
        (gemm_k * gemm_n / 8) * cutlass::sizeof_bits<ElementB>::value;

    // Outer 'persistent' loop to iterate over tiles
    while (problem_visitor.next_tile()) {
      GemmCoord problem_size = problem_visitor.problem_size();
      int32_t problem_idx = problem_visitor.problem_index();
      int32_t cta_idx = int32_t(problem_visitor.threadblock_idx());

      GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

      cutlass::gemm::GemmCoord threadblock_offset(
          int(cta_idx / grid_shape.n()) * Mma::Shape::kM,  // NOLINT
          int(cta_idx % grid_shape.n()) * Mma::Shape::kN,  // NOLINT
          0);

      // Load element pointers. Exchange pointers and strides if working on the
      // transpose
      const int64_t rows_to_jump =
          problem_idx == 0
              ? 0
              : params.problem_visitor.last_row_for_problem[problem_idx - 1];
      ElementA *ptr_A =
          reinterpret_cast<ElementA *>(params.ptr_A) + rows_to_jump * gemm_k;
      typename LayoutA::LongIndex ldm_A = gemm_k;

      char *byte_ptr_B = ((char *)params.ptr_B) +  // NOLINT
                         problem_idx * bytes_per_expert_matrix;
      ElementB *ptr_B = reinterpret_cast<ElementB *>(byte_ptr_B);
      typename LayoutB::LongIndex ldm_B =
          platform::is_same<layout::RowMajor, LayoutB>::value
              ? gemm_n
              : gemm_k * kInterleave;

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
          threadblock_offset.m(),
          0,
      };

      cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n() / kInterleave};

      cutlass::MatrixCoord tb_offset_scale{0, threadblock_offset.n()};

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(LayoutA(ldm_A),
                                         ptr_A,
                                         {problem_size.m(), problem_size.k()},
                                         thread_idx,
                                         tb_offset_A);

      typename Mma::IteratorB iterator_B(
          LayoutB(ldm_B),
          ptr_B,
          {problem_size.k() * kInterleave, problem_size.n() / kInterleave},
          thread_idx,
          tb_offset_B);

      typename Mma::FragmentC accumulators;

      accumulators.clear();

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

      int lane_idx = threadIdx.x % 32;

      //
      // Matrix multiply phase
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Wait for all threads to finish their epilogue phases from the previous
      // tile.
      __syncthreads();

      // Compute threadblock-scoped matrix multiply-add
      ElementScale *weight_scale_ptr =
          params.weight_scales + problem_idx * problem_size.n();
      run_mma<Mma>(mma,
                   gemm_k_iterations,
                   accumulators,
                   iterator_A,
                   iterator_B,
                   accumulators,
                   weight_scale_ptr,
                   {1, problem_size.n()},
                   thread_idx,
                   tb_offset_scale);

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      ElementC *ptr_C =
          reinterpret_cast<ElementC *>(params.ptr_C) + problem_idx * gemm_n;
      ElementC *ptr_D =
          reinterpret_cast<ElementC *>(params.ptr_D) + rows_to_jump * gemm_n;

      LayoutC layout_C(0);
      LayoutC layout_D(gemm_n);

      typename Epilogue::OutputTileIterator::Params params_C(layout_C);
      typename Epilogue::OutputTileIterator::Params params_D(layout_D);

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(params_C,
                                                       ptr_C,
                                                       problem_size.mn(),
                                                       thread_idx,
                                                       threadblock_offset.mn());

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(params_D,
                                                       ptr_D,
                                                       problem_size.mn(),
                                                       thread_idx,
                                                       threadblock_offset.mn());

      Epilogue epilogue(
          shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      epilogue(output_op, iterator_D, accumulators, iterator_C);

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
