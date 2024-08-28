/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or
   support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          typename KernelArch>
struct GemmFpAIntBSplitK {
 public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Element;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Mma::LayoutC;
  using ElementScale = ElementC;
  using AccumulatorTile = typename Mma::FragmentC;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformA;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC =
      Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;
  static int const kSplitKAlignment = const_max(
      128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  static constexpr int kInterleave =
      Mma::IteratorB::Shape::kRow / Mma::Shape::kK;

  /// Workspace bytes per thread block
  static size_t const kWorkspaceBytesPerBlock =
      __NV_STD_MAX(kThreadCount * sizeof(AccumulatorTile),
                   Epilogue::kWorkspaceBytesPerBlock);

  /// Block-striped reduction utility
  using BlockStripedReduceT = BlockStripedReduce<kThreadCount, AccumulatorTile>;

  /// Parameters structure
  struct Arguments {
    GemmUniversalMode mode;

    cutlass::gemm::GemmCoord problem_size;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Mma::IteratorScale::TensorRef ref_scale;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;

    // Control serial split-k
    int batch_count;

    typename EpilogueOutputOp::Params output_op;

    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    // Included so we can use Gemm Universal
    int batch_stride_D = 0;

    int avail_sms;
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Arguments() {}

    CUTLASS_HOST_DEVICE
    Arguments(GemmUniversalMode mode,
              cutlass::gemm::GemmCoord const &problem_size,
              typename Mma::IteratorA::TensorRef ref_A,
              typename Mma::IteratorB::TensorRef ref_B,
              typename Mma::IteratorScale::TensorRef ref_scale,
              typename Epilogue::OutputTileIterator::TensorRef ref_C,
              typename Epilogue::OutputTileIterator::TensorRef ref_D,
              int split_k_factor,
              typename EpilogueOutputOp::Params output_op =
                  typename EpilogueOutputOp::Params(),
              int const *gather_A_indices = nullptr,
              int const *gather_B_indices = nullptr,
              int const *scatter_D_indices = nullptr,
              int avail_sms = -1)
        : mode(mode),
          problem_size(problem_size),
          batch_count(split_k_factor),
          ref_A(ref_A),
          ref_B(ref_B),
          ref_scale(ref_scale),
          ref_C(ref_C),
          ref_D(ref_D),
          output_op(output_op),
          gather_A_indices(gather_A_indices),
          gather_B_indices(gather_B_indices),
          scatter_D_indices(scatter_D_indices),
          avail_sms(avail_sms) {}
  };

  /// Parameters structure
  struct Params {
   public:
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorScale::Params params_scale;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Mma::IteratorScale::TensorRef ref_scale;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename EpilogueOutputOp::Params output_op;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;
    int64_t batch_stride_A;
    int64_t batch_stride_B;
    GemmUniversalMode mode;

    ThreadblockSwizzle block_mapping;
    void *barrier_workspace;
    void *partials_workspace;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

   protected:
    //
    // Host-only dispatch-utilities
    //

    /// Pad the given allocation size up to the nearest cache line
    static size_t cacheline_align_up(size_t size) {
      static const int CACHELINE_SIZE = 128;
      return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
    }

    /// Get the workspace size needed for barrier
    size_t get_barrier_workspace_size() const {
      // For atomic reduction, each SK-block needs a synchronization flag.  For
      // parallel reduction, each reduction block needs its own synchronization
      // flag.
      int sk_blocks =
          block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
      int num_flags = fast_max(sk_blocks, block_mapping.reduction_blocks);

      return cacheline_align_up(sizeof(typename Barrier::T) * num_flags);
    }

    /// Get the workspace size needed for intermediate partial sums
    size_t get_partials_workspace_size() const {
      int sk_blocks =
          block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
      return cacheline_align_up(kWorkspaceBytesPerBlock * sk_blocks);
    }

    //
    // Methods
    //

   public:
    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args, int device_sms, int sm_occupancy)
        : params_A(args.ref_A.layout()),
          ref_A(args.ref_A),
          params_B(args.ref_B.layout()),
          ref_B(args.ref_B),
          params_scale(args.ref_scale.layout()),
          ref_scale(args.ref_scale),
          params_C(args.ref_C.layout()),
          ref_C(args.ref_C),
          params_D(args.ref_D.layout()),
          ref_D(args.ref_D),
          output_op(args.output_op),
          gather_A_indices(args.gather_A_indices),
          gather_B_indices(args.gather_B_indices),
          scatter_D_indices(args.scatter_D_indices),
          batch_stride_A(args.ref_A.stride()[0]),
          batch_stride_B(args.ref_B.stride()[0]),
          batch_stride_C(args.ref_C.stride()[0]),
          batch_stride_D(args.batch_stride_D),
          barrier_workspace(nullptr),
          partials_workspace(nullptr) {
      // Number of SMs to make available for StreamK decomposition
      int avail_sms = (args.avail_sms == -1)
                          ? device_sms
                          : fast_min(args.avail_sms, device_sms);

      // Initialize the block mapping structure
      block_mapping = ThreadblockSwizzle(
          typename ThreadblockSwizzle::template KernelTraits<
              GemmFpAIntBSplitK>(),
          args.mode,
          args.problem_size,
          {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
          args.batch_count,
          sm_occupancy,
          device_sms,
          avail_sms);
    }

    /// Returns the workspace size (in bytes) needed for these parameters
    size_t get_workspace_size() const {
      return get_barrier_workspace_size() + get_partials_workspace_size();
    }

    /// Assign and initialize the specified workspace buffer.  Assumes
    /// the memory allocated to workspace is at least as large as
    /// get_workspace_size().
    Status init_workspace(void *workspace, cudaStream_t stream = nullptr) {
      uint8_t *ptr = static_cast<uint8_t *>(workspace);

      // Establish partials workspace
      partials_workspace = nullptr;
      size_t partials_workspace_bytes = get_partials_workspace_size();
      if (partials_workspace_bytes > 0) {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }
        partials_workspace = ptr;
        ptr += partials_workspace_bytes;
      }

      // Establish barrier workspace
      barrier_workspace = nullptr;
      size_t barrier_workspace_bytes = get_barrier_workspace_size();
      if (barrier_workspace_bytes > 0) {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }
        barrier_workspace = ptr;
        ptr += barrier_workspace_bytes;
      }

      // Zero-initialize barrier workspace
      if (barrier_workspace) {
        size_t barrier_workspace_bytes = get_barrier_workspace_size();

        CUTLASS_TRACE_HOST("  Initialize " << barrier_workspace_bytes
                                           << " barrier bytes");

        cudaError_t result = cudaMemsetAsync(
            barrier_workspace, 0, barrier_workspace_bytes, stream);

        if (result != cudaSuccess) {
          CUTLASS_TRACE_HOST("  cudaMemsetAsync() returned error "
                             << cudaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }

      return Status::kSuccess;
    }

    /// Returns the GEMM volume in thread block tiles
    cutlass::gemm::GemmCoord get_tiled_shape() const {
      return block_mapping.tiled_shape();
    }

    /// Returns the total number of thread blocks to launch
    int get_grid_blocks() const {
      dim3 grid_dims = get_grid_dims();
      return grid_dims.x * grid_dims.y * grid_dims.z;
    }

    /// Returns the grid extents in thread blocks to launch
    dim3 get_grid_dims() const { return block_mapping.get_grid_dims(); }
  };

  /// Tile work descriptor
  struct TileWorkDesc {
    /// The linear tile index
    int tile_idx;

    /// The location of this tile (in threadblock-tile coordinates) in the
    /// output matrix
    cutlass::gemm::GemmCoord tiled_coord;

    // The first global-scoped MAC-iteration this threadblock will perform for
    // this tile
    int iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock
    // will perform for this tile
    int k_begin;

    // The ending index (one-past) in the k-domain for MAC-iterations this
    // threadblock will perform for this tile
    int k_end;

    /// The number of remaining MAC-iterations this threadblock will perform for
    /// this tile
    int k_iters_remaining;

    // Whether this block will perform the first iteration of this tile
    CUTLASS_DEVICE
    bool tile_started() { return (k_begin == 0); }

    // Whether this block will perform the last iteration of this tile
    CUTLASS_DEVICE
    bool tile_finished(Params const &params) {
      return (k_end == params.block_mapping.problem_size.k());
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

 protected:
  /// GEMM problem parameters
  Params const &params;

  /// Shared storage reference
  SharedStorage &shared_storage;

  /// ID within the threadblock
  int thread_idx;

  /// ID of warp
  int warp_idx;

  /// ID of each thread within a warp
  int lane_idx;

  /// Threadblock scoped epilogue
  Epilogue epilogue;
  //
  // Methods
  //
 public:
  CUTLASS_DEVICE
  GemmFpAIntBSplitK(Params const &params,
                    SharedStorage &shared_storage)  // NOLINT
      : params(params),
        shared_storage(shared_storage),
        thread_idx(threadIdx.x),
        warp_idx(__shfl_sync(0xffffffff,
                             threadIdx.x / 32,
                             0)),  // broadcast the warp_id computed by lane 0
                                   // to ensure dependent code
        lane_idx(threadIdx.x % 32),
        epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx) {}

  /// Determines whether the GEMM problem size satisfies this kernel's
  /// alignment requirements
  static Status can_implement(
      cutlass::gemm::GemmCoord const &problem_size) {  // NOLINT
    CUTLASS_TRACE_HOST("GemmUniversalStreamk::can_implement()");

    static int const kAlignmentA =
        (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value)
            ? 32
        : (platform::is_same<LayoutA,
                             layout::ColumnMajorInterleaved<64>>::value)
            ? 64
            : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =
        (platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value)
            ? 32
        : (platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value)
            ? 64
            : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC =
        (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value)
            ? 32
        : (platform::is_same<LayoutC,
                             layout::ColumnMajorInterleaved<64>>::value)
            ? 64
            : Epilogue::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (platform::is_same<LayoutA,
                                 layout::ColumnMajorInterleaved<32>>::value ||
               platform::is_same<LayoutA,
                                 layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (platform::is_same<LayoutB,
                                 layout::RowMajorInterleaved<32>>::value ||
               platform::is_same<LayoutB,
                                 layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (platform::is_same<LayoutC,
                                 layout::ColumnMajorInterleaved<32>>::value ||
               platform::is_same<LayoutC,
                                 layout::ColumnMajorInterleaved<64>>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    }

    if (isAMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for A operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isBMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for B operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isCMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for C operand");
      return Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

    return Status::kSuccess;
  }

  /// Determines whether the GEMM problem satisfies this kernel's
  /// alignment requirements
  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

 protected:
  //
  // Device-only utility methods
  //

  /// Iterator for fetching tile fragments from A
  CUTLASS_DEVICE
  typename Mma::IteratorA init_iterator_A(TileWorkDesc &tile_work,  // NOLINT
                                          GemmUniversalMode mode) {
    // The input A matrix
    ElementA *ptr_A = static_cast<ElementA *>(params.ref_A.data());

    // Update input pointers based on batched/array mode
    if (mode == GemmUniversalMode::kBatched) {
      ptr_A += tile_work.tiled_coord.k() * params.batch_stride_A;
    }
    // if (mode == GemmUniversalMode::kArray) {
    //   ptr_A = static_cast<ElementA * const
    //   *>(params.ref_A.data())[tile_work.tiled_coord.k()];
    // }

    int m_begin = tile_work.tiled_coord.m() * Mma::Shape::kM;
    int m_end = params.block_mapping.problem_size.m();
    return Mma::IteratorA(params.params_A,
                          ptr_A,
                          {m_end, tile_work.k_end},
                          threadIdx.x,
                          {m_begin, tile_work.k_begin});
  }

  /// Iterator for fetching tile fragments from B
  CUTLASS_DEVICE
  typename Mma::IteratorB init_iterator_B(TileWorkDesc &tile_work,  // NOLINT
                                          GemmUniversalMode mode) {
    // The input B matrix
    ElementB *ptr_B = static_cast<ElementB *>(params.ref_B.data());

    // Update input pointers based on batched/array mode
    if (mode == GemmUniversalMode::kBatched) {
      ptr_B += tile_work.tiled_coord.k() * kInterleave * params.batch_stride_B;
    }
    // if (mode == GemmUniversalMode::kArray) {
    //   ptr_B = static_cast<ElementB * const
    //   *>(params.ref_B.data())[tile_work.tiled_coord.k()];
    // }

    int n_begin = tile_work.tiled_coord.n() * Mma::Shape::kN;
    // int n_begin = 0 * Mma::Shape::kN;
    int n_end = params.block_mapping.problem_size.n();
    return Mma::IteratorB(
        params.params_B,
        ptr_B,
        {tile_work.k_end * kInterleave, n_end / kInterleave},
        threadIdx.x,
        {tile_work.k_begin * kInterleave, n_begin / kInterleave});
  }

  /// Iterator for fetching tile fragments from B
  CUTLASS_DEVICE
  typename Mma::IteratorScale init_iterator_scale(
      TileWorkDesc &tile_work,  // NOLINT
      GemmUniversalMode mode) {
    // The input B matrix
    ElementScale *ptr_scale =
        static_cast<ElementScale *>(params.ref_scale.data());

    // if (mode == GemmUniversalMode::kArray) {
    //   ptr_B = static_cast<ElementB * const
    //   *>(params.ref_B.data())[tile_work.tiled_coord.k()];
    // }

    // int n_begin = tile_work.tiled_coord.n() * Mma::Shape::kN;
    int n_begin = tile_work.tiled_coord.n() * Mma::Shape::kN;
    int n_end = params.block_mapping.problem_size.n();
    return Mma::IteratorScale(params.params_scale,
                              ptr_scale,
                              {tile_work.k_end, n_end},
                              threadIdx.x,
                              {tile_work.k_begin, n_begin});
  }

  CUTLASS_DEVICE
  void init_dp_tile_work(TileWorkDesc &tile_work, int tile_idx) {  // NOLINT
    // The linear tile index
    tile_work.tile_idx = tile_idx;

    // The first global-scoped MAC-iteration this threadblock will perform for
    // this tile
    tile_work.iter_begin = tile_idx * params.block_mapping.iters_per_tile();

    // The number of MAC-iterations this threadblock will perform for this tile
    tile_work.k_iters_remaining = params.block_mapping.iters_per_tile();

    // The starting index in the k-domain for MAC-iterations this threadblock
    // will perform for this tile
    tile_work.k_begin = 0;

    // The ending index (one-past) in the k-domain for MAC-iterations this
    // threadblock will perform for this tile
    tile_work.k_end = params.block_mapping.problem_size.k();

    // The location of this tile (in threadblock-tile coordinates) in the output
    // matrix
    tile_work.tiled_coord =
        params.block_mapping.get_tile_offset(tile_work.tile_idx);
  }

  CUTLASS_DEVICE
  void init_sk_tile_work(TileWorkDesc &tile_work,  // NOLINT
                         int tile_idx,
                         int block_iter_begin,
                         int block_iter_end) {
    // The linear tile index
    tile_work.tile_idx = tile_idx;

    // The first global-scoped MAC-iteration for this tile
    int tile_iter_begin = tile_idx * params.block_mapping.iters_per_tile();

    // The first global-scoped MAC-iteration this threadblock will perform for
    // this tile
    tile_work.iter_begin = max(block_iter_begin, tile_iter_begin);

    // The first tile-scoped MAC-iteration this threadblock will perform for
    // this tile
    int k_iter_begin = tile_work.iter_begin - tile_iter_begin;

    // The last (one past) tile-scoped MAC-iteration this threadblock will
    // perform for this tile
    int k_iter_end = block_iter_end - tile_iter_begin;

    // The number of MAC-iterations this threadblock will perform for this tile
    tile_work.k_iters_remaining = k_iter_end - k_iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock
    // will perform for this tile
    tile_work.k_begin = k_iter_begin * Mma::Shape::kK;

    // The ending index (one-past) in the k-domain for MAC-iterations this
    // threadblock will perform for this tile
    tile_work.k_end =
        min(params.block_mapping.problem_size.k(),  // extent of k domain
            (k_iter_end * Mma::Shape::kK));  // extent of the threadblock's
                                             // global iteration assignment

    // The location of this tile (in threadblock-tile coordinates) in the output
    // matrix
    tile_work.tiled_coord =
        params.block_mapping.get_tile_offset(tile_work.tile_idx);
  }

  /// Share accumulators with peers
  CUTLASS_DEVICE
  void share_accumulators(AccumulatorTile const &accumulator_tile,
                          int block_idx,
                          int first_block_idx) {
    AccumulatorTile *accum_tile_workspace =
        reinterpret_cast<AccumulatorTile *>(params.partials_workspace);

    int accum_tile_offset = first_block_idx * kThreadCount;

    if (block_idx == first_block_idx) {
      // First peer initializes the workspace partials
      BlockStripedReduceT::store(accum_tile_workspace + accum_tile_offset,
                                 accumulator_tile,
                                 thread_idx);
    } else {
      // Subsequent peers atomically accumulate into the workspace partials
      if (ThreadblockSwizzle::kReductionStrategy ==
          ThreadblockSwizzle::kAtomic) {
        // Non-deterministic reduction order: wait for the first peer to have
        // initialized the partials before we add to them
        Barrier::wait_lt(
            params.barrier_workspace, thread_idx, first_block_idx, 1);
      } else {
        // Turnstile reduction order: wait until the previous peer has written
        int wait_count = block_idx - first_block_idx;
        Barrier::wait_eq(
            params.barrier_workspace, thread_idx, first_block_idx, wait_count);
      }

      // Perform reduction in workspace
      BlockStripedReduceT::reduce(accum_tile_workspace + accum_tile_offset,
                                  accumulator_tile,
                                  thread_idx);
    }

    // Signal our arrival
    Barrier::arrive_inc(params.barrier_workspace, thread_idx, first_block_idx);
  }

  /// Acquire accumulators from peers
  CUTLASS_DEVICE
  void acquire_accumulators(AccumulatorTile &accumulator_tile,  // NOLINT
                            int block_idx,
                            int first_block_idx) {
    AccumulatorTile *accum_tile_workspace =
        reinterpret_cast<AccumulatorTile *>(params.partials_workspace);

    // Wait for arrival
    int num_carry_in = block_idx - first_block_idx;
    Barrier::wait_eq_reset(
        params.barrier_workspace, thread_idx, first_block_idx, num_carry_in);

    // Load and add peer-partials accumulator tile to local accumulator tile
    int accum_tile_offset = first_block_idx * kThreadCount;
    BlockStripedReduceT::load_add(
        accumulator_tile, accum_tile_workspace + accum_tile_offset, thread_idx);
  }

  /// Perform epilogue computations and output
  CUTLASS_DEVICE
  void do_epilogue(TileWorkDesc &tile_work,              // NOLINT
                   AccumulatorTile &accumulator_tile) {  // NOLINT
    ElementC *ptr_C = static_cast<ElementC *>(params.ref_C.data());
    ElementC *ptr_D = static_cast<ElementC *>(params.ref_D.data());

    // Update pointers for batched/array mode(s)
    if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C += tile_work.tiled_coord.k() * params.batch_stride_C;
      ptr_D += tile_work.tiled_coord.k() * params.batch_stride_D;
    }
    // if (params.mode == GemmUniversalMode::kArray) {
    //   ptr_C = static_cast<ElementC * const
    //   *>(params.ref_C.data())[tile_work.tiled_coord.k()]; ptr_D =
    //   static_cast<ElementC * const
    //   *>(params.ref_D.data())[tile_work.tiled_coord.k()];
    // }

    // Location of this tile in item-coords
    MatrixCoord threadblock_item_begin(
        tile_work.tiled_coord.m() * Mma::Shape::kM,
        tile_work.tiled_coord.n() * Mma::Shape::kN);

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        ptr_C,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        ptr_D,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Execute the epilogue operator to update the destination tensor.
    epilogue.unified(EpilogueOutputOp(params.output_op),
                     iterator_D,
                     accumulator_tile,
                     iterator_C);
  }

  CUTLASS_DEVICE
  void separate_reduction(int reduce_idx) {
    int peer_idx_begin, peer_idx_last, reduce_tile_idx, reduce_fragment_idx;

    // Reduce by sk-tile (every tile contributed to by one or more blocks)
    reduce_tile_idx = reduce_idx / Epilogue::kAccumulatorFragments;
    reduce_fragment_idx = reduce_idx % Epilogue::kAccumulatorFragments;

    int iter_tile_first =
        reduce_tile_idx * params.block_mapping.iters_per_tile();
    int iter_tile_last =
        iter_tile_first + params.block_mapping.iters_per_tile() - 1;

    peer_idx_begin = params.block_mapping.get_sk_block_idx(iter_tile_first);
    peer_idx_last = params.block_mapping.get_sk_block_idx(iter_tile_last);

    // Wait for peers to complete
    int peer_idx_end = peer_idx_last + 1;
    int num_peers = peer_idx_end - peer_idx_begin;
    Barrier::wait_eq_reset(params.barrier_workspace,
                           thread_idx,
                           (reduce_tile_idx * Epilogue::kAccumulatorFragments) +
                               reduce_fragment_idx,
                           num_peers);

    /// The location of this tile (in threadblock-tile coordinates) in the
    /// output matrix
    GemmCoord tiled_coord =
        params.block_mapping.get_tile_offset(reduce_tile_idx);

    // Location of this tile in item-coords
    MatrixCoord threadblock_item_begin(tiled_coord.m() * Mma::Shape::kM,
                                       tiled_coord.n() * Mma::Shape::kN);

    ElementC *ptr_C = static_cast<ElementC *>(params.ref_C.data());
    ElementC *ptr_D = static_cast<ElementC *>(params.ref_D.data());

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        ptr_C,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        ptr_D,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Execute the epilogue operator to update the destination tensor.
    epilogue.reduce(peer_idx_begin,
                    peer_idx_end,
                    reduce_fragment_idx,
                    params.partials_workspace,
                    EpilogueOutputOp(params.output_op),
                    iterator_D,
                    iterator_C);
  }

  CUTLASS_DEVICE
  void process_tile(TileWorkDesc tile_work,
                    int block_idx,
                    int dp_start_block_idx,
                    int block_iter_begin) {
    // TODO(wangbojun) for debug

    // static_assert(print_type<Mma::Policy::Operator::FragmentC>());
    // Initialize input iterators
    typename Mma::IteratorA iterator_A =
        init_iterator_A(tile_work, params.mode);
    typename Mma::IteratorB iterator_B =
        init_iterator_B(tile_work, params.mode);
    typename Mma::IteratorScale iterator_scale =
        init_iterator_scale(tile_work, params.mode);
    // Initialize accumulators
    AccumulatorTile accumulator_tile;
    accumulator_tile.clear();
    // static_assert(print_type<Mma::>());

    // Perform this tile's range of multiply-accumulate (MAC) iterations
    Mma mma(shared_storage.main_loop, -1, thread_idx, warp_idx, lane_idx);

    mma(tile_work.k_iters_remaining,
        accumulator_tile,
        iterator_A,
        iterator_B,
        iterator_scale,
        accumulator_tile);

    if ((ThreadblockSwizzle::kReductionStrategy ==
         ThreadblockSwizzle::kAtomic) ||
        (params.block_mapping.reduction_blocks == 0) ||
        (block_idx >= dp_start_block_idx)) {
      //
      // Cooperative SK peer reduction or DP block
      //

      int first_block_idx = params.block_mapping.get_first_block_idx(
          tile_work.tile_idx, block_idx);

      if (!tile_work.tile_finished(params)) {
        // Non "finishing" SK blocks must share their partial accumulator sums
        // through global scratch workspace
        share_accumulators(accumulator_tile, block_idx, first_block_idx);
      } else {
        // DP blocks and "finishing" SK blocks must perform epilogue operations
        // and write the output tile
        if (!tile_work.tile_started()) {
          // A "finishing" SK block must first aggregate its accumulator partial
          // sums with those shared by peer threadblocks
          acquire_accumulators(accumulator_tile, block_idx, first_block_idx);
        }

        do_epilogue(tile_work, accumulator_tile);
      }
    } else {
      //
      // Separate peer reduction
      //

      // Share accumulator partial sums with peer threadblock(s) through scratch
      // workspace
      epilogue.share(block_idx,
                     params.partials_workspace,
                     accumulator_tile,
                     tile_work.tile_started());

      // Signal arrival
      Barrier::arrive_range_inc(
          params.barrier_workspace,
          thread_idx,
          tile_work.tile_idx * Epilogue::kAccumulatorFragments,
          Epilogue::kAccumulatorFragments);
    }
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void gemm() {
    // Initialize block's iteration range
    int tile_idx, block_iter_begin, block_iters_remaining;

    int sk_padding_start_block_idx =
        params.block_mapping.sk_regions() *
        params.block_mapping.sk_blocks_per_region();
    int dp_start_block_idx =
        params.block_mapping.sk_waves * params.block_mapping.avail_sms;
    int reduce_start_block_idx =
        dp_start_block_idx + params.block_mapping.dp_blocks;
    int grid_padding_start_block_idx =
        reduce_start_block_idx + params.block_mapping.reduction_blocks;

    int block_idx = params.block_mapping.get_block_idx();
    if (block_idx < sk_padding_start_block_idx) {
      // This is a SK block
      int block_iter_end;
      params.block_mapping.get_iter_extents(
          block_idx, block_iter_begin, block_iter_end);
      block_iters_remaining = block_iter_end - block_iter_begin;

      tile_idx = params.block_mapping.get_sk_tile_idx(block_iter_end - 1);
    } else if (block_idx < dp_start_block_idx) {
      // This is a filler block
      return;
    } else if (block_idx < reduce_start_block_idx) {
      // This is a DP block
      int dp_block_idx = block_idx - dp_start_block_idx;
      int first_dp_tile = (params.block_mapping.cohort_raster)
                              ? 0
                              : params.block_mapping.sk_tiles;

      // Blocks in first DP wave get configured number of tiles
      tile_idx = first_dp_tile + dp_block_idx;
      int tile_allottment = params.block_mapping.dp_first_wave_tiles;

      // Blocks in subsequent DP waves get 1 tile
      if (dp_block_idx >= params.block_mapping.avail_sms) {
        tile_allottment = 1;
        tile_idx += (params.block_mapping.dp_first_wave_tiles - 1) *
                    params.block_mapping.avail_sms;
      }

      block_iter_begin = 0;
      block_iters_remaining =
          params.block_mapping.iters_per_tile() * tile_allottment;
    } else if ((ThreadblockSwizzle::kReductionStrategy ==
                ThreadblockSwizzle::kMixed) &&
               (block_idx < grid_padding_start_block_idx)) {
      // This is a reduction threadblock
      int reduce_block_idx = block_idx - reduce_start_block_idx;
      separate_reduction(reduce_block_idx);
      return;
    } else {
      // This is a filler block
      return;
    }

    // Iteration-processing loop body
    CUTLASS_PRAGMA_NO_UNROLL
    while (true) {
      // Initialize tile work descriptor
      TileWorkDesc tile_work;
      if (block_idx >= dp_start_block_idx) {
        init_dp_tile_work(tile_work, tile_idx);

        // DP blocks exit if out of bounds or overlap an SK tile (only possible
        // during cohort rasterization, where dp_first_wave_tiles must be 1)
        if ((tile_idx < params.block_mapping.sk_tiles) ||
            (tile_work.tiled_coord.m() >=
             params.block_mapping.tiled_shape().m()) ||
            (tile_work.tiled_coord.n() >=
             params.block_mapping.tiled_shape().n())) {
          break;
        }
      } else {
        init_sk_tile_work(tile_work,
                          tile_idx,
                          block_iter_begin,
                          block_iter_begin + block_iters_remaining);
      }

      // Perform this block's share of work for this tile
      process_tile(tile_work, block_idx, dp_start_block_idx, block_iter_begin);

      // Update remaining work for this block
      block_iters_remaining -= tile_work.k_iters_remaining;
      if (block_iters_remaining == 0) {
        // Done
        break;
      }

      // Continue to next tile
      __syncthreads();

      if (block_idx >= dp_start_block_idx) {
        // DP block consume their tiles at stride
        tile_idx += params.block_mapping.avail_sms;
      } else {
        // SK blocks consume their tiles in backwards order
        tile_idx--;
      }
    }
  }

 public:
  CUTLASS_DEVICE
  static void invoke(Params const &params,
                     SharedStorage &shared_storage) {  // NOLINT
    GemmFpAIntBSplitK op(params, shared_storage);
    op();
  }

  /*
      To improve compilation speed, we do not compile the device operator if the
     CUDA_ARCH does not correspond to the ArchTag of the cutlass kernel
     operator.
    */
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
    gemm();
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && (__CUDA_ARCH__ < 800)
    gemm();
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 910)
    gemm();
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
