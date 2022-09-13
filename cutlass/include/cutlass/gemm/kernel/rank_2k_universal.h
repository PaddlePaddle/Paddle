/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief 

*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma1_,                 ///! Threadblock-scoped matrix multiply-accumulate (A*B^T)
  typename Mma2_,                 ///! Threadblock-scoped matrix multiply-accumulate (B*A^T)
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  FillMode FillModeC_,            ///! Fill Mode for C (kLower or kUpper)
  BlasMode BlasMode_              ///! Blas3 computation mode
>
struct Rank2KUniversal {
public:

  using Mma1 = Mma1_;
  using Mma2 = Mma2_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma1::IteratorA::Element;
  using ElementB = typename Mma1::IteratorB::Element;

  // Mma1 (A x B^T)
  using LayoutA = typename Mma1::IteratorA::Layout;
  using LayoutBT = typename Mma1::IteratorB::Layout;
  static ComplexTransform const kMma1TransformA = Mma1::kTransformA;
  static ComplexTransform const kMma1TransformB = Mma1::kTransformB;

  // Mma2 (B x A^T)
  using LayoutB = typename Mma2::IteratorA::Layout;
  using LayoutAT = typename Mma2::IteratorB::Layout;
  static ComplexTransform const kMma2TransformA = Mma2::kTransformA;
  static ComplexTransform const kMma2TransformB = Mma2::kTransformB;

  // Common type definitions for Mma1 and Mma2
  using Operator = typename Mma1::Operator;
  using OperatorClass = typename Mma1::Operator::OperatorClass;
  using ThreadblockShape = typename Mma1::Shape;
  using WarpShape = typename Mma1::Operator::Shape;
  using InstructionShape = typename Mma1::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma1::ArchTag;

  static int const kStages = Mma1::kStages;
  static int const kAlignmentA = Mma1::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma1::IteratorB::AccessType::kElements;

  // Output related typedefinitions
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Epilogue::OutputTileIterator::Layout;
  static FillMode const kFillModeC = FillModeC_;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;
  static BlasMode const kBlasMode = BlasMode_;


  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma1::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;


  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmUniversalMode mode;
    GemmCoord problem_size;
    int batch_count;

    typename EpilogueOutputOp::Params epilogue;

    void const * ptr_A;
    void const * ptr_B;
    void const * ptr_C;
    void * ptr_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

    typename LayoutA::Stride::Index lda;
    typename LayoutB::Stride::Index ldb;
    typename LayoutC::Stride::Index ldc;
    typename LayoutC::Stride::Index ldd;

    //
    // Methods
    //
    
    Arguments(): 
      mode(GemmUniversalMode::kGemm), 
      batch_count(1), 
      ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr), ptr_D(nullptr) { }

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      typename LayoutA::Stride::Index lda,
      typename LayoutB::Stride::Index ldb,
      typename LayoutC::Stride::Index ldc,
      typename LayoutC::Stride::Index ldd
    ):
      mode(mode), 
      problem_size(problem_size), 
      batch_count(batch_count),
      epilogue(epilogue), 
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D), 
      batch_stride_A(batch_stride_A), batch_stride_C(batch_stride_C), batch_stride_D(batch_stride_D), 
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd) {

      }

      /// Returns arguments for a the transposed problem
      Arguments transposed_problem() const {
        Arguments args(*this);
        
        std::swap(args.ptr_A, args.ptr_B);
        std::swap(args.lda, args.ldb);
        std::swap(args.batch_stride_A, args.batch_stride_B);

        return args;
      }

  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    
    // Mma1 Iterator A and B params
    typename Mma1::IteratorA::Params params_A;
    typename Mma1::IteratorB::Params params_BT;

    // Mma2 Iterator A and B params 
    typename Mma2::IteratorA::Params params_B;
    typename Mma2::IteratorB::Params params_AT;

    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    
    typename EpilogueOutputOp::Params output_op;

    GemmUniversalMode mode;
    int batch_count;
    int gemm_k_size;

    void * ptr_A;
    void * ptr_B;
    void * ptr_C;
    void * ptr_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

    int *semaphore;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      swizzle_log_tile(0),
      params_A(0),
      params_BT(0),
      params_B(0),
      params_AT(0),
      params_C(0),
      params_D(0),
      batch_count(0),
      gemm_k_size(0),
      mode(cutlass::gemm::GemmUniversalMode::kGemm),
      ptr_A(nullptr),
      ptr_B(nullptr),
      ptr_C(nullptr),
      ptr_D(nullptr),
      batch_stride_A(0),
      batch_stride_B(0),
      batch_stride_C(0),
      batch_stride_D(0),
      semaphore(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      int gemm_k_size,
      void *workspace = nullptr
    ):
      problem_size(args.problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(args.lda),
      params_BT(args.ldb),
      params_B(args.ldb),
      params_AT(args.lda),
      params_C(args.ldc),
      params_D(args.ldd),
      output_op(args.epilogue),
      mode(args.mode),
      batch_count(args.batch_count),
      gemm_k_size(gemm_k_size),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C(const_cast<void *>(args.ptr_C)),
      ptr_D(const_cast<void *>(args.ptr_D)),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C(args.batch_stride_C),
      batch_stride_D(args.batch_stride_D),
      semaphore(static_cast<int *>(workspace)) {
    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr) {

      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C = const_cast<void *>(args.ptr_C);
      ptr_D = args.ptr_D;

      output_op = args.epilogue;

      semaphore = static_cast<int *>(workspace);
    }

  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma1::SharedStorage mma1_main_loop;
    typename Mma2::SharedStorage mma2_main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  Rank2KUniversal() { } 

  /// Determines whether kernel satisfies alignment
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size) {

    static int const kAlignmentA = Mma1::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma1::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if ((problem_size.m() % kAlignmentA) || (problem_size.k() % kAlignmentA) ||
      (problem_size.n() % kAlignmentB) || (problem_size.k() % kAlignmentB) ||
      (problem_size.m() % kAlignmentC) || (problem_size.n() % kAlignmentC)) {

      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }
   
    // Early exit if Fill Mode is Lower and
    // if the entire tile is above the main diagonal (bottom-left corner is at or above the diagonal)
    if (kFillModeC == cutlass::FillMode::kLower &&
        (threadblock_tile_offset.m() + 1) * Mma1::Shape::kM <= threadblock_tile_offset.n() * Mma1::Shape::kN) {
      return;
    }    
    
    // Early exit if Fill Mode is Upper and
    // if the entire tile is below the main diagonal (top-right corner is at or below the diagonal)
    if (kFillModeC == cutlass::FillMode::kUpper &&
        threadblock_tile_offset.m() * Mma1::Shape::kM >= (threadblock_tile_offset.n() + 1) * Mma1::Shape::kN) {
      return;
    }    
    
    bool tile_on_diagonal = false;
    // Mark tiles that are being crossed by the main diagonal
    // (top-right and bottom-left corners are on either side of the diagonal)
    if ((threadblock_tile_offset.m() + 1) * Mma1::Shape::kM > threadblock_tile_offset.n() * Mma1::Shape::kN
        && threadblock_tile_offset.m() * Mma1::Shape::kM < (threadblock_tile_offset.n() + 1) * Mma1::Shape::kN) {
      tile_on_diagonal = true;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A); 
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    //
    // Fetch pointers based on mode.
    //
    if (params.mode == GemmUniversalMode::kGemm || 
      params.mode == GemmUniversalMode::kGemmSplitKParallel) {

      if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {

        problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size; 
      }

      offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
    }

    __syncthreads();

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_MxK{
      threadblock_tile_offset.m() * Mma1::Shape::kM,
      offset_k,
    };

    cutlass::MatrixCoord tb_offset_KxN{
      offset_k,
      threadblock_tile_offset.n() * Mma1::Shape::kN
    };


    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands for Mma1
    typename Mma1::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_MxK);

    typename Mma1::IteratorB iterator_BT(
      params.params_BT,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_KxN);

    // Construct iterators to A and B operands for Mma2
    typename Mma2::IteratorA iterator_B(
      params.params_B,
      ptr_B,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_MxK);

    typename Mma2::IteratorB iterator_AT(
      params.params_AT,
      ptr_A,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_KxN);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply for Mma1 (A x BT)
    Mma1 mma1(shared_storage.mma1_main_loop, thread_idx, warp_idx, lane_idx);

    // Construct thread-scoped matrix multiply for Mma2 (B x AT)
    Mma2 mma2(shared_storage.mma2_main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma1::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - offset_k + Mma1::Shape::kK - 1) / Mma1::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add (A x BT)
    mma1(
      gemm_k_iterations, 
      accumulators, 
      iterator_A, 
      iterator_BT, 
      accumulators);

    // HER2K kernel needs Alpha to be complex and is conj(Alpha) is applied to the second HERK.
    if (kBlasMode == BlasMode::kHermitian) {

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      //
      // Masked tile iterators constructed from members
      //

      threadblock_tile_offset =
          threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

      //assume identity swizzle
      MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * Mma1::Shape::kM,
        threadblock_tile_offset.n() * Mma1::Shape::kN
      );

      int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

      ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C); 
      ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);

      //
      // Fetch pointers based on mode.
      //
      
      // Construct the semaphore.
      Semaphore semaphore(params.semaphore + block_idx, thread_idx);

      if (params.mode == GemmUniversalMode::kGemm) {

        // If performing a reduction via split-K, fetch the initial synchronization
        if (params.grid_tiled_shape.k() > 1) {
          
          // Fetch the synchronization lock initially but do not block.
          semaphore.fetch();

          // Indicate which position in a serial reduction the output operator is currently updating
          output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
        }
      }
      else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
        ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
      }
      else if (params.mode == GemmUniversalMode::kBatched) {
        ptr_C += threadblock_tile_offset.k() * params.batch_stride_C;
        ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
      }
      else if (params.mode == GemmUniversalMode::kArray) {
        ptr_C = static_cast<ElementC * const *>(params.ptr_C)[threadblock_tile_offset.k()];
        ptr_D = static_cast<ElementC * const *>(params.ptr_D)[threadblock_tile_offset.k()];
      }

      
      // If CTA not on diagonal, FillMode doesn't apply. 
      FillMode kFillModeCTA = tile_on_diagonal ? kFillModeC : FillMode::kNone;

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        ptr_C,
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset,
        kFillModeCTA
      );

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        ptr_D,
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset,
        kFillModeCTA
      );

      Epilogue epilogue(
        shared_storage.epilogue, 
        thread_idx, 
        warp_idx, 
        lane_idx);

      // Wait on the semaphore - this latency may have been covered by iterator construction
      if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {
          
        // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
        if (threadblock_tile_offset.k()) {
          iterator_C = iterator_D;
        }

        semaphore.wait(threadblock_tile_offset.k());

        __threadfence();
      }

      // Execute the epilogue operator to update the destination tensor.
      epilogue(
        output_op, 
        iterator_D, 
        accumulators, 
        iterator_C); 
      
      //
      // Release the semaphore
      //

      if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) { 

        int lock = 0;
        if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

          // The final threadblock resets the semaphore for subsequent grids.
          lock = 0;
        }
        else {
          // Otherwise, the semaphore is incremented
          lock = threadblock_tile_offset.k() + 1;
        }
        
        semaphore.release(lock);
      }

      __syncthreads();

      accumulators.clear();
    }

    // Compute threadblock-scoped matrix multiply-add (B x AT)
    mma2(
      gemm_k_iterations, 
      accumulators, 
      iterator_B, 
      iterator_AT, 
      accumulators);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    /* Needed for HER2K where the second HERK is multiplied by conj(alpha) */
    typename EpilogueOutputOp::Params second_her2k_params(conj(params.output_op.alpha), 1);
    EpilogueOutputOp output_op_her2k(second_her2k_params);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma1::Shape::kM,
      threadblock_tile_offset.n() * Mma1::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C);

    // HER2K kernel needs Alpha to be complex and is conj(Alpha) is applied to the second HERK.
    if (kBlasMode == BlasMode::kHermitian) {
      ptr_C = static_cast<ElementC *>(params.ptr_D);
    }

    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);

    //
    // Fetch pointers based on mode.
    //
    
    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if (params.mode == GemmUniversalMode::kGemm) {

      // If performing a reduction via split-K, fetch the initial synchronization
      if (params.grid_tiled_shape.k() > 1) {
        
        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is currently updating
        if (kBlasMode == BlasMode::kSymmetric) {
          output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
        } else {
          output_op_her2k.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
        }
      }
    }
    else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C += threadblock_tile_offset.k() * params.batch_stride_C;
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_C = static_cast<ElementC * const *>(params.ptr_C)[threadblock_tile_offset.k()];
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[threadblock_tile_offset.k()];
    }

    
    // If CTA not on diagonal, FillMode doesn't apply. 
    FillMode kFillModeCTA = tile_on_diagonal ? kFillModeC : FillMode::kNone;

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      ptr_C,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      kFillModeCTA
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      ptr_D,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      kFillModeCTA
    );

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }

    // Execute the epilogue operator to update the destination tensor.
    if (kBlasMode == BlasMode::kSymmetric) {
      epilogue(
        output_op,
        iterator_D,
        accumulators,
        iterator_C);
    } else {
      epilogue(
        output_op_her2k,
        iterator_D,
        accumulators,
        iterator_C);
    }
    
    //
    // Release the semaphore
    //

    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) { 

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }
      
      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
