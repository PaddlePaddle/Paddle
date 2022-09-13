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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename B2bMma_,               ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
>
struct B2bGemm {

  using B2bMma = B2bMma_;
  using Epilogue = Epilogue_;
  using OutputOp0 = typename B2bMma::OutputOp;
  using OutputOp1 = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount0 = typename B2bMma::WarpCount0;
  static int const kThreadCount = 32 * WarpCount0::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size_0;
    cutlass::gemm::GemmCoord problem_size_1;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename B2bMma::IteratorA0::Params params_A0;
    typename B2bMma::IteratorA0::TensorRef ref_A0;
    typename B2bMma::IteratorB0::Params params_B0;
    typename B2bMma::IteratorB0::TensorRef ref_B0;
    typename Epilogue::OutputTileIterator::Params params_C0;
    typename Epilogue::OutputTileIterator::TensorRef ref_C0;
    typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Scale0;
    typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Bias0;
    typename B2bMma::IteratorB1::Params params_B1;
    typename B2bMma::IteratorB1::TensorRef ref_B1;
    typename Epilogue::OutputTileIterator::Params params_C1;
    typename Epilogue::OutputTileIterator::TensorRef ref_C1;
    typename Epilogue::OutputTileIterator::Params params_D1;
    typename Epilogue::OutputTileIterator::TensorRef ref_D1;
    typename OutputOp0::Params output_op_0;
    typename OutputOp1::Params output_op_1;
    int *semaphore;
    int gemm_k_iterations_0;
    int gemm_k_size_0;
    int gemm_k_iterations_1;
    int gemm_k_size_1;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), semaphore(0), gemm_k_iterations_0(0), gemm_k_size_0(0),
        gemm_k_iterations_1(0), gemm_k_size_1(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size_0,
      cutlass::gemm::GemmCoord const & problem_size_1,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename B2bMma::IteratorA0::TensorRef ref_A0,
      typename B2bMma::IteratorB0::TensorRef ref_B0,
      typename Epilogue::OutputTileIterator::TensorRef ref_C0,
      typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Scale0,
      typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Bias0,
      typename B2bMma::IteratorB1::TensorRef ref_B1,
      typename Epilogue::OutputTileIterator::TensorRef ref_C1,
      typename Epilogue::OutputTileIterator::TensorRef ref_D1,
      typename OutputOp0::Params output_op_0 = typename OutputOp0::Params(),
      typename OutputOp1::Params output_op_1 = typename OutputOp1::Params(),
      int *workspace = nullptr
    ):
      problem_size_0(problem_size_0),
      problem_size_1(problem_size_1),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A0(ref_A0.layout()),
      ref_A0(ref_A0),
      params_B0(ref_B0.layout()),
      ref_B0(ref_B0),
      params_C0(ref_C0.layout()),
      ref_C0(ref_C0),
      ref_Scale0(ref_Scale0),
      ref_Bias0(ref_Bias0),
      params_B1(ref_B1.layout()),
      ref_B1(ref_B1),
      params_C1(ref_C1.layout()),
      ref_C1(ref_C1),
      params_D1(ref_D1.layout()),
      ref_D1(ref_D1),
      output_op_0(output_op_0),
      output_op_1(output_op_1) {

      int total_gemm_k_iterations_0 = (problem_size_0.k() + B2bMma::Shape0::kK - 1) / B2bMma::Shape0::kK;
      int gemm_k_iterations_0 = (total_gemm_k_iterations_0 + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      gemm_k_size_0 = gemm_k_iterations_0 * B2bMma::Shape0::kK;
      int total_gemm_k_iterations_1 = (problem_size_1.k() + B2bMma::Shape1::kK - 1) / B2bMma::Shape1::kK;
      int gemm_k_iterations_1 = (total_gemm_k_iterations_1 + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      gemm_k_size_1 = gemm_k_iterations_1 * B2bMma::Shape1::kK;

    semaphore = workspace;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename B2bMma::B2bMmaSharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  B2bGemm() { } 

  /// Determines whether kernel satisfies alignment
    static Status can_implement(
      cutlass::gemm::GemmCoord const & problem_size_0,
      cutlass::gemm::GemmCoord const & problem_size_1,
      typename B2bMma::IteratorA0::TensorRef ref_A0,
      typename B2bMma::IteratorB0::TensorRef ref_B0,
      typename Epilogue::OutputTileIterator::TensorRef ref_C0,
      typename B2bMma::IteratorB1::TensorRef ref_B1,
      typename Epilogue::OutputTileIterator::TensorRef ref_C1,
      typename Epilogue::OutputTileIterator::TensorRef ref_D1) {

    static int const kAlignmentA = B2bMma::IteratorA0::AccessType::kElements;
    static int const kAlignmentB = B2bMma::IteratorB0::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A0, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B0, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C0, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B1, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C1, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D1, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if ((problem_size_0.m() % kAlignmentA) || (problem_size_0.k() % kAlignmentA) ||
      (problem_size_0.n() % kAlignmentB) || (problem_size_0.k() % kAlignmentB) ||
      (problem_size_0.m() % kAlignmentC) || (problem_size_0.n() % kAlignmentC) ||
      (problem_size_1.m() % kAlignmentA) || (problem_size_1.k() % kAlignmentA) ||
      (problem_size_1.n() % kAlignmentB) || (problem_size_1.k() % kAlignmentB) ||
      (problem_size_1.m() % kAlignmentC) || (problem_size_1.n() % kAlignmentC)) {

      return Status::kErrorMisalignedOperand;
    }

    // Determine if fusion sizes are valid
    if(problem_size_0.m() != problem_size_1.m())
      return Status::kErrorInvalidProblem;

    if(problem_size_0.n() != problem_size_1.k())
      return Status::kErrorInvalidProblem;

    if(problem_size_0.n() > B2bMma::Shape0::kN)
      return Status::kErrorInvalidProblem;
    
    if(problem_size_1.n() > B2bMma::Shape1::kN)
      return Status::kErrorInvalidProblem;

    return Status::kSuccess;
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

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A0{
      threadblock_tile_offset.m() * B2bMma::Shape0::kM,
      threadblock_tile_offset.k() * params.gemm_k_size_0,
    };

    cutlass::MatrixCoord tb_offset_B0{
      threadblock_tile_offset.k() * params.gemm_k_size_0,
      threadblock_tile_offset.n() * B2bMma::Shape0::kN
    };

    cutlass::MatrixCoord tb_offset_B1{
      threadblock_tile_offset.k() * params.gemm_k_size_1,
      threadblock_tile_offset.n() * B2bMma::Shape1::kN
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k_0 = min(
      params.problem_size_0.k(), 
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size_0);

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations_0 = (problem_size_k_0 - tb_offset_A0.column() + B2bMma::Shape0::kK - 1) / B2bMma::Shape0::kK;

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k_1 = min(
      params.problem_size_1.k(), 
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size_1);

    // Compute threadblock-scoped matrix multiply-add
//    int gemm_k_iterations_1 = (problem_size_k_1 - tb_offset_B1.row() + B2bMma::Shape1::kK - 1) / B2bMma::Shape1::kK;


    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename B2bMma::IteratorA0 iterator_A0(
      params.params_A0,
      params.ref_A0.data(),
      {params.problem_size_0.m(), problem_size_k_0},
      thread_idx,
      tb_offset_A0);

    typename B2bMma::IteratorB0 iterator_B0(
      params.params_B0,
      params.ref_B0.data(),
      {problem_size_k_0, params.problem_size_0.n()},
      thread_idx,
      tb_offset_B0);

    typename B2bMma::IteratorB1 iterator_B1(
      params.params_B1,
      params.ref_B1.data(),
      {problem_size_k_1, params.problem_size_1.n()},
      thread_idx,
      tb_offset_B1);


    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0x1f, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    // Construct iterators to accumulator scale/bias vector
    typename B2bMma::IteratorAccumulatorScaleBias iterator_Scale0(
      params.ref_Scale0.data(),
      {1, params.problem_size_0.n()},
      thread_idx,
      warp_idx,
      MatrixCoord(
        0, threadblock_tile_offset.n() * B2bMma::Shape0::kN
      )
    );

    typename B2bMma::IteratorAccumulatorScaleBias iterator_Bias0(
      params.ref_Bias0.data(),
      {1, params.problem_size_0.n()},
      thread_idx,
      warp_idx,
      MatrixCoord(
        0, threadblock_tile_offset.n() * B2bMma::Shape0::kN
      )
    );



    //
    // Main loop
    //

    OutputOp0 output_op_0(params.output_op_0);

    // Construct thread-scoped matrix multiply
    B2bMma b2bMma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename B2bMma::FragmentC0 src_accum;
    typename B2bMma::FragmentC1 accumulators;

    src_accum.clear();
    accumulators.clear();

    if (!kSplitKSerial || gemm_k_iterations_0 > 0) {
      // Compute threadblock-scoped matrix multiply-add
      b2bMma(gemm_k_iterations_0, accumulators, iterator_A0, iterator_B0,
        iterator_Scale0, iterator_Bias0, iterator_B1, src_accum, output_op_0);
    }

    //
    // Epilogue
    //

    OutputOp1 output_op_1(params.output_op_1);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * B2bMma::Shape1::kM,
      threadblock_tile_offset.n() * B2bMma::Shape1::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op_1.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C1(
      params.params_C1,
      params.ref_C1.data(),
      params.problem_size_1.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D1(
      params.params_D1,
      params.ref_D1.data(),
      params.problem_size_1.mn(),
      thread_idx,
      threadblock_offset
    );

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C1 = iterator_D1;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op_1, iterator_D1, accumulators, iterator_C1); 
    
    //
    // Release the semaphore
    //

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      __threadfence();
      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

