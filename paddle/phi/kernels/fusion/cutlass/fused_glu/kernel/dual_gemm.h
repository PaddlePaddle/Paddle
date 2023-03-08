// /***************************************************************************************************
//  * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//  * SPDX-License-Identifier: BSD-3-Clause
//  *
//  * Redistribution and use in source and binary forms, with or without
//  * modification, are permitted provided that the following conditions are met:
//  *
//  * 1. Redistributions of source code must retain the above copyright notice, this
//  * list of conditions and the following disclaimer.
//  *
//  * 2. Redistributions in binary form must reproduce the above copyright notice,
//  * this list of conditions and the following disclaimer in the documentation
//  * and/or other materials provided with the distribution.
//  *
//  * 3. Neither the name of the copyright holder nor the names of its
//  * contributors may be used to endorse or promote products derived from
//  * this software without specific prior written permission.
//  *
//  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//  *
//  **************************************************************************************************/
// /*! \file
//     \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
// */

// #pragma once

// #include "cutlass/cutlass.h"

// #include "cutlass/gemm/gemm.h"
// #include "cutlass/matrix_coord.h"
// #include "cutlass/semaphore.h"

// #include "../threadblock/mma_from_smem.h"

// #include "../threadblock/dual_epilogue.h"

// /////////////////////////////////////////////////////////////////////////////////////////////////

// namespace cutlass {
// namespace gemm {
// namespace kernel {

// /////////////////////////////////////////////////////////////////////////////////////////////////

// template <
//   typename DualMma_,               ///! Threadblock-scoped matrix multiply-accumulate 
//   typename Epilogue0_,             ///! Epilogue
//   typename Epilogue1_,             ///! Epilogue
//   typename OutputOp2_,             ///! Epilogue
//   typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
//   bool SplitKSerial,              ///! If true, code supporting split-K via serial reduction is enabled.
//   bool StoreD0,
//   bool StoreD1
// >
// struct DualGemm {

//   using DualMma = DualMma_;

//   using Epilogue0 = Epilogue0_;
//   using Epilogue1 = Epilogue1_;
//   using OutputOp0 = typename Epilogue0::OutputOp;
//   using OutputOp1 = typename Epilogue1::OutputOp;
//   using OutputOp2 = OutputOp2_;
//   using ThreadblockSwizzle = ThreadblockSwizzle_;
//   static constexpr bool kStoreD0 = StoreD0;
//   static constexpr bool kStoreD1 = StoreD1;

//   using DualEpilogue = cutlass::epilogue::threadblock::DualEpilogue<
//       typename Epilogue0::Shape,
//       typename Epilogue0::WarpMmaOperator,
//       Epilogue0::kPartitionsK,
//       typename Epilogue0::OutputTileIterator,
//       typename Epilogue0::AccumulatorFragmentIterator,
//       typename Epilogue0::WarpTileIterator,
//       typename Epilogue0::SharedLoadIterator,
//       OutputOp0,
//       OutputOp1,
//       OutputOp2,
//       typename Epilogue0::Padding,
//       kStoreD0,
//       kStoreD1,
//       Epilogue0::kFragmentsPerIteration,
//       true // IterationsUnroll
//   >;

//   static bool const kSplitKSerial = SplitKSerial;
//   static_assert(!kSplitKSerial || (kStoreD0 && kStoreD1),
//     "Split-K serial requires buffers for D0/D1 for reduction");

//   /// Warp count (concept: GemmShape)
//   using WarpCount0 = typename DualMma::WarpCount;
//   static int const kThreadCount = 32 * WarpCount0::kCount;

//   /// Parameters structure
//   struct Params {
//     cutlass::gemm::GemmCoord problem_size;
//     cutlass::gemm::GemmCoord grid_tiled_shape;
//     int swizzle_log_tile;

//     // Mma0
//     typename DualMma::IteratorA::Params params_A0;
//     typename DualMma::IteratorA::TensorRef ref_A0;
//     typename DualMma::IteratorB::Params params_B0;
//     typename DualMma::IteratorB::TensorRef ref_B0;
//     typename Epilogue0::OutputTileIterator::Params params_C0;
//     typename Epilogue0::OutputTileIterator::TensorRef ref_C0;
//     typename Epilogue0::OutputTileIterator::Params params_D0;
//     typename Epilogue0::OutputTileIterator::TensorRef ref_D0;
//     typename OutputOp0::Params output_op_0;

//     // Mma1
//     typename DualMma::IteratorB::Params params_B1;
//     typename DualMma::IteratorB::TensorRef ref_B1;
//     typename Epilogue1::OutputTileIterator::Params params_C1;
//     typename Epilogue1::OutputTileIterator::TensorRef ref_C1;
//     typename Epilogue1::OutputTileIterator::Params params_D1;
//     typename Epilogue1::OutputTileIterator::TensorRef ref_D1;
//     typename OutputOp1::Params output_op_1;

//     typename Epilogue1::OutputTileIterator::Params params_D2;
//     typename Epilogue1::OutputTileIterator::TensorRef ref_D2;
//     typename OutputOp2::Params output_op_2;

//     int *semaphore;
//     int gemm_k_size;

//     //
//     // Methods
//     //

//     CUTLASS_HOST_DEVICE
//     Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0) { }

//     CUTLASS_HOST_DEVICE
//     Params(
//       cutlass::gemm::GemmCoord const & problem_size,
//       cutlass::gemm::GemmCoord const & grid_tiled_shape,
//       // Mma0: D0 = A @ B0 + C0
//       typename DualMma::IteratorA::TensorRef ref_A0,
//       typename DualMma::IteratorB::TensorRef ref_B0,
//       typename Epilogue0::OutputTileIterator::TensorRef ref_C0,
//       typename Epilogue0::OutputTileIterator::TensorRef ref_D0,
//       // Mma1: D1 = A @ B1 + C1
//       typename DualMma::IteratorB::TensorRef ref_B1,
//       typename Epilogue1::OutputTileIterator::TensorRef ref_C1,
//       typename Epilogue1::OutputTileIterator::TensorRef ref_D1,

//       typename Epilogue1::OutputTileIterator::TensorRef ref_D2,
//       typename OutputOp0::Params output_op_0 = typename OutputOp0::Params(),
//       typename OutputOp1::Params output_op_1 = typename OutputOp1::Params(),
//       typename OutputOp2::Params output_op_2 = typename OutputOp2::Params(),
//       int *workspace = nullptr
//     ):
//       problem_size(problem_size),
//       grid_tiled_shape(grid_tiled_shape),
//       swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
//       // Mma0
//       params_A0(ref_A0.layout()),
//       ref_A0(ref_A0),
//       params_B0(ref_B0.layout()),
//       ref_B0(ref_B0),
//       params_C0(ref_C0.layout()),
//       ref_C0(ref_C0),
//       params_D0(ref_D0.layout()),
//       ref_D0(ref_D0),
//       // Mma1
//       params_B1(ref_B1.layout()),
//       ref_B1(ref_B1),
//       params_C1(ref_C1.layout()),
//       ref_C1(ref_C1),
//       params_D1(ref_D1.layout()),
//       ref_D1(ref_D1),
//       params_D2(ref_D2.layout()),
//       ref_D2(ref_D2),
//       output_op_0(output_op_0),
//       output_op_1(output_op_1),
//       output_op_2(output_op_2) {

//       int total_gemm_k_iterations = (problem_size.k() + DualMma::Shape::kK - 1) / DualMma::Shape::kK;
//       int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
//       gemm_k_size = gemm_k_iterations * DualMma::Shape::kK;

//     semaphore = workspace;
//     }
//   };

//   /// Shared memory storage structure
//   union SharedStorage {
//     typename DualMma::SharedStorage main_loop;
//     typename DualEpilogue::SharedStorage epilogue;
//   };

//   //
//   // Methods
//   //

//   CUTLASS_HOST_DEVICE
//   DualGemm() { }

//   /// Determines whether kernel satisfies alignment
//     static Status can_implement(
//       cutlass::gemm::GemmCoord const & problem_size,
//       typename DualMma::IteratorA::TensorRef ref_A0,
//       typename DualMma::IteratorB::TensorRef ref_B0,
//       typename Epilogue0::OutputTileIterator::TensorRef ref_C0,
//       typename Epilogue0::OutputTileIterator::TensorRef ref_D0,
//       typename DualMma::IteratorB::TensorRef ref_B1,
//       typename Epilogue1::OutputTileIterator::TensorRef ref_C1,
//       typename Epilogue1::OutputTileIterator::TensorRef ref_D1,
//       typename Epilogue1::OutputTileIterator::TensorRef ref_D2) {

//     static int const kAlignmentA = DualMma::IteratorA::AccessType::kElements;
//     static int const kAlignmentB = DualMma::IteratorB::AccessType::kElements;
//     static int const kAlignmentC = Epilogue0::OutputTileIterator::kElementsPerAccess;

//     if (!TensorRef_aligned(ref_A0, kAlignmentA)) {
//       return Status::kErrorMisalignedOperand;
//     }

//     if (!TensorRef_aligned(ref_B0, kAlignmentB)) {
//       return Status::kErrorMisalignedOperand;
//     }

//     if (!TensorRef_aligned(ref_C0, kAlignmentC)) {
//       return Status::kErrorMisalignedOperand;
//     }

//     if (!TensorRef_aligned(ref_D0, kAlignmentC)) {
//       return Status::kErrorMisalignedOperand;
//     }

//     if (!TensorRef_aligned(ref_B1, kAlignmentB)) {
//       return Status::kErrorMisalignedOperand;
//     }

//     if (!TensorRef_aligned(ref_C1, kAlignmentC)) {
//       return Status::kErrorMisalignedOperand;
//     }

//     if (!TensorRef_aligned(ref_D1, kAlignmentC)) {
//       return Status::kErrorMisalignedOperand;
//     }

//     if (!TensorRef_aligned(ref_D2, kAlignmentC)) {
//       return Status::kErrorMisalignedOperand;
//     }

//     return Status::kSuccess;
//   }

//   /// Executes one GEMM
//   CUTLASS_DEVICE
//   void operator()(Params const &params, SharedStorage &shared_storage) {
//     // Compute threadblock location
//     ThreadblockSwizzle threadblock_swizzle;

//     cutlass::gemm::GemmCoord threadblock_tile_offset =
//         threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

//     // Early exit if CTA is out of range
//     if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
//       params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

//       return;
//     }

//     // Compute initial location in logical coordinates
//     cutlass::MatrixCoord tb_offset_A0{
//       threadblock_tile_offset.m() * DualMma::Shape::kM,
//       threadblock_tile_offset.k() * params.gemm_k_size,
//     };

//     cutlass::MatrixCoord tb_offset_B0{
//       threadblock_tile_offset.k() * params.gemm_k_size,
//       threadblock_tile_offset.n() * DualMma::Shape::kN
//     };

//     cutlass::MatrixCoord tb_offset_B1{
//       threadblock_tile_offset.k() * params.gemm_k_size,
//       threadblock_tile_offset.n() * DualMma::Shape::kN
//     };

//     // Problem size is a function of threadblock index in the K dimension
//     int problem_size_k =
//       (params.problem_size.k() < (threadblock_tile_offset.k() + 1) * params.gemm_k_size) ?
//        params.problem_size.k() :
//        (threadblock_tile_offset.k() + 1) * params.gemm_k_size;

//     // Compute threadblock-scoped matrix multiply-add
//     int gemm_k_iterations = (problem_size_k - tb_offset_A0.column() + DualMma::Shape::kK - 1) / DualMma::Shape::kK;

//     // Compute position within threadblock
//     int thread_idx = threadIdx.x;

//     // Construct iterators to A and B operands
//     typename DualMma::IteratorA iterator_A0(
//       params.params_A0,
//       params.ref_A0.data(),
//       {params.problem_size.m(), problem_size_k},
//       thread_idx,
//       tb_offset_A0);

//     typename DualMma::IteratorB iterator_B0(
//       params.params_B0,
//       params.ref_B0.data(),
//       {problem_size_k, params.problem_size.n()},
//       thread_idx,
//       tb_offset_B0);

//     typename DualMma::IteratorB iterator_B1(
//       params.params_B1,
//       params.ref_B1.data(),
//       {problem_size_k, params.problem_size.n()},
//       thread_idx,
//       tb_offset_B1);


//     // Broadcast the warp_id computed by lane 0 to ensure dependent code
//     // is compiled as warp-uniform.
//     int warp_idx = __shfl_sync(0x1f, threadIdx.x / 32, 0);
//     int lane_idx = threadIdx.x % 32;

//     //
//     // Main loop
//     //


//     // Construct thread-scoped matrix multiply
//     typename DualMma::FragmentC accum0;
//     typename DualMma::FragmentC accum1;
//     accum0.clear();
//     accum1.clear();

//     DualMma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
//     if (!kSplitKSerial || gemm_k_iterations > 0) {
//       // Compute threadblock-scoped matrix multiply-add
//       mma(gemm_k_iterations,
//         accum0, accum1,
//         iterator_A0, iterator_B0, iterator_B1,
//         accum0, accum1);
//     }

//     //
//     // Epilogue
//     //

//     OutputOp0 output_op_0(params.output_op_0);
//     OutputOp1 output_op_1(params.output_op_1);
//     OutputOp2 output_op_2(params.output_op_2);

//     //
//     // Masked tile iterators constructed from members
//     //

//     threadblock_tile_offset =
//         threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

//     //assume identity swizzle
//     MatrixCoord threadblock_offset(
//       threadblock_tile_offset.m() * DualMma::Shape::kM,
//       threadblock_tile_offset.n() * DualMma::Shape::kN
//     );

//     int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

//     // Construct the semaphore.
//     Semaphore semaphore(params.semaphore + block_idx, thread_idx);

//     // If performing a reduction via split-K, fetch the initial synchronization
//     if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
//       // Fetch the synchronization lock initially but do not block.
//       semaphore.fetch();

//       // Indicate which position in a serial reduction the output operator is currently updating
//       output_op_0.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
//       output_op_1.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
//     }

//     // Tile iterator loading from source tensor.
//     typename Epilogue0::OutputTileIterator iterator_C0(
//       params.params_C0,
//       params.ref_C0.data(),
//       params.problem_size.mn(),
//       thread_idx,
//       threadblock_offset
//     );
//     typename Epilogue1::OutputTileIterator iterator_C1(
//       params.params_C1,
//       params.ref_C1.data(),
//       params.problem_size.mn(),
//       thread_idx,
//       threadblock_offset
//     );

//     // Tile iterator writing to destination tensor.
//     typename Epilogue0::OutputTileIterator iterator_D0(
//       params.params_D0,
//       params.ref_D0.data(),
//       params.problem_size.mn(),
//       thread_idx,
//       threadblock_offset
//     );
//     typename Epilogue1::OutputTileIterator iterator_D1(
//       params.params_D1,
//       params.ref_D1.data(),
//       params.problem_size.mn(),
//       thread_idx,
//       threadblock_offset
//     );
//     typename Epilogue1::OutputTileIterator iterator_D2(
//       params.params_D2,
//       params.ref_D2.data(),
//       params.problem_size.mn(),
//       thread_idx,
//       threadblock_offset
//     );

//     DualEpilogue epilogue(
//       shared_storage.epilogue,
//       thread_idx, 
//       warp_idx, 
//       lane_idx);

//     // Wait on the semaphore - this latency may have been covered by iterator construction
//     if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
//       // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
//       if (threadblock_tile_offset.k()) {
//         iterator_C0 = iterator_D0;
//         iterator_C1 = iterator_D1;
//       }

//       semaphore.wait(threadblock_tile_offset.k());

//       __threadfence();
//     }

//     // Execute the epilogue operator to update the destination tensor.
//     typename Epilogue0::OutputTileIterator source_iters[] = {
//       iterator_C0, iterator_C1
//     };
//     const bool writeToD2 = (!kSplitKSerial || params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1);
//     epilogue(
//       output_op_0, output_op_1, output_op_2,
//       iterator_D0, iterator_D1, iterator_D2,
//       accum0, accum1,
//       source_iters,
//       writeToD2
//     );
    
//     //
//     // Release the semaphore
//     //

//     if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
//       int lock = 0;
//       if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

//         // The final threadblock resets the semaphore for subsequent grids.
//         lock = 0;
//       }
//       else {
//         // Otherwise, the semaphore is incremented
//         lock = threadblock_tile_offset.k() + 1;
//       }

//       __threadfence();
//       semaphore.release(lock);
//     }
//   }
// };

// /////////////////////////////////////////////////////////////////////////////////////////////////

// } // namespace kernel
// } // namespace gemm
// } // namespace cutlass











/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma.h"

#include "../threadblock/mma_from_smem.h"
#include "../threadblock/dual_epilogue.h"
#include "paddle/phi/kernels/fusion/cutlass/fused_glu/thread/right_act_and_mul.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Element type for A matrix operand
    typename ElementA_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// store for backward
    bool StoreD,
    /// Activate Type
    template <typename> class ActivationType, 
    /// Tag indicating architecture to tune for
    typename ArchTag_>
struct DualGemm {
  using ArchTag = ArchTag_;
  using ElementA = ElementA_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementAccumulator_;
  using ElementOutput = ElementA_; 

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor; 
  using LayoutC = cutlass::layout::RowMajor; 

  using GemmType = gemm_kernel_utils::DefaultGemmType<ArchTag, ElementA>;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, GemmType::ThreadK>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, GemmType::WarpK>;
  using InstructionShape = typename GemmType::InstructionShape;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  using OpClass = typename GemmType::OpClass;

  static int const AlignmentA =
      cutlass::gemm::device::DefaultGemmConfiguration<OpClass, ArchTag_, ElementA, ElementA,
                                ElementA, ElementAccumulator>::kAlignmentA;
  static int const AlignmentB =
      cutlass::gemm::device::DefaultGemmConfiguration<OpClass, ArchTag_, ElementA, ElementA,
                                ElementA, ElementAccumulator>::kAlignmentB;
  static bool const kSplitKSerial = false;

  using Operator = typename cutlass::gemm::device::DefaultGemmConfiguration<OpClass, 
                                                                            ArchTag_, 
                                                                            ElementA, 
                                                                            ElementA, 
                                                                            ElementA, 
                                                                            ElementAccumulator>::Operator; 

  using DefaultConfig =
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OpClass,
        ArchTag,
        ElementA,
        ElementA,
        ElementA, // ElementC
        ElementAccumulator // ElementAccumulator
        >;

  using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        ElementA, // ElementA,
        LayoutA, // LayoutA,
        AlignmentA,
        ElementA, // ElementB,
        LayoutB, // LayoutB,
        AlignmentB,
        ElementA,
        cutlass::layout::RowMajor, // LayoutC,
        ElementAccumulator,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        ThreadblockSwizzle, 
        DefaultConfig::kStages, 
        kSplitKSerial, // SplitKSerial
        Operator>;

  using DualMmaFromSmem =
        typename threadblock::DualMmaFromSharedMemory<
            typename DefaultGemm::Mma>; 
  using DualMma = typename DualMmaFromSmem::Mma;

  using EpilogueOutputMatmulOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using EpilogueOutputOp = cutlass::epilogue::thread::RightActAndMul<
      ActivationType,
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementOutput,
      ElementCompute>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  
  /// Define the epilogue
  using Epilogue0 =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename DualMma::Operator, kPartitionsK, EpilogueOutputMatmulOp,
          EpilogueOutputMatmulOp::kCount>::Epilogue;
  using Epilogue1 =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename DualMma::Operator, kPartitionsK, EpilogueOutputMatmulOp,
          EpilogueOutputMatmulOp::kCount>::Epilogue;

  using OutputOp0 = typename Epilogue0::OutputOp;
  using OutputOp1 = typename Epilogue0::OutputOp;
  using OutputOp2 = EpilogueOutputOp;

  static constexpr bool kStoreD0 = StoreD;
  static constexpr bool kStoreD1 = StoreD;

  using DualEpilogue = cutlass::epilogue::threadblock::DualEpilogue<
      typename Epilogue0::Shape,
      typename Epilogue0::WarpMmaOperator,
      Epilogue0::kPartitionsK,
      typename Epilogue0::OutputTileIterator,
      typename Epilogue0::AccumulatorFragmentIterator,
      typename Epilogue0::WarpTileIterator,
      typename Epilogue0::SharedLoadIterator,
      OutputOp0,
      OutputOp1,
      OutputOp2,
      typename Epilogue0::Padding,
      kStoreD0,
      kStoreD1,
      Epilogue0::kFragmentsPerIteration,
      true // IterationsUnroll
  >;

  static_assert(!kSplitKSerial || (kStoreD0 && kStoreD1),
    "Split-K serial requires buffers for D0/D1 for reduction");

  /// Warp count (concept: GemmShape)
  using WarpCount0 = typename DualMma::WarpCount;
  static int const kThreadCount = 32 * WarpCount0::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    int split_k_slices; 

    // Mma0
    typename DualMma::IteratorA::Params params_A0;
    typename DualMma::IteratorA::TensorRef ref_A0;
    typename DualMma::IteratorB::Params params_B0;
    typename DualMma::IteratorB::TensorRef ref_B0;
    typename Epilogue0::OutputTileIterator::Params params_C0;
    typename Epilogue0::OutputTileIterator::TensorRef ref_C0;
    typename Epilogue0::OutputTileIterator::Params params_D0;
    typename Epilogue0::OutputTileIterator::TensorRef ref_D0;
    typename OutputOp0::Params output_op_0;

    // Mma1
    typename DualMma::IteratorB::Params params_B1;
    typename DualMma::IteratorB::TensorRef ref_B1;
    typename Epilogue1::OutputTileIterator::Params params_C1;
    typename Epilogue1::OutputTileIterator::TensorRef ref_C1;
    typename Epilogue1::OutputTileIterator::Params params_D1;
    typename Epilogue1::OutputTileIterator::TensorRef ref_D1;
    typename OutputOp1::Params output_op_1;

    typename Epilogue1::OutputTileIterator::Params params_D2;
    typename Epilogue1::OutputTileIterator::TensorRef ref_D2;
    typename OutputOp2::Params output_op_2;

    int *semaphore;
    int gemm_k_size;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      int split_k_slices, 
      // Mma0: D0 = A @ B0 + C0
      typename DualMma::IteratorA::TensorRef ref_A0,
      typename DualMma::IteratorB::TensorRef ref_B0,
      typename Epilogue0::OutputTileIterator::TensorRef ref_C0,
      typename Epilogue0::OutputTileIterator::TensorRef ref_D0,
      // Mma1: D1 = A @ B1 + C1
      typename DualMma::IteratorB::TensorRef ref_B1,
      typename Epilogue1::OutputTileIterator::TensorRef ref_C1,
      typename Epilogue1::OutputTileIterator::TensorRef ref_D1,

      typename Epilogue1::OutputTileIterator::TensorRef ref_D2,
      typename OutputOp0::Params output_op_0 = typename OutputOp0::Params(),
      typename OutputOp1::Params output_op_1 = typename OutputOp1::Params(),
      typename OutputOp2::Params output_op_2 = typename OutputOp2::Params(),
      int *workspace = nullptr
    ):
      problem_size(problem_size),
      split_k_slices(split_k_slices), 
      // Mma0
      params_A0(ref_A0.layout()),
      ref_A0(ref_A0),
      params_B0(ref_B0.layout()),
      ref_B0(ref_B0),
      params_C0(ref_C0.layout()),
      ref_C0(ref_C0),
      params_D0(ref_D0.layout()),
      ref_D0(ref_D0),
      // Mma1
      params_B1(ref_B1.layout()),
      ref_B1(ref_B1),
      params_C1(ref_C1.layout()),
      ref_C1(ref_C1),
      params_D1(ref_D1.layout()),
      ref_D1(ref_D1),
      params_D2(ref_D2.layout()),
      ref_D2(ref_D2),
      output_op_0(output_op_0),
      output_op_1(output_op_1),
      output_op_2(output_op_2) {
      ThreadblockSwizzle threadblock_swizzle;

      cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
                                                problem_size, 
                                                {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
                                                split_k_slices);
      grid_tiled_shape = grid_shape; 
      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape); 
      int total_gemm_k_iterations = (problem_size.k() + DualMma::Shape::kK - 1) / DualMma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      gemm_k_size = gemm_k_iterations * DualMma::Shape::kK;
      semaphore = workspace;
    }

    CUTLASS_HOST_DEVICE void set_workspace_ptr(int *workspace){
      semaphore = workspace;
    }

    __host__ dim3 getBlocksGrid() const {
      ThreadblockSwizzle threadblock_swizzle;
      dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
      return threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    }

    __host__ dim3 getThreadsGrid() const {
      return dim3(kThreadCount, 1, 1);
    }

    /// Gets the workspace size
    __host__ size_t get_workspace_size() {

      size_t bytes = 0;
        
      // Determine grid shape
      ThreadblockSwizzle threadblock_swizzle;

      cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size, 
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        split_k_slices);

      if (kSplitKSerial && split_k_slices > 1) {
        bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
      }
      return bytes;
    }

  };

  /// Shared memory storage structure
  union SharedStorage {
    typename DualMma::SharedStorage main_loop;
    typename DualEpilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  DualGemm() { }

  /// Determines whether kernel satisfies alignment
    static Status can_implement(
      cutlass::gemm::GemmCoord const & problem_size,
      typename DualMma::IteratorA::TensorRef ref_A0,
      typename DualMma::IteratorB::TensorRef ref_B0,
      typename Epilogue0::OutputTileIterator::TensorRef ref_C0,
      typename Epilogue0::OutputTileIterator::TensorRef ref_D0,
      typename DualMma::IteratorB::TensorRef ref_B1,
      typename Epilogue1::OutputTileIterator::TensorRef ref_C1,
      typename Epilogue1::OutputTileIterator::TensorRef ref_D1,
      typename Epilogue1::OutputTileIterator::TensorRef ref_D2) {

    static int const kAlignmentA = DualMma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = DualMma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue0::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A0, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B0, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C0, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D0, kAlignmentC)) {
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

    if (!TensorRef_aligned(ref_D2, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params) {
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    // Dynamic shared memory base pointer
    extern __shared__ int smem_buffer[];
    // Declare pointer to dynamic shared memory.
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A0{
      threadblock_tile_offset.m() * DualMma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B0{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * DualMma::Shape::kN
    };

    cutlass::MatrixCoord tb_offset_B1{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * DualMma::Shape::kN
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k =
      (params.problem_size.k() < (threadblock_tile_offset.k() + 1) * params.gemm_k_size) ?
       params.problem_size.k() :
       (threadblock_tile_offset.k() + 1) * params.gemm_k_size;

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A0.column() + DualMma::Shape::kK - 1) / DualMma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename DualMma::IteratorA iterator_A0(
      params.params_A0,
      params.ref_A0.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A0);

    typename DualMma::IteratorB iterator_B0(
      params.params_B0,
      params.ref_B0.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B0);

    typename DualMma::IteratorB iterator_B1(
      params.params_B1,
      params.ref_B1.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B1);


    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0x1f, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //


    // Construct thread-scoped matrix multiply
    typename DualMma::FragmentC accum0;
    typename DualMma::FragmentC accum1;
    accum0.clear();
    accum1.clear();

    DualMma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations,
        accum0, accum1,
        iterator_A0, iterator_B0, iterator_B1,
        accum0, accum1);
    }

    //
    // Epilogue
    //

    OutputOp0 output_op_0(params.output_op_0);
    OutputOp1 output_op_1(params.output_op_1);
    OutputOp2 output_op_2(params.output_op_2);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * DualMma::Shape::kM,
      threadblock_tile_offset.n() * DualMma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op_0.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
      output_op_1.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue0::OutputTileIterator iterator_C0(
      params.params_C0,
      params.ref_C0.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );
    typename Epilogue1::OutputTileIterator iterator_C1(
      params.params_C1,
      params.ref_C1.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue0::OutputTileIterator iterator_D0(
      params.params_D0,
      params.ref_D0.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );
    typename Epilogue1::OutputTileIterator iterator_D1(
      params.params_D1,
      params.ref_D1.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );
    typename Epilogue1::OutputTileIterator iterator_D2(
      params.params_D2,
      params.ref_D2.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    DualEpilogue epilogue(
      shared_storage.epilogue,
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C0 = iterator_D0;
        iterator_C1 = iterator_D1;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }

    // Execute the epilogue operator to update the destination tensor.
    typename Epilogue0::OutputTileIterator source_iters[] = {
      iterator_C0, iterator_C1
    };
    const bool writeToD2 = (!kSplitKSerial || params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1);
    epilogue(
      output_op_0, output_op_1, output_op_2,
      iterator_D0, iterator_D1, iterator_D2,
      accum0, accum1,
      source_iters,
      writeToD2
    );
    
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

template <typename Operator>
__global__ void DualKernel(typename Operator::Params params); 
