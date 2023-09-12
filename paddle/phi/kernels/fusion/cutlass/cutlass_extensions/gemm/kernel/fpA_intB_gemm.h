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
          typename KernelArch,  ///! The Architecture this kernel is compiled
                                /// for. Used since SIMT kernels lose top-level
                                /// arch.
          bool SplitKSerial     ///! If true, code supporting split-K via serial
                                /// reduction is enabled.
          >
struct GemmFpAIntB {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Element;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Mma::LayoutC;
  using ElementScale = float;

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

  /// Parameters structure
  struct Arguments : UniversalArgumentsBase {
    cutlass::gemm::GemmCoord problem_size;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Mma::IteratorScale::TensorRef ref_scale;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;

    typename EpilogueOutputOp::Params output_op;

    // For gather+scatter operations
    int const* gather_A_indices;
    int const* gather_B_indices;
    int const* scatter_D_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Arguments() {}

    CUTLASS_HOST_DEVICE
    Arguments(cutlass::gemm::GemmCoord const& problem_size,
              typename Mma::IteratorA::TensorRef ref_A,
              typename Mma::IteratorB::TensorRef ref_B,
              typename Mma::IteratorScale::TensorRef ref_scale,
              typename Epilogue::OutputTileIterator::TensorRef ref_C,
              typename Epilogue::OutputTileIterator::TensorRef ref_D,
              int serial_split_k_factor,
              typename EpilogueOutputOp::Params output_op =
                  typename EpilogueOutputOp::Params(),
              int const* gather_A_indices = nullptr,
              int const* gather_B_indices = nullptr,
              int const* scatter_D_indices = nullptr)
        :  // TODO(wangbojun) hard code here for GemmUniversalMode::kGemm and
           // batch_stride_D
          UniversalArgumentsBase(
              GemmUniversalMode::kGemm,
              problem_size,
              /*serial_split_k_factor=*/serial_split_k_factor,
              /*batch_stride_D=*/0),
          ref_A(ref_A),
          ref_B(ref_B),
          ref_scale(ref_scale),
          ref_C(ref_C),
          ref_D(ref_D),
          output_op(output_op),
          gather_A_indices(gather_A_indices),
          gather_B_indices(gather_B_indices),
          scatter_D_indices(scatter_D_indices) {}
  };

  /// Parameters structure
  struct Params : UniversalParamsBase<ThreadblockSwizzle,
                                      ThreadblockShape,
                                      ElementA,
                                      ElementB,
                                      ElementC> {
    using ParamsBase = UniversalParamsBase<ThreadblockSwizzle,
                                           ThreadblockShape,
                                           ElementA,
                                           ElementB,
                                           ElementC>;

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
    int const* gather_A_indices;
    int const* gather_B_indices;
    int const* scatter_D_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Arguments const& args, int device_sms, int sm_occupancy)
        : ParamsBase(args, device_sms, sm_occupancy),
          params_A(args.ref_A.layout()),
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
          scatter_D_indices(args.scatter_D_indices) {}
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  GemmFpAIntB() {}

  /// Determines whether kernel satisfies alignment
  CUTLASS_HOST_DEVICE
  static Status can_implement(Arguments const& args) {
    static int const kAlignmentA =
        (platform::is_same<typename Mma::IteratorA::Layout,
                           layout::ColumnMajorInterleaved<32>>::value)
            ? 32
        : (platform::is_same<typename Mma::IteratorA::Layout,
                             layout::ColumnMajorInterleaved<64>>::value)
            ? 64
            : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =
        (platform::is_same<typename Mma::IteratorB::Layout,
                           layout::RowMajorInterleaved<32>>::value)
            ? 32
        : (platform::is_same<typename Mma::IteratorB::Layout,
                             layout::RowMajorInterleaved<64>>::value)
            ? 64
            : Mma::IteratorB::AccessType::kElements;

    static int const kAlignmentScale =
        Mma::IteratorScale::AccessType::kElements;

    static int const kAlignmentC =
        (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                           layout::ColumnMajorInterleaved<32>>::value)
            ? 32
        : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                             layout::ColumnMajorInterleaved<64>>::value)
            ? 64
            : Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(args.ref_A, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_B, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_scale, kAlignmentScale)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(args.ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  static size_t get_extra_workspace_size(
      Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape) {
    return 0;
  }

  CUTLASS_DEVICE
  static void invoke(Params const& params,
                     SharedStorage& shared_storage) {  // NOLINT
    GemmFpAIntB op;
    op(params, shared_storage);
  }

  // The dummy template parameter is not used and exists so that we can compile
  // this code using a standard earlier than C++17. Prior to C++17, fully
  // specialized templates HAD to exists in a namespace
  template <bool B, typename dummy = void>
  struct KernelRunner {
    CUTLASS_DEVICE
    static void run_kernel(Params const& params,
                           SharedStorage& shared_storage) {  // NOLINT
      CUTLASS_NOT_IMPLEMENTED();
    }
  };

  template <typename dummy>
  struct KernelRunner<true, dummy> {
    CUTLASS_DEVICE
    static void run_kernel(Params const& params,
                           SharedStorage& shared_storage) {  // NOLINT
      using LayoutB = typename Mma::IteratorB::Layout;
      static_assert(
          platform::is_same<LayoutB, layout::RowMajor>::value &&
                  kInterleave == 1 ||
              platform::is_same<LayoutB, layout::ColumnMajor>::value &&
                  kInterleave >= 1,
          "B must be row major/col major OR col major interleaved.");

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
      cutlass::MatrixCoord tb_offset_A{
          threadblock_tile_offset.m() * Mma::Shape::kM,
          threadblock_tile_offset.k() * params.gemm_k_size,
      };

      cutlass::MatrixCoord tb_offset_B{
          threadblock_tile_offset.k() * params.gemm_k_size * kInterleave,
          threadblock_tile_offset.n() * Mma::Shape::kN / kInterleave};

      cutlass::MatrixCoord tb_offset_scale{
          0, threadblock_tile_offset.n() * Mma::Shape::kN};

      // Problem size is a function of threadblock index in the K dimension
      int problem_size_k =
          min(params.problem_size.k(),
              (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations =
          (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) /
          Mma::Shape::kK;

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
          params.params_A,
          params.ref_A.data(),
          {params.problem_size.m(), problem_size_k},
          thread_idx,
          tb_offset_A,
          params.gather_A_indices);

      typename Mma::IteratorB iterator_B(
          params.params_B,
          params.ref_B.data(),
          {problem_size_k * kInterleave, params.problem_size.n() / kInterleave},
          thread_idx,
          tb_offset_B,
          params.gather_B_indices);

      typename Mma::IteratorScale iterator_scale(params.params_scale,
                                                 params.ref_scale.data(),
                                                 {1, params.problem_size.n()},
                                                 thread_idx,
                                                 tb_offset_scale);

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
      int lane_idx = threadIdx.x % 32;

      //
      // Main loop
      //
      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      typename Mma::FragmentC accumulators;

      accumulators.clear();

      if (!kSplitKSerial || gemm_k_iterations > 0) {
        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations,
            accumulators,
            iterator_A,
            iterator_B,
            iterator_scale,
            accumulators);
      }

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      //
      // Masked tile iterators constructed from members
      //

      threadblock_tile_offset =
          threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

      // assume identity swizzle
      MatrixCoord threadblock_offset(
          threadblock_tile_offset.m() * Mma::Shape::kM,
          threadblock_tile_offset.n() * Mma::Shape::kN);

      int block_idx = threadblock_tile_offset.m() +
                      threadblock_tile_offset.n() * params.grid_tiled_shape.m();

      // Construct the semaphore.
      Semaphore semaphore(params.semaphore + block_idx, thread_idx);

      // If performing a reduction via split-K, fetch the initial
      // synchronization
      if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is
        // currently updating
        output_op.set_k_partition(threadblock_tile_offset.k(),
                                  params.grid_tiled_shape.k());
      }

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(
          params.params_C,
          params.ref_C.data(),
          params.problem_size.mn(),
          thread_idx,
          threadblock_offset,
          params.scatter_D_indices);

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(
          params.params_D,
          params.ref_D.data(),
          params.problem_size.mn(),
          thread_idx,
          threadblock_offset,
          params.scatter_D_indices);

      Epilogue epilogue(
          shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

      // Wait on the semaphore - this latency may have been covered by iterator
      // construction
      if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        // For subsequent threadblocks, the source matrix is held in the 'D'
        // tensor.
        if (threadblock_tile_offset.k()) {
          iterator_C = iterator_D;
        }

        semaphore.wait(threadblock_tile_offset.k());
      }

      // Execute the epilogue operator to update the destination tensor.
      epilogue(output_op, iterator_D, accumulators, iterator_C);

      //
      // Release the semaphore
      //

      if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        int lock = 0;
        if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {
          // The final threadblock resets the semaphore for subsequent grids.
          lock = 0;
        } else {
          // Otherwise, the semaphore is incremented
          lock = threadblock_tile_offset.k() + 1;
        }

        semaphore.release(lock);
      }
    }
  };

  /*
      To improve compilation speed, we do not compile the device operator if the
     CUDA_ARCH does not correspond to the ArchTag of the cutlass kernel
     operator.
    */
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params,
                  SharedStorage& shared_storage) {  // NOLINT
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
    static constexpr bool compile_needed =
        platform::is_same<KernelArch, arch::Sm70>::value;
    KernelRunner<compile_needed>::run_kernel(params, shared_storage);

#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && (__CUDA_ARCH__ < 800)
    // static constexpr bool compile_needed = platform::is_same<KernelArch,
    // arch::Sm75>::value; KernelRunner<compile_needed>::run_kernel(params,
    // shared_storage);
    CUTLASS_NOT_IMPLEMENTED();
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 900)
    static constexpr bool compile_needed =
        platform::is_same<KernelArch, arch::Sm80>::value;
    KernelRunner<compile_needed>::run_kernel(params, shared_storage);
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
