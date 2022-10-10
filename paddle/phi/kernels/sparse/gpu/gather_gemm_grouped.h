/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

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

#include "cutlass/gemm/kernel/gemm_grouped.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          bool Transposed = false>
struct GatherGemmGrouped {
 public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kTransposed = Transposed;

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
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;

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

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //

    GemmCoord *problem_sizes;
    int problem_count;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    ElementA **ptr_A;
    ElementB **ptr_B;
    ElementC **ptr_C;
    ElementC **ptr_D;

    typename LayoutA::Stride::LongIndex *lda;
    typename LayoutB::Stride::LongIndex *ldb;
    typename LayoutC::Stride::LongIndex *ldc;
    typename LayoutC::Stride::LongIndex *ldd;

    int const **ptr_gather_A_indices;

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
          ptr_C(nullptr),
          ptr_D(nullptr),
          lda(nullptr),
          ldb(nullptr),
          ldc(nullptr),
          ldd(nullptr),
          ptr_gather_A_indices(nullptr) {}

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(GemmCoord *problem_sizes,
              int problem_count,
              int threadblock_count,
              typename EpilogueOutputOp::Params output_op,
              ElementA **ptr_A,
              ElementB **ptr_B,
              ElementC **ptr_C,
              ElementC **ptr_D,
              typename LayoutA::Stride::LongIndex *lda,
              typename LayoutB::Stride::LongIndex *ldb,
              typename LayoutC::Stride::LongIndex *ldc,
              typename LayoutC::Stride::LongIndex *ldd,
              int const **ptr_gather_A_indices = nullptr)
        : problem_sizes(problem_sizes),
          problem_count(problem_count),
          threadblock_count(threadblock_count),
          output_op(output_op),
          ptr_A(ptr_A),
          ptr_B(ptr_B),
          ptr_C(ptr_C),
          ptr_D(ptr_D),
          lda(lda),
          ldb(ldb),
          ldc(ldc),
          ldd(ldd),
          ptr_gather_A_indices(ptr_gather_A_indices) {}
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {
    typename GemmGroupedProblemVisitor<kTransposed>::Params problem_visitor;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    ElementA **ptr_A;
    ElementB **ptr_B;
    ElementC **ptr_C;
    ElementC **ptr_D;

    typename LayoutA::Stride::LongIndex *lda;
    typename LayoutB::Stride::LongIndex *ldb;
    typename LayoutC::Stride::LongIndex *ldc;
    typename LayoutC::Stride::LongIndex *ldd;

    int **ptr_gather_A_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : ptr_A(nullptr),
          ptr_B(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          lda(nullptr),
          ldb(nullptr),
          ldc(nullptr),
          ldd(nullptr),
          ptr_gather_A_indices(nullptr) {}

    CUTLASS_HOST_DEVICE
    explicit Params(Arguments const &args, void *workspace = nullptr)
        : problem_visitor(args.problem_sizes, args.problem_count),
          threadblock_count(args.threadblock_count),
          output_op(args.output_op),
          ptr_A(args.ptr_A),
          ptr_B(args.ptr_B),
          ptr_C(args.ptr_C),
          ptr_D(args.ptr_D),
          lda(args.lda),
          ldb(args.ldb),
          ldc(args.ldc),
          ldd(args.ldd),
          ptr_gather_A_indices(const_cast<int **>(args.ptr_gather_A_indices)) {}

    CUTLASS_HOST_DEVICE
    void update(Arguments const &args, void *workspace = nullptr) {
      problem_visitor = typename GemmGroupedProblemVisitor<kTransposed>::Params(
          args.problem_sizes, args.problem_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B = args.ptr_B;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      lda = args.lda;
      ldb = args.ldb;
      ldc = args.ldc;
      ldd = args.ldd;
      ptr_gather_A_indices = const_cast<int **>(args.ptr_gather_A_indices);
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename GemmGroupedProblemVisitor<kTransposed>::SharedStorage
        problem_visitor;
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  GatherGemmGrouped() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const &problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  static size_t get_extra_workspace_size(
      Arguments const &args, cutlass::gemm::GemmCoord const &grid_tiled_shape) {
    return 0;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
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

    //
    // Problem visitor.
    //
    GemmGroupedProblemVisitor<kTransposed> problem_visitor(
        params.problem_visitor,
        shared_storage.problem_visitor,
        {Mma::Shape::kM, Mma::Shape::kN},
        blockIdx.x);

    // Outer 'persistent' loop to iterate over tiles
    while (problem_visitor.next_tile()) {
      GemmCoord problem_size = problem_visitor.problem_size();
      int32_t problem_idx = problem_visitor.problem_index();
      int32_t cta_idx = int32_t(problem_visitor.threadblock_index());

      GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

      cutlass::gemm::GemmCoord threadblock_offset(
          static_cast<int>(cta_idx / grid_shape.n()) * Mma::Shape::kM,
          static_cast<int>(cta_idx % grid_shape.n()) * Mma::Shape::kN,
          0);

      // Load element pointers. Exchange pointers and strides if working on the
      // transpose
      ElementA *ptr_A = reinterpret_cast<ElementA *>((
          kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]));
      typename LayoutA::LongIndex ldm_A =
          (kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx]);

      ElementB *ptr_B = reinterpret_cast<ElementB *>((
          kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]));
      typename LayoutB::LongIndex ldm_B =
          (kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx]);

      int const *ptr_gather_A_indices =
          params.ptr_gather_A_indices[problem_idx];

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
          threadblock_offset.m(),
          0,
      };

      cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n()};

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(LayoutA(ldm_A),
                                         ptr_A,
                                         {problem_size.m(), problem_size.k()},
                                         thread_idx,
                                         tb_offset_A,
                                         ptr_gather_A_indices);

      typename Mma::IteratorB iterator_B(LayoutB(ldm_B),
                                         ptr_B,
                                         {problem_size.k(), problem_size.n()},
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
      mma(gemm_k_iterations,
          accumulators,
          iterator_A,
          iterator_B,
          accumulators);

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      ElementC *ptr_C = params.ptr_C[problem_idx];
      ElementC *ptr_D = params.ptr_D[problem_idx];

      LayoutC layout_C(params.ldc[problem_idx]);
      LayoutC layout_D(params.ldd[problem_idx]);

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
