// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/kernel/gemm_universal_streamk.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/permute.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_ref.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/epilogue/threadblock/epilogue_tensor_op_int32.h"

namespace cutlass {
namespace gemm {
namespace kernel {
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    ///
    typename Enable = void>
struct DefaultDequantGemm;

///////////////////////////////////////////////////
/// Partial specialization for Ampere Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout>
struct DefaultDequantGemm<ElementA,
                          LayoutA,
                          kAlignmentA,
                          ElementB,
                          LayoutB,
                          kAlignmentB,
                          ElementC,
                          LayoutC,
                          ElementAccumulator,
                          arch::OpClassTensorOp,
                          arch::Sm80,
                          ThreadblockShape,
                          WarpShape,
                          InstructionShape,
                          EpilogueOutputOp,
                          ThreadblockSwizzle,
                          Stages,
                          SplitKSerial,
                          Operator,
                          SharedMemoryClear,
                          GatherA,
                          GatherB,
                          ScatterD,
                          PermuteDLayout> {
  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value ||
                    platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
                "Epilogue in the kernel level must be row major");

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma =
      typename cutlass::gemm::threadblock::DefaultMma<ElementA,
                                                      LayoutA,
                                                      kAlignmentA,
                                                      ElementB,
                                                      LayoutB,
                                                      kAlignmentB,
                                                      ElementAccumulator,
                                                      LayoutC,
                                                      arch::OpClassTensorOp,
                                                      arch::Sm80,
                                                      ThreadblockShape,
                                                      WarpShape,
                                                      InstructionShape,
                                                      Stages,
                                                      Operator,
                                                      false,
                                                      SharedMemoryClear,
                                                      GatherA,
                                                      GatherB>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using RegularEpilogue =
      typename epilogue::threadblock::DequantEpilogueTensorOp<
          ThreadblockShape,
          typename Mma::Operator,
          kPartitionsK,
          EpilogueOutputOp,
          EpilogueOutputOp::kCount,
          ScatterD,
          PermuteDLayout>::Epilogue;

  using Epilogue = RegularEpilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel =
      kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    ///
    typename Enable = void>
struct DefaultDequantGemmUniversal;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued GEMM kernels
//

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout>
struct DefaultDequantGemmUniversal<
    ElementA,
    LayoutA,
    ComplexTransform::kNone,  // transform A
    kAlignmentA,
    ElementB,
    LayoutB,
    ComplexTransform::kNone,  // transform B
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    Operator,
    SharedMemoryClear,
    GatherA,
    GatherB,
    ScatterD,
    PermuteDLayout,
    typename platform::enable_if<
        !cutlass::is_complex<ElementAccumulator>::value>::type> {
  using DefaultGemmKernel =
      typename kernel::DefaultDequantGemm<ElementA,
                                          LayoutA,
                                          kAlignmentA,
                                          ElementB,
                                          LayoutB,
                                          kAlignmentB,
                                          ElementC,
                                          LayoutC,
                                          ElementAccumulator,
                                          OperatorClass,
                                          ArchTag,
                                          ThreadblockShape,
                                          WarpShape,
                                          InstructionShape,
                                          EpilogueOutputOp,
                                          ThreadblockSwizzle,
                                          Stages,
                                          true,
                                          Operator,
                                          SharedMemoryClear,
                                          GatherA,
                                          GatherB,
                                          ScatterD,
                                          PermuteDLayout>::GemmKernel;

  /// Universal kernel without StreamkFeature member type
  template <class SwizzleT, class Enable = void>
  class SelectBase
      : public kernel::GemmUniversal<typename DefaultGemmKernel::Mma,
                                     typename DefaultGemmKernel::Epilogue,
                                     SwizzleT> {};

  /// Universal kernel with StreamkFeature member type
  template <class SwizzleT>
  class SelectBase<SwizzleT, typename SwizzleT::StreamkFeature>
      : public kernel::GemmUniversalStreamk<
            typename DefaultGemmKernel::Mma,
            typename DefaultGemmKernel::Epilogue,
            SwizzleT> {};

  /// Select kernel by ThreadblockSwizzle's support for StreamkFeature
  using GemmKernel = SelectBase<ThreadblockSwizzle>;
};

template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,           ///! Epilogue
          typename ThreadblockSwizzle_  ///! Threadblock swizzling function
          >
struct GemmWithEpilogueVisitorDequant {
 public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueVisitor = typename Epilogue::Visitor;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using TensorRefB = TensorRef<ElementB, LayoutB>;

  using ElementC = typename EpilogueVisitor::ElementOutput;
  using LayoutC = typename Epilogue::Layout;
  using TensorRefC = TensorRef<ElementC, LayoutC>;

  using ElementScale = typename EpilogueVisitor::AlphaScaleElementType;
  using LayoutScale = cutlass::layout::RowMajor;
  using TensorRefScale = TensorRef<ElementScale, LayoutScale>;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = EpilogueVisitor::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(
      128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

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

    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefC ref_C;
    TensorRefC ref_D;
    TensorRefScale ref_scale;

    typename EpilogueVisitor::Arguments epilogue_visitor;

    //
    // Methods
    //

    Arguments() : mode(GemmUniversalMode::kGemm) {}

    /// constructs an arguments structure
    Arguments(GemmUniversalMode mode_,
              GemmCoord problem_size_,
              TensorRefA ref_A_,
              TensorRefB ref_B_,
              TensorRefC ref_C_,
              TensorRefC ref_D_,
              TensorRefScale ref_scale_,
              typename EpilogueVisitor::Arguments epilogue_visitor_)
        : mode(mode_),
          problem_size(problem_size_),
          ref_A(ref_A_),
          ref_B(ref_B_),
          ref_C(ref_C_),
          ref_D(ref_D_),
          ref_scale(ref_scale_),
          epilogue_visitor(epilogue_visitor_) {}
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename EpilogueVisitor::OutputTileIterator::Params params_C;
    typename EpilogueVisitor::OutputTileIterator::Params params_D;
    typename EpilogueVisitor::ScaleTileIterator::Params params_scale;

    GemmUniversalMode mode;
    int gemm_k_size;

    void* ptr_A;
    void* ptr_B;
    ElementC* ptr_C;
    ElementC* ptr_D;
    typename EpilogueVisitor::AlphaScaleElementType* ptr_dequant_scale;

    typename EpilogueVisitor::Params epilogue_visitor;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : swizzle_log_tile(0),
          params_A(0),
          params_B(0),
          params_C(0),
          params_D(0),
          params_scale(0),
          gemm_k_size(0),
          mode(cutlass::gemm::GemmUniversalMode::kGemm),
          ptr_A(nullptr),
          ptr_B(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          ptr_dequant_scale(nullptr) {}

    explicit Params(Arguments const& args)
        : problem_size(args.problem_size),
          swizzle_log_tile(0),
          params_A(args.ref_A.layout()),
          params_B(args.ref_B.layout()),
          params_C(args.ref_C.layout()),
          params_D(args.ref_D.layout()),
          mode(args.mode),
          gemm_k_size(args.problem_size.k()),
          ptr_A(args.ref_A.data()),
          ptr_B(args.ref_B.data()),
          ptr_C(args.ref_C.data()),
          ptr_D(args.ref_D.data()),
          ptr_dequant_scale(args.ref_scale.data()),
          epilogue_visitor(args.epilogue_visitor) {
      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
          args.problem_size,
          {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
          1);

      if (args.mode == GemmUniversalMode::kGemm ||
          args.mode == GemmUniversalMode::kGemmSplitKParallel) {
        int const kAlignK =
            const_max(const_max(128 / sizeof_bits<ElementA>::value,
                                128 / sizeof_bits<ElementB>::value),
                      1);

        gemm_k_size = round_up(ceil_div(args.problem_size.k(), 1), kAlignK);

        if (gemm_k_size) {
          grid_tiled_shape.k() = ceil_div(args.problem_size.k(), gemm_k_size);
        }
      }

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;

    struct {
      typename Epilogue::SharedStorage epilogue;
      typename EpilogueVisitor::SharedStorage visitor;
    } epilogue;
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  GemmWithEpilogueVisitorDequant() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const& problem_size) {
    CUTLASS_TRACE_HOST("GemmWithEpilogueVisitorDequant::can_implement()");

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC =
        Epilogue::OutputTileIterator::kElementsPerAccess;

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

  static Status can_implement(Arguments const& args) {
    return can_implement(args.problem_size);
  }

#define SPLIT_K_ENABLED 1

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params,
                  SharedStorage& shared_storage) {  // NOLINT
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range

    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA* ptr_A = static_cast<ElementA*>(params.ptr_A);
    ElementB* ptr_B = static_cast<ElementB*>(params.ptr_B);

#if SPLIT_K_ENABLED
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
#endif

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        offset_k,
    };

    cutlass::MatrixCoord tb_offset_B{
        offset_k, threadblock_tile_offset.n() * Mma::Shape::kN};

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
        params.params_A,
        ptr_A,
        {params.problem_size.m(), problem_size_k},
        thread_idx,
        tb_offset_A);

    typename Mma::IteratorB iterator_B(
        params.params_B,
        ptr_B,
        {problem_size_k, params.problem_size.n()},
        thread_idx,
        tb_offset_B);

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

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations =
        (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

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

    //
    // Construct the epilogue visitor
    //

    EpilogueVisitor epilogue_visitor(params.epilogue_visitor,
                                     shared_storage.epilogue.visitor,
                                     params.problem_size.mn(),
                                     thread_idx,
                                     warp_idx,
                                     lane_idx,
                                     params.params_C,
                                     params.params_D,
                                     params.params_scale,
                                     params.ptr_C,
                                     params.ptr_D,
                                     params.ptr_dequant_scale,
                                     threadblock_offset,
                                     blockIdx.y * params.problem_size.m());

    if (params.mode == GemmUniversalMode::kGemm) {
      // Indicate which position in a serial reduction the output operator is
      // currently updating
      epilogue_visitor.set_k_partition(threadblock_tile_offset.k(),
                                       params.grid_tiled_shape.k());
    } else if (params.mode == GemmUniversalMode::kBatched ||
               params.mode == GemmUniversalMode::kArray) {
      epilogue_visitor.set_batch_index(threadblock_tile_offset.k());
    }

    // Construct the epilogue
    Epilogue epilogue(
        shared_storage.epilogue.epilogue, thread_idx, warp_idx, lane_idx);

    // Execute the epilogue operator to update the destination tensor.
    epilogue(epilogue_visitor, accumulators);
  }
};
}  // namespace kernel
}  // namespace gemm

namespace epilogue {
namespace threadblock {
template <typename ThreadblockShape_,
          int ThreadCount,
          typename ScaleTileIterator_,
          typename OutputTileIterator_,
          typename ElementAccumulator_,
          typename ElementCompute_,
          typename ElementwiseFunctor_,
          bool UseMasking_ = false>
class DequantEpilogueVisitor {
 public:
  using ThreadblockShape = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using ScaleTileIterator = ScaleTileIterator_;
  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  using ElementOutput = typename OutputTileIterator::Element;
  using LayoutOutput = cutlass::layout::RowMajor;
  using ElementAccumulator = ElementAccumulator_;

  using AlphaScaleElementType = typename ScaleTileIterator::Element;

  using ElementCompute = ElementCompute_;
  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using ComputeFragment = Array<ElementCompute_, kElementsPerAccess>;
  using OutputVector = Array<ElementOutput, kElementsPerAccess>;

  static int const kThreadsPerRow =
      OutputTileIterator::ThreadMap::Detail::kAccessWidth;
  static bool const kHasMultiStepsInRow =
      (OutputTileIterator::ThreadMap::Iterations::kColumn > 1);

  /// Argument structure
  struct Arguments {
    typename ElementwiseFunctor::Params elementwise;

    //
    // Methods
    //

    explicit Arguments(typename ElementwiseFunctor::Params elementwise_)
        : elementwise(elementwise_) {}
  };

  struct Params {
    typename ElementwiseFunctor::Params elementwise;
    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    explicit Params(Arguments const& args) : elementwise(args.elementwise) {}
  };

  /// Shared storage
  struct SharedStorage {};

 private:
  Params const& params_;
  SharedStorage& shared_storage_;
  MatrixCoord extent_;
  MatrixCoord extent_real_;
  ElementwiseFunctor elementwise_;

  AlphaScaleElementType* ptr_dequant_scale_;
  ScaleTileIterator iterator_dequant_scale_;
  OutputTileIterator iterator_C_;
  OutputTileIterator iterator_D_;

  AlphaScaleElementType element_alpha_row_ = 1.0f;
  AlphaScaleElementType element_alpha_col_ = 1.0f;
  typename ScaleTileIterator::Fragment fragment_dequant_scale_;
  typename OutputTileIterator::Fragment fragment_C_;
  typename OutputTileIterator::Fragment fragment_D_;

  ElementAccumulator beta_;

  int column_offset_;

  MatrixCoord thread_offset_;

 public:
  CUTLASS_DEVICE
  DequantEpilogueVisitor(
      Params const& params,
      SharedStorage& shared_storage,  // NOLINT
      cutlass::MatrixCoord const& problem_size,
      int thread_idx,
      int warp_idx,
      int lane_idx,
      typename OutputTileIterator::Params params_C,
      typename OutputTileIterator::Params params_D,
      typename ScaleTileIterator::Params params_dequant_scale,
      typename OutputTileIterator::Element* ptr_C,
      typename OutputTileIterator::Element* ptr_D,
      AlphaScaleElementType* ptr_dequant_scale,
      cutlass::MatrixCoord const& threadblock_offset = cutlass::MatrixCoord(0,
                                                                            0),
      int column_offset = 0,
      cutlass::MatrixCoord const& problem_size_real = cutlass::MatrixCoord(0,
                                                                           0))
      : params_(params),
        shared_storage_(shared_storage),
        extent_(problem_size),
        elementwise_(params.elementwise),
        ptr_dequant_scale_(ptr_dequant_scale),
        iterator_dequant_scale_(params_dequant_scale,
                                ptr_dequant_scale,
                                problem_size,
                                thread_idx,
                                threadblock_offset),
        iterator_C_(
            params_C, ptr_C, problem_size, thread_idx, threadblock_offset),
        iterator_D_(
            params_D, ptr_D, problem_size, thread_idx, threadblock_offset),
        extent_real_(problem_size_real) {
    beta_ = (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr
                                         : params.elementwise.beta);

    if (beta_ == ElementAccumulator()) {
      iterator_C_.clear_mask();
    }
  }

  /// Helper to indicate split-K behavior
  CUTLASS_DEVICE
  void set_k_partition(
      int split_k_index,     ///< Index of this threadblock within split-K
                             ///< partitioned scheme
      int split_k_slices) {  ///< Total number of split-K slices
  }

  /// Called to set the batch index
  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {}

  /// Called at the start of the epilogue just before iterating over accumulator
  /// slices
  CUTLASS_DEVICE
  void begin_epilogue() {
    iterator_dequant_scale_.load(fragment_dequant_scale_);
  }

  /// Called at the start of one step before starting accumulator exchange
  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    fragment_D_.clear();
    fragment_C_.clear();

    iterator_C_.load(fragment_C_);
    ++iterator_C_;
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void begin_row(int row_idx) {
    // Clear accumulators for max and sum when starting a whole row
  }

  /// Called after accumulators have been exchanged for each accumulator vector
  CUTLASS_DEVICE
  void visit(int iter_idx,
             int row_idx,
             int column_idx,
             int frag_idx,
             AccumulatorFragment const& accum) {
    NumericArrayConverter<ElementCompute,
                          ElementAccumulator,
                          kElementsPerAccess>
        source_converter;

    ComputeFragment result = source_converter(accum);

    // printf("start_row:%d start_col:%d\niter_idx: %d, row_idx: %d, column_idx:
    // %d, frag_idx: %d, i: %d, item: %f\n", iterator_D_.thread_start_row(),
    // iterator_D_.thread_start_column(), iter_idx, row_idx, column_idx,
    // frag_idx, i, result[i]);

    ComputeFragment alpha_col =
        reinterpret_cast<ComputeFragment*>(&fragment_dequant_scale_)[frag_idx];
    result = per_token_channel_scale_accumulator_(
        result, alpha_col, element_alpha_row_);

    // printf("%d %e\n", accum[0], result[0]);
    // scale_accumulator_(result, alpha_row_vector[0]); //TODO(mseznec)

    // if (elementwise_.kScale ==
    // cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
    //   result = source_converter(elementwise_(result));
    // } else {
    //   result = source_converter(elementwise_(result, source_vector));
    // }

    // Convert to the output
    NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess>
        output_converter;
    OutputVector& output =
        reinterpret_cast<OutputVector*>(&fragment_D_)[frag_idx];
    output = output_converter(result);
  }

  /// Called at the end of a row
  CUTLASS_DEVICE
  void end_row(int row_idx) {
    /* using ConvertSumOutput = cutlass::NumericConverter<ElementSum,
     * ElementSoftmaxCompute>; */
    /* using ConvertNormOutput = cutlass::NumericConverter<ElementNorm,
     * ElementSoftmaxCompute>; */

    /* ConvertSumOutput   convert_sum_output; */
    /* ConvertNormOutput  convert_norm_output; */

    /* Compute accumulate sum only in the last step */
    /* accum_sum_ = warp_reduce_sum_(accum_sum_); */

    /* bool is_first_thread_in_tile = ((threadIdx.x % kThreadsPerRow) == 0); */
    /* bool row_guard = thread_offset_.row() < extent_.row(); */
    /* bool is_write_thread = row_guard && is_first_thread_in_tile; */

    /* int block_batch = blockIdx.z; */

    /* ElementNorm *curr_ptr_max = ptr_Max_ + thread_offset_.row() +
     * column_offset_ + block_batch * params_.batch_stride_Max; */
    /* ElementSum *curr_ptr_sum = ptr_Sum_ + thread_offset_.row() +
     * column_offset_ + block_batch * params_.batch_stride_Sum; */

    /* arch::global_store<ElementNorm, sizeof(ElementNorm)>( */
    /*           convert_norm_output(accum_max_), */
    /*           (void *)curr_ptr_max, */
    /*           is_write_thread); */

    /* arch::global_store<ElementSum, sizeof(ElementSum)>( */
    /*           convert_sum_output(accum_sum_), */
    /*           (void *)curr_ptr_sum, */
    /*           is_write_thread); */

    // Clear accumulators for max and sum when finishing a whole row
    /* clear_accum_(); */
  }

  /// Called after all accumulator elements have been visited
  CUTLASS_DEVICE
  void end_step(int step_idx) {
    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  /// Called after all steps have been completed
  CUTLASS_DEVICE
  void end_epilogue() {}

 private:
  CUTLASS_DEVICE
  ComputeFragment per_token_channel_scale_accumulator_(
      ComputeFragment const& accum,
      ComputeFragment const& scale_col,
      AlphaScaleElementType const& scale_row) {
    ComputeFragment result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ComputeFragment::kElements; ++i) {
      result[i] = accum[i] * (scale_col[i] * scale_row);
    }

    return result;
  }

  CUTLASS_DEVICE
  ComputeFragment per_token_scale_accumulator_(
      ComputeFragment const& accum,
      AlphaScaleElementType const& scale_col,
      AlphaScaleElementType const& scale_row) {
    ComputeFragment result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ComputeFragment::kElements; ++i) {
      result[i] = accum[i] * (scale_col * scale_row);
    }

    return result;
  }
};

/// Epilogue operator
template <typename Visitor_,  ///< Functor containing fused operations
                              ///< (satisfies EpilogueFusedVisitorConcept)
          typename Shape_,  ///< Shape of threadblock tile (concept: GemmShape)
          typename WarpMmaOperator_,  ///< Warp-level MMA operator (concept:
                                      ///< gemm::warp::MmaTensorOp)
          int PartitionsK,  ///< Number of partitions of the K dimension
          typename AccumulatorFragmentIterator_,  ///< Fragment iterator
                                                  ///< selecting accumulators
          typename WarpTileIterator_,    ///< Warp-scoped tile iterator writing
                                         ///< accumulators to SMEM
          typename SharedLoadIterator_,  ///< Threadblock-scoped tile iterator
                                         ///< loading from SMEM
          typename Padding_,  ///< Padding added to SMEM allocation to avoid
                              ///< bank conflicts (concept: MatrixShape)
          int FragmentsPerPartition =
              1,                  ///< Used to coarsten the epilogue granularity
          int IterationsUnroll =  ///< Used to reduce binary size when epilogue
                                  ///< op is large
          (true || !IsEpilogueFunctorHeavy<Visitor_>::value)>
class DequantEpilogueWithVisitor
    : public EpilogueBase<Shape_,
                          typename WarpMmaOperator_::Shape,
                          PartitionsK,
                          AccumulatorFragmentIterator_,
                          WarpTileIterator_,
                          Padding_,
                          FragmentsPerPartition> {
 public:
  using Visitor = Visitor_;

  using Base = EpilogueBase<Shape_,
                            typename WarpMmaOperator_::Shape,
                            PartitionsK,
                            AccumulatorFragmentIterator_,
                            WarpTileIterator_,
                            Padding_,
                            FragmentsPerPartition>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;

  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using Padding = Padding_;

  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename Base::AccumulatorTile;

  /// Accumulator element
  using ElementAccumulator = typename WarpTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = Visitor::kElementsPerAccess;

  /// Tensor reference to sync tensor
  using SyncTensorRef =
      typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

  /// Array type used by output functor
  using AccumulatorAccessType =
      Array<typename WarpTileIterator::Element, kElementsPerAccess>;

  /// Number of warps
  using WarpCount = typename Base::WarpCount;

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1
                                        ? Base::kFragmentsPerIteration
                                        : kPartitionsK;
  static int constexpr kSmemPointerOffset =
      Base::SharedStorage::StorageShape::kCount / kSmemTiles;

  using SharedStorage = typename Base::SharedStorage;

 private:
  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;

 public:
  /// Constructor
  CUTLASS_DEVICE
  DequantEpilogueWithVisitor(
      SharedStorage& shared_storage,  ///< Shared storage object.  // NOLINT
      int thread_idx,                 ///< ID of a thread within the threadblock
      int warp_idx,                   ///< ID of warp within threadblock
      int lane_idx                    ///< Id of thread within warp
      )
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        shared_load_iterator_(shared_storage.reference(), thread_idx) {}

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(Visitor& visitor,  // NOLINT
                  AccumulatorTile const&
                      accumulators) {  ///< Threadblock tile coordinate in GEMM
                                       ///< (in units of threadblock tiles)

    visitor.begin_epilogue();

    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

#pragma unroll(IterationsUnroll ? Visitor::kIterations : 1)
    for (int iter_idx = 0; iter_idx < Visitor::kIterations; ++iter_idx) {
      //
      // Load the source
      //

      visitor.begin_step(iter_idx);

      //
      // Convert and store fragment
      //

      __syncthreads();

      acc2smem_source_needed<cutlass::make_index_sequence<
          Visitor::kIterations>>::push(iter_idx,
                                       accum_fragment_iterator,
                                       this->warp_tile_iterator_);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment
          aligned_accum_fragment[kPartitionsK];

      shared_load_iterator_.load(aligned_accum_fragment[0]);

      // If the number of k-slices is > 1 - perform a reduction amongst the
      // k-slices if (kPartitionsK > 1) {

      //   plus <typename SharedLoadIterator::Fragment> add_fragments;

      //   CUTLASS_PRAGMA_UNROLL
      //   for ( int i = 1; i < kPartitionsK; ++i) {
      //     shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
      //     shared_load_iterator_.load(aligned_accum_fragment[i]);
      //     aligned_accum_fragment[0] =
      //     add_fragments(aligned_accum_fragment[0],
      //     aligned_accum_fragment[i]);
      //   }

      //   shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) *
      //   kSmemPointerOffset);
      // }

      //
      // Iterate over output fragments
      //

      AccumulatorAccessType const* accum_frag_ptr =
          reinterpret_cast<AccumulatorAccessType const*>(
              &aligned_accum_fragment[0]);

      int const kAccumulatorFragmentCount =
          AccumulatorTile::kElements /
          (Visitor::kIterations * AccumulatorAccessType::kElements);

      CUTLASS_PRAGMA_UNROLL
      for (int idx = 0; idx < kAccumulatorFragmentCount; ++idx) {
        int row_idx = idx / SharedLoadIterator::ThreadMap::Iterations::kColumn;
        int col_idx = idx % SharedLoadIterator::ThreadMap::Iterations::kColumn;

        // Start a new row of the output fragment
        // if (!col_idx) {
        //   visitor.begin_row(row_idx);
        // }

        visitor.visit(iter_idx, row_idx, col_idx, idx, accum_frag_ptr[idx]);

        // End the row of the output fragment
        // if (col_idx + 1 ==
        // SharedLoadIterator::ThreadMap::Iterations::kColumn) {
        //   visitor.end_row(row_idx);
        // }
      }

      //
      // Conclude the step
      //

      visitor.end_step(iter_idx);
    }

    visitor.end_epilogue();
  }

 private:
  template <class Seq>
  struct acc2smem_source_needed;

  template <size_t... Seq>
  struct acc2smem_source_needed<cutlass::index_sequence<Seq...>> {
    template <int Advance>
    CUTLASS_DEVICE static void helper(
        AccumulatorFragmentIterator accum_fragment_iterator,
        WarpTileIterator& warp_tile_iterator) {  // NOLINT
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      typename AccumulatorFragmentIterator::Fragment accum_fragment;
      accum_fragment_iterator.load(accum_fragment);
      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void push(size_t pos,
                     AccumulatorFragmentIterator const& iterator_begin,
                     WarpTileIterator& warp_tile_iterator) {  // NOLINT
      int dummy[] = {(pos == Seq) &&
                     (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
    }
  };
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper to create an EpilogueWithVisitor from an existing epilogue
template <typename Visitor_, typename Existing_, bool IterationsUnroll = true>
struct DequantEpilogueWithVisitorFromExistingEpilogue {
  using Epilogue = DequantEpilogueWithVisitor<
      Visitor_,
      typename Existing_::Shape,
      typename Existing_::WarpMmaOperator,
      Existing_::kPartitionsK,
      typename Existing_::AccumulatorFragmentIterator,
      typename Existing_::WarpTileIterator,
      typename Existing_::SharedLoadIterator,
      typename Existing_::Padding,
      Existing_::kFragmentsPerIteration,
      IterationsUnroll>;
};

}  // namespace threadblock
}  // namespace epilogue

template <typename ElementA_,
          typename LayoutA_,
          typename ElementB_,
          typename LayoutB_,
          typename ElementC_,
          typename ElementCompute_,
          typename OperatorClass_,
          typename ArchTag_,
          typename ThreadblockShape_,
          typename WarpShape_,
          typename InstructionShape_,
          typename EpilogueFunctorOp_,
          int kStages_,
          int AlignmentA_ = 128 / cutlass::sizeof_bits<ElementA_>::value,
          int AlignmentB_ = 128 / cutlass::sizeof_bits<ElementB_>::value>
class GemmDequant {
 public:
  ///////////////////////////////////////////////////////////////////////////////////////////////

  //
  // Type definitions
  //

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementCompute = ElementCompute_;

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;

  using EpilogueFunctorOp = EpilogueFunctorOp_;

  // These are mandatory layouts.
  using LayoutC = cutlass::layout::RowMajor;

  using TensorRefA = TensorRef<ElementA, LayoutA>;
  using TensorRefB = TensorRef<ElementB, LayoutB>;
  using TensorRefC = TensorRef<ElementC, LayoutC>;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;

  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;

  static int const kStages = kStages_;
  static int const AlignmentA = AlignmentA_;
  static int const AlignmentB = AlignmentB_;

  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // basic GEMM kernel
  using DefaultGemmKernel =
      typename cutlass::gemm::kernel::DefaultDequantGemmUniversal<
          // using DefaultGemmKernel = typename
          // cutlass::gemm::kernel::DefaultGemm<
          ElementA,
          LayoutA,
          cutlass::ComplexTransform::kNone,
          AlignmentA,
          ElementB,
          LayoutB,
          cutlass::ComplexTransform::kNone,
          AlignmentB,
          ElementC,
          LayoutC,
          ElementCompute,
          OperatorClass,
          ArchTag,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EpilogueFunctorOp,
          ThreadblockSwizzle,
          kStages,
          cutlass::arch::OpMultiplyAddSaturate
          // typename cutlass::gemm::device::DefaultGemmConfiguration<
          //     OperatorClass, ArchTag, ElementA, ElementB, ElementC,
          //     ElementCompute>::Operator,
          // cutlass::gemm::SharedMemoryClearOption::kNone
          >::GemmKernel;

  ///////////////////////////////////////////////////////////////////////////////////////////////
  using ElementEpilogueCompute = float;
  using ElementEpilogueAcc = int32_t;

  using DequantScaleIterator =
      cutlass::epilogue::threadblock::PredicatedTileIterator<
          cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
              typename DefaultGemmKernel::Epilogue::OutputTileIterator::
                  ThreadMap::Shape,
              typename DefaultGemmKernel::Epilogue::OutputTileIterator::
                  ThreadMap::Count,
              DefaultGemmKernel::Epilogue::OutputTileIterator::ThreadMap::
                  kThreads,
              DefaultGemmKernel::Epilogue::OutputTileIterator::
                  kElementsPerAccess,
              cutlass::sizeof_bits<ElementEpilogueCompute>::value>,
          ElementEpilogueCompute>;

  // Epilogue visitor
  using EpilogueVisitor =
      typename cutlass::epilogue::threadblock::DequantEpilogueVisitor<
          ThreadblockShape,
          DefaultGemmKernel::kThreadCount,
          DequantScaleIterator,
          typename DefaultGemmKernel::Epilogue::OutputTileIterator,
          ElementEpilogueAcc,
          ElementEpilogueCompute,
          EpilogueFunctorOp>;

  using ElementScale = typename EpilogueVisitor::AlphaScaleElementType;
  using LayoutScale = cutlass::layout::RowMajor;
  using TensorRefScale = TensorRef<ElementScale, LayoutScale>;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      DequantEpilogueWithVisitorFromExistingEpilogue<
          EpilogueVisitor,
          typename DefaultGemmKernel::Epilogue>::Epilogue;

  // GEMM
  using GemmKernel = gemm::kernel::GemmWithEpilogueVisitorDequant<
      typename DefaultGemmKernel::Mma,
      Epilogue,
      ThreadblockSwizzle>;

 public:
  /// Arguments class
  struct Arguments {
    typename GemmKernel::Arguments gemm;
    cutlass::gemm::GemmCoord extend;

    //
    // Methods
    //
    Arguments() : gemm(), extend() {}

    Arguments(cutlass::gemm::GemmCoord problem_size,
              TensorRefA ref_A_,
              TensorRefB ref_B_,
              TensorRefC ref_C_,
              TensorRefC ref_D_,
              TensorRefScale ref_scale_,
              typename EpilogueFunctorOp::Params linear_scaling)
        : gemm(cutlass::gemm::GemmUniversalMode::kGemm,
               problem_size,
               ref_A_,
               ref_B_,
               ref_C_,
               ref_D_,
               ref_scale_,
               typename EpilogueVisitor::Arguments(linear_scaling)),
          extend(problem_size) {}
  };

  struct Params {
    typename GemmKernel::Params gemm;
    MatrixCoord extend;
    //
    // Methods
    //
    Params() {}

    explicit Params(Arguments const& args)
        : gemm(args.gemm),
          extend(MatrixCoord(args.extend.m(), args.extend.n())) {}
  };

 public:
  // Gemm

  //
  // Methods
  //

 private:
  Params params_;

 public:
  /// Ctor
  GemmDequant() {}

  /// Initialize
  Status initialize(Arguments const& args) {
    params_ = Params(args);

    return cutlass::Status::kSuccess;
  }

  /// Run
  Status run(cudaStream_t stream) {
    //
    // Launch the GEMM + max kernel
    //

    dim3 gemm_grid =
        ThreadblockSwizzle().get_grid_shape(params_.gemm.grid_tiled_shape);
    dim3 gemm_block(GemmKernel::kThreadCount, 1, 1);

    int gemm_smem_size =
        static_cast<int>(sizeof(typename GemmKernel::SharedStorage));

    cudaError_t result;

    if (gemm_smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    gemm_smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel>
        <<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      std::cerr << "gemm_with_dequant kernel error: "
                << cudaGetErrorString(result) << std::endl;
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  /// Function call operator
  Status operator()(cudaStream_t stream = nullptr) { return run(stream); }
};

}  // namespace cutlass
