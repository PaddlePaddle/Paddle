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
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm_array.h"

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

/*! Gemm device-level operator. This is an interface to efficient CUTLASS GEMM kernels that may
  be invoked from host code.

  The contributions of this class are:
    
    1. At compile time, it maps data types and high-level structural parameters onto 
       specific CUTLASS components.

    2. At runtime, it maps logical arguments to GEMM problems to kernel parameters.

    3. At runtime, it launches kernels on the device.

  The intent is to provide a convenient mechanism for interacting with most plausible GEMM
  configurations for each supported architecture. Consequently, not all parameters are exposed
  to the top-level interface. Rather, sensible defaults at each level of the CUTLASS hierarchy
  are selected to tradeoff simplicity of the interface with flexibility. We expect 
  most configurations to be specified at this level. Applications with more exotic requirements 
  may construct their kernels of interest using CUTLASS components at the threadblock, warp, 
  and thread levels of abstraction.

  CUTLASS exposes computations using the functor design pattern in which objects compose some
  internal state with an overloaded function call operator. This enables decoupling of
  initialization from execution, possibly reducing overhead during steady state phases of
  application execution.

  CUTLASS device-level operators expose an Arguments structure encompassing each logical
  input to the computation. This is distinct from the kernel-level Params structure pattern
  which contains application-specific precomputed state needed by the device code.

  Example of a CUTLASS GEMM operator implementing the functionality of cuBLAS's SGEMM NN
  is as follows:

    //
    // Instantiate the CUTLASS GEMM operator.
    //

    cutlass::gemm::device::Gemm<
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor
    > gemm_op;

    //
    // Launch the GEMM operation on the device
    //

    cutlass::Status status = gemm_op({
      {m, n, k},                          // GemmCoord problem_size,
      {A, lda},                           // TensorRef<float, layout::ColumnMajor> ref_A,
      {B, ldb},                           // TensorRef<float, layout::ColumnMajor> ref_B,
      {C, ldc},                           // TensorRef<float, layout::ColumnMajor> ref_C,
      {D, ldd},                           // TensorRef<float, layout::ColumnMajor> ref_D,
      {alpha, beta}                       // EpilogueOutputOp::Params epilogue_op_params
    });


  A simplified view of the template is listed below.

    template <
      /// Element type for A matrix operand
      typename ElementA,
      
      /// Layout type for A matrix operand
      typename LayoutA,
      
      /// Element type for B matrix operand
      typename ElementB,
      
      /// Layout type for B matrix operand
      typename LayoutB,
      
      /// Element type for C and D matrix operands
      typename ElementC,
      
      /// Layout type for C and D matrix operands
      typename LayoutC,
      
      /// Element type for internal accumulation
      typename ElementAccumulator,

      /// Operator class tag
      typename OperatorClass,
      
      /// Tag indicating architecture to tune for.  This is the minimum SM that
      /// supports the intended feature. The device kernel can be built
      /// targeting any SM larger than this number.
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
      int Stages
    >
    class Gemm;
*/
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = threadblock::GemmBatchedIdentityThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator
>
class GemmArray {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  using Operator = Operator_;

  /// Define the kernel
  using DefaultGemmKernel = typename kernel::DefaultGemm<
    ElementA,
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
    kStages,
    false,
    Operator
  >::GemmKernel;

  using GemmKernel = kernel::GemmArray<typename DefaultGemmKernel::Mma, typename DefaultGemmKernel::Epilogue, ThreadblockSwizzle>;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;

    ElementA const * const *ptr_A;
    LayoutA layout_A;

    ElementB const * const *ptr_B;
    LayoutB layout_B;

    ElementC const * const *ptr_C;
    LayoutC layout_C;

    ElementC * const * ptr_D;
    LayoutC layout_D;
    
    typename EpilogueOutputOp::Params epilogue;
    int batch_count;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      ElementA const * const *ptr_A_,
      LayoutA layout_A_,
      ElementB const * const *ptr_B_,
      LayoutB layout_B_,
      ElementC const * const *ptr_C_,
      LayoutC layout_C_,
      ElementC * const * ptr_D_,
      LayoutC layout_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int batch_count_
    ):
      problem_size(problem_size_),
      ptr_A(ptr_A_),
      layout_A(layout_A_),
      ptr_B(ptr_B_),
      layout_B(layout_B_),
      ptr_C(ptr_C_),
      layout_C(layout_C_),
      ptr_D(ptr_D_),
      layout_D(layout_D_),
      epilogue(epilogue_),
      batch_count(batch_count_) { }
  };

private:

  /// Kernel parameters object
  typename GemmKernel::Params params_;

public:

  /// Constructs the GEMM.
  GemmArray() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (args.layout_A.stride(0) % kAlignmentA) {
      return Status::kErrorMisalignedOperand;
    }

    if (args.layout_B.stride(0) % kAlignmentB) {
      return Status::kErrorMisalignedOperand;
    }

    if (args.layout_C.stride(0) % kAlignmentC) {
      return Status::kErrorMisalignedOperand;
    }

    if (args.layout_D.stride(0) % kAlignmentC) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    return 0;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.batch_count);

    // Initialize the Params structure
    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ptr_A,
      args.layout_A,
      args.ptr_B,
      args.layout_B,
      args.ptr_C,
      args.layout_C,
      args.ptr_D,
      args.layout_D,
      args.epilogue,
      args.batch_count
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      args.batch_count,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK});

    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ptr_A,
      args.layout_A,
      args.ptr_B,
      args.layout_B,
      args.ptr_C,
      args.layout_C,
      args.ptr_D,
      args.layout_D,
      args.epilogue,
      args.batch_count
    };

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Parital specialization for column-major output exchanges problem size and operand.
template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Operator class tag
  typename OperatorClass_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape_,
  /// Epilogue output operator
  typename EpilogueOutputOp_,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle_,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// Access granularity of A matrix in units of elements
  int AlignmentA,
  /// Access granularity of B matrix in units of elements
  int AlignmentB,
  typename Operator_
>
class GemmArray<
  ElementA_,
  LayoutA_,
  ElementB_,
  LayoutB_,
  ElementC_,
  layout::ColumnMajor,
  ElementAccumulator_,
  OperatorClass_,
  ArchTag_,
  ThreadblockShape_,
  WarpShape_,
  InstructionShape_,
  EpilogueOutputOp_,
  ThreadblockSwizzle_,
  Stages,
  AlignmentA,
  AlignmentB,
  Operator_
> {
public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = layout::ColumnMajor;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;

  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = false;

  //
  using UnderlyingOperator = GemmArray< 
    ElementB,
    typename layout::LayoutTranspose<LayoutB>::type,
    ElementA,
    typename layout::LayoutTranspose<LayoutA>::type,
    ElementC,
    layout::RowMajor,    
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    kAlignmentB,
    kAlignmentA
  >;

  using UnderlyingArguments = typename UnderlyingOperator::Arguments;
  using GemmKernel = typename UnderlyingOperator::GemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;

    ElementA const * const *ptr_A;
    LayoutA layout_A;

    ElementB const * const *ptr_B;
    LayoutB layout_B;

    ElementC const * const *ptr_C;
    LayoutC layout_C;

    ElementC * const * ptr_D;
    LayoutC layout_D;
    
    typename EpilogueOutputOp::Params epilogue;
    int batch_count;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      ElementA const * const *ptr_A_,
      LayoutA layout_A_,
      ElementB const * const *ptr_B_,
      LayoutB layout_B_,
      ElementC const * const *ptr_C_,
      LayoutC layout_C_,
      ElementC * const * ptr_D_,
      LayoutC layout_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int batch_count_
    ):
      problem_size(problem_size_),
      ptr_A(ptr_A_),
      layout_A(layout_A_),
      ptr_B(ptr_B_),
      layout_B(layout_B_),
      ptr_C(ptr_C_),
      layout_C(layout_C_),
      ptr_D(ptr_D_),
      layout_D(layout_D_),
      epilogue(epilogue_),
      batch_count(batch_count_) { }
  };

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  GemmArray() { }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static UnderlyingArguments to_underlying_arguments(Arguments const &args) {

    GemmCoord problem_size{
      args.problem_size.n(), 
      args.problem_size.m(), 
      args.problem_size.k()
    };

    return UnderlyingArguments(
      problem_size,
      args.ptr_B,
      args.layout_B.stride(),
      args.ptr_A,
      args.layout_A.stride(),
      args.ptr_C,
      args.layout_C.stride(),
      args.ptr_D,
      args.layout_D.stride(),
      args.epilogue,
      args.batch_count
    );
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args));
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workspace);
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    return underlying_operator_.update(to_underlying_arguments(args), workspace);
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    return underlying_operator_.run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
