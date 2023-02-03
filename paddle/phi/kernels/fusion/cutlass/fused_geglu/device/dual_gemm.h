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
    \brief Performs a dual gemm in one fused kernel:
```
D0 = epilogue0(X @ B0, C0)
D1 = epilogue1(X @ B1, C1)
D2 = element_wise(D0, D1)
```
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

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
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "../kernel/dual_gemm.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

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
    typename ElementAccumulator_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Epilogue output operator
    typename EpilogueOutputOp0_,
    typename EpilogueOutputOp1_,
    typename EpilogueOutputOp2_,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    bool StoreD0 = true,
    bool StoreD1 = true,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
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
        ElementAccumulator_>::Operator>
class DualGemm {
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
  using EpilogueOutputOp0 = EpilogueOutputOp0_;
  using EpilogueOutputOp1 = EpilogueOutputOp1_;
  using EpilogueOutputOp2 = EpilogueOutputOp2_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp1::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static bool constexpr kStoreD0 = StoreD0;
  static bool constexpr kStoreD1 = StoreD1;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  using LayoutScaleBias = layout::RowMajor;
  /// Define the kernel
  /// Define the threadblock-scoped matrix multiply-accumulate
  static_assert(ArchTag::kMinComputeCapability >= 80, "Only multistage is implemented");
  static_assert(kStages >= 3, "Only multistage is implemented");

  using GemmType = gemm_kernel_utils::DefaultGemmType<ArchTag, ElementA>;
  using OpClass = typename GemmType::OpClass;
  using DefaultConfig =
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OpClass,
        ArchTag,
        ElementA,
        ElementB,
        ElementC, // ElementC
        ElementC // ElementAccumulator
        >;

  // using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
  using InstructionShape = typename GemmType::InstructionShape;

  using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        ElementA, // ElementA,
        LayoutA, // LayoutA,
        kAlignmentA,
        ElementB, // ElementB,
        LayoutB, // LayoutB,
        kAlignmentB,
        ElementC,
        cutlass::layout::RowMajor, // LayoutC,
        ElementC,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

  using DualMmaFromSmem =
        typename threadblock::DualMmaFromSharedMemory<
            typename DefaultGemm::Mma>; 
  using DualMma = typename DualMmaFromSmem::Mma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue0 =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename DualMma::Operator, kPartitionsK, EpilogueOutputOp0,
          EpilogueOutputOp0::kCount>::Epilogue;
  using Epilogue1 =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename DualMma::Operator, kPartitionsK, EpilogueOutputOp1,
          EpilogueOutputOp1::kCount>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using DualGemmKernel = kernel::DualGemm<
    DualMma,
    Epilogue0, Epilogue1, EpilogueOutputOp2,
    ThreadblockSwizzle, kSplitKSerial,
    kStoreD0, kStoreD1>;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A0;
    TensorRef<ElementB const, LayoutB> ref_B0;
    TensorRef<ElementC const, LayoutC> ref_C0;
    TensorRef<ElementC, LayoutC> ref_D0;
    TensorRef<ElementB const, LayoutB> ref_B1;
    TensorRef<ElementC const, LayoutC> ref_C1;
    TensorRef<ElementC, LayoutC> ref_D1;
    TensorRef<ElementC, LayoutC> ref_D2;
    typename EpilogueOutputOp0::Params epilogue0;
    typename EpilogueOutputOp1::Params epilogue1;
    typename EpilogueOutputOp2::Params epilogue2;
    int split_k_slices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0), split_k_slices(1) {

    }

    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A0_,
      TensorRef<ElementB const, LayoutB> ref_B0_,
      TensorRef<ElementC const, LayoutC> ref_C0_,
      TensorRef<ElementC, LayoutC> ref_D0_,
      TensorRef<ElementB const, LayoutB> ref_B1_,
      TensorRef<ElementC const, LayoutC> ref_C1_,
      TensorRef<ElementC, LayoutC> ref_D1_,
      TensorRef<ElementC, LayoutC> ref_D2_,
      typename EpilogueOutputOp0::Params epilogue0_ =
        typename EpilogueOutputOp0::Params(),
      typename EpilogueOutputOp1::Params epilogue1_ =
        typename EpilogueOutputOp1::Params(),
      typename EpilogueOutputOp2::Params epilogue2_ =
        typename EpilogueOutputOp2::Params(),
      int split_k_slices_ = 1
    ):
      problem_size(problem_size_),
      ref_A0(ref_A0_),
      ref_B0(ref_B0_),
      ref_C0(ref_C0_),
      ref_D0(ref_D0_),
      ref_B1(ref_B1_),
      ref_C1(ref_C1_),
      ref_D1(ref_D1_),
      ref_D2(ref_D2_),
      epilogue0(epilogue0_),
      epilogue1(epilogue1_),
      epilogue2(epilogue2_),
      split_k_slices(split_k_slices_) {

    }
  };

private:

  /// Kernel parameters object
  typename DualGemmKernel::Params params_;

public:

  /// Constructs the GEMM.
  DualGemm() = default;

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }
    if (kStoreD0 != (args.ref_D0.data() != nullptr)) {
      return Status::kErrorInternal;
    }
    if (kStoreD1 != (args.ref_D1.data() != nullptr)) {
      return Status::kErrorInternal;
    }

    Status status = DualGemmKernel::can_implement(
      args.problem_size,
      args.ref_A0.non_const_ref(),
      args.ref_B0.non_const_ref(),
      args.ref_C0.non_const_ref(),
      args.ref_D0,
      args.ref_B1.non_const_ref(),
      args.ref_C1.non_const_ref(),
      args.ref_D1,
      args.ref_D2
    );

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {

    size_t bytes = 0;
      
    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size, 
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (kSplitKSerial && args.split_k_slices > 1) {


      bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
    }

    return bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size, 
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (kSplitKSerial) {
      if (args.split_k_slices > 1) {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }

        size_t bytes = get_workspace_size(args);
      
        cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
      }
    }
    else {

      if (args.split_k_slices > 1) {
        return Status::kErrorInvalidProblem;
      }
    }

    // Initialize the Params structure
    params_ = typename DualGemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A0.non_const_ref(),
      args.ref_B0.non_const_ref(),
      args.ref_C0.non_const_ref(),
      args.ref_D0,
      args.ref_B1.non_const_ref(),
      args.ref_C1.non_const_ref(),
      args.ref_D1,
      args.ref_D2,
      args.epilogue0,
      args.epilogue1,
      args.epilogue2,
      reinterpret_cast<int *>(workspace),
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    
    if (kSplitKSerial && args.split_k_slices > 1) {  
      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }
    }

    params_.ref_A0.reset(args.ref_A0.non_const_ref().data());
    params_.ref_B0.reset(args.ref_B0.non_const_ref().data());
    params_.ref_C0.reset(args.ref_C0.non_const_ref().data());
    params_.ref_D0.reset(args.ref_D0.data());
    params_.ref_B1.reset(args.ref_B1.non_const_ref().data());
    params_.ref_C1.reset(args.ref_C1.non_const_ref().data());
    params_.ref_D1.reset(args.ref_D1.data());
    params_.ref_D2.reset(args.ref_D2.data());
    params_.output_op_0 = args.epilogue0;
    params_.output_op_1 = args.epilogue1;
    params_.output_op_2 = args.epilogue2;
    params_.semaphore = reinterpret_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(DualGemmKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename DualGemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<DualGemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<DualGemmKernel><<<grid, block, smem_size, stream>>>(params_);

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
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
