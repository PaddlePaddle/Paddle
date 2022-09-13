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
/* \file
   \brief Template for device-level Implicit GEMM Convolution
*/

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/conv/convolution.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ImplicitGemmKernel_>
class ImplicitGemmConvolution {
public:

  using ImplicitGemmKernel = ImplicitGemmKernel_;

  using ElementA = typename ImplicitGemmKernel::ElementA;
  using LayoutA = typename ImplicitGemmKernel::LayoutA;
  using ElementB = typename ImplicitGemmKernel::ElementB;
  using LayoutB = typename ImplicitGemmKernel::LayoutB;
  using ElementC = typename ImplicitGemmKernel::ElementC;
  using LayoutC = typename ImplicitGemmKernel::LayoutC;
  using ElementAccumulator = typename ImplicitGemmKernel::ElementAccumulator;
  using ElementCompute = typename ImplicitGemmKernel::ElementCompute;
  using OperatorClass = typename ImplicitGemmKernel::OperatorClass;
  using ArchTag = typename ImplicitGemmKernel::ArchTag;
  using ThreadblockShape = typename ImplicitGemmKernel::ThreadblockShape;
  using WarpShape = typename ImplicitGemmKernel::WarpShape;
  using InstructionShape = typename ImplicitGemmKernel::InstructionShape;
  using ThreadblockSwizzle = typename ImplicitGemmKernel::ThreadblockSwizzle;
  using EpilogueOutputOp = typename ImplicitGemmKernel::EpilogueOutputOp;
  static int const kStages = ImplicitGemmKernel::kStages;
  static int const kConvDim = ImplicitGemmKernel::kConvDim;
  using WarpMmaOperator = typename ImplicitGemmKernel::WarpMmaOperator;
  using ArchMmaOperator = typename ImplicitGemmKernel::ArchMmaOperator;
  using MathOperator = typename ImplicitGemmKernel::MathOperator; 

  static cutlass::conv::Operator const kConvolutionalOperator = ImplicitGemmKernel::kConvolutionalOperator;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = ImplicitGemmKernel::kIteratorAlgorithm;
  static cutlass::conv::StrideSupport const kStrideSupport = ImplicitGemmKernel::kStrideSupport;

  static int const kWarpCount = 
    (ThreadblockShape::kM / WarpShape::kM) * 
    (ThreadblockShape::kN / WarpShape::kN) *
    (ThreadblockShape::kK / WarpShape::kK);

  /// Argument structure
  using Arguments = typename ImplicitGemmKernel::Arguments;

private:

  /// Kernel parameters object
  typename ImplicitGemmKernel::Params params_;

public:

  /// Constructs Implicit GEMM
  ImplicitGemmConvolution() { }

  /// Determines whether the Implicit GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    // dispatch to iterators
    Status status = ImplicitGemmKernel::Mma::IteratorA::can_implement(args.problem_size);
    if (Status::kSuccess != status) {
      return status;
    }

    status = ImplicitGemmKernel::Mma::IteratorB::can_implement(args.problem_size);
    if (Status::kSuccess != status) {
      return status;
    }

    static int const kAlignmentC = ImplicitGemmKernel::Epilogue::OutputTileIterator::kElementsPerAccess;
    if (kConvolutionalOperator == conv::Operator::kFprop) {
      if (args.problem_size.K % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    } else if (kConvolutionalOperator == conv::Operator::kDgrad) {
       if (args.problem_size.C % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    } else if (kConvolutionalOperator == conv::Operator::kWgrad) {
       if (args.problem_size.C % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    }

    // check for unsupported problem sizes for strided dgrad implementation
    if (kConvolutionalOperator == conv::Operator::kDgrad && 
      kStrideSupport == conv::StrideSupport::kStrided) {

      // Unity stride (1x1) is supported by strided dgrad but disabled for performance 
      // reasons. For unity stride, use strided dgrad optimized unity stride specialization.
      // Note that unit tests strided dgrad for unity stride to make sure that strided 
      // dgrad implemetnation is functionaly sound. 
      // Strided dgrad implementation also support mixed strides, i.e., (1x2) and (2x1)
      if(args.problem_size.stride_h == 1 && args.problem_size.stride_w == 1) {
        return Status::kErrorNotSupported;
      }

      // split-k (serial or parallel) is not supported for strided dgrad
      if(args.problem_size.split_k_slices > 1) {
        return Status::kErrorNotSupported;
      }
      
      // dilation > {1x1} is not supported for strided dgrad
      if(args.problem_size.dilation_h > 1 || args.problem_size.dilation_w > 1) {
        return Status::kErrorNotSupported;
      }
    }

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(
      threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices));

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max())) {

      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
  
    size_t workspace_bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);

    if(args.split_k_mode == SplitKMode::kParallel) {

      // Split-K parallel: CTAs in k-dimension write the partial results in a temporary workspace.
      // The user needs to call a reduction operator to optain the final output tensor
      workspace_bytes = 
        sizeof(ElementAccumulator) *
        size_t(cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, args.problem_size)) *
        size_t(grid_tiled_shape.k());
    }

    else if(args.split_k_mode == SplitKMode::kSerial && args.problem_size.split_k_slices > 1) {

      // Split-K serial: The user workspace is used to store semaphore and serialize writing the 
      // final reduced output to user's output tensor
      workspace_bytes = sizeof(int) * size_t(grid_tiled_shape.m()) * size_t(grid_tiled_shape.n());
    }

    return workspace_bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
   
    if (args.problem_size.split_k_slices > 1) {

      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }

      cudaError_t status = cudaMemsetAsync(workspace, 0, get_workspace_size(args), stream);

      if (status != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    // initialize the params structure from the arguments
    params_ = typename ImplicitGemmKernel::Params(
    	args,
    	static_cast<int *>(workspace)
    );
    
    int smem_size = int(sizeof(typename ImplicitGemmKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(cutlass::Kernel<ImplicitGemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }
    
    return Status::kSuccess;
  }

  /// Initializes GEMM state from arguments.
  Status update(Arguments const &args, void *workspace = nullptr) {

    // update the params structure from the arguments
    params_.ptr_A = args.ref_A.data();
    params_.ptr_B = args.ref_B.data();
    params_.ptr_C = args.ref_C.data();
    params_.ptr_D = args.ref_D.data();
    params_.output_op = args.output_op;
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {


    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);

    int smem_size = int(sizeof(typename ImplicitGemmKernel::SharedStorage));

    cutlass::Kernel<ImplicitGemmKernel><<<grid, block, smem_size, stream>>>(params_);

    cudaError_t result = cudaGetLastError();

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

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
