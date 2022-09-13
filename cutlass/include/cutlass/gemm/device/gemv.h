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

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm_universal_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemvKernel_>
class Gemv {
public:

  using GemvKernel = GemvKernel_;


  using ElementA = typename GemvKernel::ElementA;
  using LayoutA  = typename GemvKernel::LayoutA;
  using ElementB = typename GemvKernel::ElementB;
  using ElementC = typename GemvKernel::ElementC;

  using ElementAccumulator = typename GemvKernel::ElementAccumulator;
  using EpilogueOutputOp = typename GemvKernel::EpilogueOutputOp;

  static ComplexTransform const kTransformA = GemvKernel::kTransformA;
  static ComplexTransform const kTransformB = GemvKernel::kTransformB;

  static int const kThreadCount = GemvKernel::kThreadCount;
  static int const kStages = GemvKernel::kStages;

  static int const kAlignmentA = GemvKernel::kAlignmentA;
  static int const kAlignmentB = GemvKernel::kAlignmentB;
  static int const kAlignmentC = GemvKernel::kAlignmentC;

  using Arguments = typename GemvKernel::Arguments;
  using Params = typename GemvKernel::Params;

private:

  Params params_;

public:

  /// Constructs the Gemv.
  Gemv() { }

  /// Determines whether the Gemv can execute the given problem.
  static Status can_implement(Arguments const &args) {

    return GemvKernel::can_implement(args);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    return 0;
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const &args) { 
    return dim3((args.problem_size.row() + (kThreadCount - 1)) / kThreadCount, 1, args.batch_count % 65565);
  }

  /// Initializes Gemv state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
    params_ = Params(args);
    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    return params_.update(args);    
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    dim3 grid = get_grid_shape(params_);
    dim3 block(GemvKernel::kThreadCount, 1, 1);

    int smem_size = int(sizeof(typename GemvKernel::SharedStorage));
    
    // Launch
    cutlass::Kernel<GemvKernel><<<grid, block, smem_size, stream>>>(params_);

    //
    // Query for errors
    //
    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
      return Status::kErrorInternal;
    }
  
    return Status::kSuccess;
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
