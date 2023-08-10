/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
  \file
  \brief Base device-level grouped kernel.
*/

#pragma once

#include <limits>
#include <numeric>
#include <vector>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "cutlass/trace.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GEMM Grouped
template <typename BaseKernel_>
class BaseGrouped {
 public:
  using BaseKernel = BaseKernel_;

  using ElementA = typename BaseKernel::ElementA;
  using LayoutA = typename BaseKernel::LayoutA;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  static ComplexTransform const kTransformA = BaseKernel::kTransformA;
  static int const kAlignmentA = BaseKernel::kAlignmentA;

  using ElementB = typename BaseKernel::ElementB;
  using LayoutB = typename BaseKernel::LayoutB;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  static ComplexTransform const kTransformB = BaseKernel::kTransformB;
  static int const kAlignmentB = BaseKernel::kAlignmentB;

  using ElementC = typename BaseKernel::ElementC;
  using LayoutC = typename BaseKernel::LayoutC;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  static int const kAlignmentC = BaseKernel::kAlignmentC;

  using ElementAccumulator =
      typename BaseKernel::Mma::Policy::Operator::ElementC;

  using EpilogueOutputOp = typename BaseKernel::EpilogueOutputOp;
  using ThreadblockSwizzle = typename BaseKernel::ThreadblockSwizzle;

  using Operator = typename BaseKernel::Operator;
  using WarpMmaOperator = typename BaseKernel::Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename WarpMmaOperator::MathOperator;
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;
  using ThreadblockShape = typename BaseKernel::Mma::Shape;
  using WarpShape = typename BaseKernel::WarpShape;
  using InstructionShape = typename BaseKernel::InstructionShape;
  static int const kStages = BaseKernel::Mma::kStages;

  /// Argument structure
  using Arguments = typename BaseKernel::Arguments;

  using ProblemInfo = typename BaseKernel::ProblemVisitor::ProblemInfo;

 protected:
  /// Kernel parameters object
  typename BaseKernel::Params params_;

 private:
  /// Get the number of tiles across all problems in a group
  static int32_t group_tile_count(
      const cutlass::gemm::GemmCoord* problem_sizes_ptr, int problem_count) {
    int32_t tiles = 0;
    for (int32_t i = 0; i < problem_count; ++i) {
      cutlass::gemm::GemmCoord problem = problem_sizes_ptr[i];
      BaseKernel::ProblemVisitor::possibly_transpose_problem(problem);
      tiles += problem_tile_count(problem);
    }
    return tiles;
  }

  /// Copy from `data` to `workspace`
  Status copy_to_workspace(void* workspace, void* data, size_t bytes) {
    cudaError_t cuda_error =
        cudaMemcpy(workspace, data, bytes, cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
      // Call cudaGetLastError() to clear the error bit
      cuda_error = cudaGetLastError();
      CUTLASS_TRACE_HOST("  cudaMemcpy() returned error "
                         << cudaGetErrorString(cuda_error));
      return Status::kErrorInternal;
    }

    return Status::kSuccess;
  }

  /// Precomputes scheduling information for the grouped GEMM
  Status precompute(Arguments const& args,
                    int32_t tile_count,
                    void* workspace) {
    size_t workspace_bytes = get_workspace_size(args);
    std::vector<uint8_t> host_workspace(workspace_bytes);
    BaseKernel::ProblemVisitor::host_precompute(
        args.host_problem_sizes,
        args.problem_count,
        args.threadblock_count,
        reinterpret_cast<void*>(host_workspace.data()));
    return copy_to_workspace(workspace, host_workspace.data(), workspace_bytes);
  }

  /// Reorder `data` according to `indices`
  template <typename T>
  static void reorder_array(T* data, const std::vector<size_t>& indices) {
    // For now, simply create a copy of the data and then copy over to the
    // original.
    std::vector<T> copy(indices.size());
    for (int i = 0; i < indices.size(); ++i) {
      copy.at(i) = data[indices[i]];
    }

    memcpy(data, copy.data(), indices.size() * sizeof(T));
  }

 public:
  /// Constructs the GEMM.
  BaseGrouped() {}

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const& args) {
    return BaseKernel::can_implement(args);
  }

  /// Get the number of tiles in a problem
  static int32_t problem_tile_count(cutlass::gemm::GemmCoord const& problem) {
    auto grid = BaseKernel::ProblemVisitor::grid_shape(problem);
    return BaseKernel::ProblemVisitor::tile_count(grid);
  }

  /// Get the number of tiles across all problems in a group
  static int32_t group_tile_count(Arguments const& args) {
    if (args.host_problem_sizes == nullptr) {
      CUTLASS_TRACE_HOST("Received nullptr for `args.host_problem_sizes");
      return -1;
    }

    return group_tile_count(args.host_problem_sizes, args.problem_count);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const& args) {
    if (BaseKernel::ProblemVisitor::kRequiresPrecomputation) {
      return BaseKernel::ProblemVisitor::get_workspace_size(
          args.host_problem_sizes, args.problem_count, args.threadblock_count);
    } else {
      return 0;
    }
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const& args) {
    return dim3(args.threadblock_count, 1, 1);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int smem_capacity = -1) {
    CUTLASS_TRACE_HOST("BaseGrouped::maximum_active_blocks()");

    int smem_size =
        static_cast<int>(sizeof(typename BaseKernel::SharedStorage));

    CUTLASS_TRACE_HOST("  smem_size: " << smem_size << " bytes");

    cudaError_t result;
    if (smem_size > (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<BaseKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        // Call cudaGetLastError() to clear the error bit
        result = cudaGetLastError();
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error "
                           << cudaGetErrorString(result));
        return -1;
      }
    }

    int max_active_blocks = -1;
    result =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                      Kernel<BaseKernel>,
                                                      BaseKernel::kThreadCount,
                                                      smem_size);

    if (result != cudaSuccess) {
      // Call cudaGetLastError() to clear the error bit
      result = cudaGetLastError();
      CUTLASS_TRACE_HOST(
          "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error "
          << cudaGetErrorString(result));
      return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  /// Sorts each pointer passed in according to the indices that sort
  /// `problem_sizes_ptr` in descending order of problem-K dimension.
  static void sort_problems(int problem_count,
                            cutlass::gemm::GemmCoord* problem_sizes_ptr,
                            int64_t* lda_host_ptr,
                            int64_t* ldb_host_ptr,
                            int64_t* ldc_host_ptr,
                            int64_t* ldd_host_ptr,
                            int64_t* offset_A_ptr,
                            int64_t* offset_B_ptr,
                            int64_t* offset_C_ptr,
                            int64_t* offset_D_ptr) {
    std::vector<size_t> indices(problem_count);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(),
                     indices.end(),
                     [&problem_sizes_ptr](size_t i, size_t j) {
                       return problem_sizes_ptr[i].k() >
                              problem_sizes_ptr[j].k();
                     });

    reorder_array(problem_sizes_ptr, indices);
    reorder_array(lda_host_ptr, indices);
    reorder_array(ldb_host_ptr, indices);
    reorder_array(ldc_host_ptr, indices);
    reorder_array(ldd_host_ptr, indices);
    reorder_array(offset_A_ptr, indices);
    reorder_array(offset_B_ptr, indices);
    reorder_array(offset_C_ptr, indices);
    reorder_array(offset_D_ptr, indices);
  }

  /// Computes the number of threadblocks to launch for the grouped kernel
  static int sufficient(
      const cutlass::gemm::GemmCoord* problem_sizes_ptr = nullptr,
      int problem_count = 0,
      int available_sm_count = -1) {
    // Determine the number of blocks that would be launched to fill up a single
    // wave on the GPU with each SM having maximum occupancy.
    // printf("custom base\n");
    static cudaDeviceProp properties;
    static bool count = true;
    if (count) {
      int device_idx;
      cudaError_t result = cudaGetDevice(&device_idx);
      if (result != cudaSuccess) {
        // Call cudaGetLastError() to clear the error bit
        result = cudaGetLastError();
        CUTLASS_TRACE_HOST("  cudaGetDevice() returned error "
                           << cudaGetErrorString(result));
        return 0;
      }

      result = cudaGetDeviceProperties(&properties, device_idx);
      if (result != cudaSuccess) {
        // Call cudaGetLastError() to clear the error bit
        result = cudaGetLastError();
        CUTLASS_TRACE_HOST("  cudaGetDeviceProperties() returned error "
                           << cudaGetErrorString(result));
        return 0;
      }
    }
    count = false;

    bool override_sm_count =
        (available_sm_count < 0 ||
         available_sm_count > properties.multiProcessorCount);
    if (override_sm_count) {
      available_sm_count = properties.multiProcessorCount;
    }

    int max_active_blocks = maximum_active_blocks();
    if (max_active_blocks <= 0) {
      return 0;
    }

    int occupancy_based_block_count = available_sm_count * max_active_blocks;

    if (problem_sizes_ptr == nullptr || problem_count == 0) {
      return occupancy_based_block_count;
    }

    int total_tiles = group_tile_count(problem_sizes_ptr, problem_count);

    // If the group contains a single problem, launching the exact number of
    // threadblocks needed to cover the problem minimizes the work performed
    // per threadblock in finding the next tile to compute. We return
    // total_tiles unless the user has provided the SM count.
    if (problem_count == 1 && override_sm_count) {
      return total_tiles;
    }

    // Choose between the full wave of threadblocks and the tile count. If there
    // are fewer tiles in the group than threadblocks in the full wave, only
    // some threadblocks will be assigned tiles. Those threadblocks
    // which are not assigned tiles still need to perform the work of iterating
    // through problem sizes to determine that they have no work to do. This
    // competes for cycles with those threadblocks that are assigned tiles to
    // compute.
    return min(total_tiles, occupancy_based_block_count);
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const& args,
                    void* workspace = nullptr,
                    cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("BaseGrouped::initialize() - workspace "
                       << workspace
                       << ", stream: " << (stream ? "non-null" : "null"));

    // Workspace
    size_t workspace_bytes = get_workspace_size(args);

    if (workspace_bytes && !workspace) {
      return Status::kErrorWorkspaceNull;
    }

    if (BaseKernel::ProblemVisitor::kRequiresPrecomputation) {
      int32_t tile_count = group_tile_count(args);
      Status status = precompute(args, tile_count, workspace);
      if (status != Status::kSuccess) {
        return status;
      }

      params_ = typename BaseKernel::Params(args, workspace, tile_count);
    } else {
      params_ = typename BaseKernel::Params(args, workspace);
    }

    // Specify shared memory capacity for kernel.
    int smem_size =
        static_cast<int>(sizeof(typename BaseKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      cudaError_t result =
          cudaFuncSetAttribute(Kernel<BaseKernel>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const& args, void* workspace = nullptr) {
    size_t workspace_bytes = get_workspace_size(args);

    if (workspace_bytes && !workspace) {
      return Status::kErrorWorkspaceNull;
    }

    if (BaseKernel::ProblemVisitor::kRequiresPrecomputation) {
      int32_t tile_count = group_tile_count(args);
      Status status = precompute(args, tile_count, workspace);
      if (status != Status::kSuccess) {
        return status;
      }

      params_.update(args, workspace, tile_count);
    } else {
      params_.update(args, workspace);
    }

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {
    //
    // Configure grid and block dimensions
    //

    if (!params_.problem_visitor.problem_count) {
      return Status::kSuccess;
    }

    dim3 grid(params_.threadblock_count, 1, 1);
    dim3 block(BaseKernel::kThreadCount, 1, 1);

    int smem_size =
        static_cast<int>(sizeof(typename BaseKernel::SharedStorage));

    //
    // Launch kernel
    //

    // Launch
    cutlass::Kernel<BaseKernel><<<grid, block, smem_size, stream>>>(params_);

    //
    // Query for errors
    //
    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
      // Call cudaGetLastError() to clear the error bit
      result = cudaGetLastError();
      CUTLASS_TRACE_HOST("  grid launch failed with error "
                         << cudaGetErrorString(result));
      return Status::kErrorInternal;
    }

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

  /// Initializes and runs the kernel.
  Status operator()(Arguments const& args,
                    void* workspace,
                    cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace device
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
