// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"

#include "cutlass_patch/batched_matrix_coord.h"
#include "cutlass_patch/cuda/default_config_id.h"
#include "cutlass_patch/epilogue/thread/linear_combination_unary.h"
#include "cutlass_patch/epilogue/thread/linear_combination_variadic.h"
#include "cutlass_patch/gemm/device/gemm_universal_with_variadic.h"

#include "params.h"  // NOLINT

#define CHECK_CUTLASS(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

namespace ap {
using bfloat16 = nv_bfloat16;

template <typename T, int N>
using Array = cutlass::Array<T, N>;

using MatrixCoord = cutlass_patch::BatchedMatrixCoord;

// Convert CUDA data type to cutlass data type
template <typename T>
struct CutlassDataType {
  using Type = T;
};

template <>
struct CutlassDataType<half> {
  using Type = cutlass::half_t;
};

template <>
struct CutlassDataType<__nv_bfloat16> {
  using Type = cutlass::bfloat16_t;
};

// Convert to cutlass layout
template <bool Transposed>
struct MatrixLayout {
  using Type = cutlass::layout::RowMajor;
};

template <>
struct MatrixLayout<true> {
  using Type = cutlass::layout::ColumnMajor;
};

// Operation performed by GEMM
template <typename ElementT>
struct GemmOperation {
  using Type = cutlass::arch::OpMultiplyAdd;
};

template <>
struct GemmOperation<float> {
  using Type = cutlass::arch::OpMultiplyAddFastF32;
};

static cutlass::gemm::GemmUniversalMode GetGemmMode(int batch_count) {
  return batch_count > 1 ? cutlass::gemm::GemmUniversalMode::kBatched
                         : cutlass::gemm::GemmUniversalMode::kGemm;
}

static void *GetWorkspace(size_t workspace_size) {
  static cutlass::device_memory::allocation<uint8_t> workspace;
  if (workspace.size() < workspace_size) {
    workspace.reset(workspace_size);
  }
  return workspace.get();
}

template <typename GemmFunc>
cutlass::Status SetMaxDynamicSharedMemorySize() {
  cudaError_t cudart_result;

  // If requires more than 48KB: configure for extended, dynamic shared memory
  if constexpr (GemmFunc::kSharedStorageSize >= (48 << 10)) {
    cudart_result =
        cudaFuncSetAttribute(cutlass::Kernel2<typename GemmFunc::GemmKernel>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             GemmFunc::kSharedStorageSize);
    if (cudart_result != cudaSuccess) {
      CUTLASS_TRACE_HOST("cudaFuncSetAttribute() returned error "
                         << cudaGetErrorString(cudart_result));
      return cutlass::Status::kErrorInternal;
    }
  }

#if AP_ENABLE_DEBUG
  // Update SM occupancy member
  int sm_occupancy = -1;
  cudart_result = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &sm_occupancy,
      cutlass::Kernel2<typename GemmFunc::GemmKernel>,
      GemmFunc::GemmKernel::kThreadCount,
      GemmFunc::kSharedStorageSize,
      cudaOccupancyDisableCachingOverride);
  if (cudart_result != cudaSuccess) {
    CUTLASS_TRACE_HOST(
        "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() returned "
        "error "
        << cudaGetErrorString(cudart_result));
    return cutlass::Status::kErrorInternal;
  }
  CUTLASS_TRACE_HOST("sm_occupancy: (" << sm_occupancy
                                       << ") "
                                          "smem_size: ("
                                       << GemmFunc::kSharedStorageSize
                                       << ") "
                                          "GemmKernel::kThreadCount: ("
                                       << GemmFunc::GemmKernel::kThreadCount
                                       << ")");
#endif
  return cutlass::Status::kSuccess;
}

template <typename ElementT,
          typename ElementComputeT,
          template <typename T>
          class VariadicFunctor,
          int AlignA = 128 / cutlass::sizeof_bits<ElementT>::value,
          int AlignB = 128 / cutlass::sizeof_bits<ElementT>::value,
          int ConfigId = DefaultConfig::kConfigId,
          int SwizzleFactor = DefaultConfig::kSwizzleFactor,
          bool Batched = DefaultConfig::kBatched>
void MatmulAddVariadic(
    const GemmEpilogueParams &params,
    const typename VariadicFunctor<ElementComputeT>::Arguments &variadic_args) {
  using ElementAccumulator =
      typename CutlassDataType<ElementComputeT>::Type;  // <- data type of
                                                        // accumulator
  using ElementComputeEpilogue =
      ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA =
      typename CutlassDataType<ElementT>::Type;  // <- data type of elements in
                                                 // input matrix A
  using ElementInputB =
      typename CutlassDataType<ElementT>::Type;  // <- data type of elements in
                                                 // input matrix B
  using ElementOutput =
      typename CutlassDataType<ElementT>::Type;  // <- data type of elements in
                                                 // output matrix D

  constexpr int AlignC = AlignB;

  // Epilogue operation as LinearCombination:
  //  alpha * accumulator + beta * source
  using EpilogueOutputOp =
      cutlass_patch::epilogue::thread::LinearCombinationVariadic<
          VariadicFunctor,
          ElementOutput,
          AlignC,
          ElementAccumulator,
          ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::NoBetaScaling>;  // <- alpha x
                                                                 // AB + bias

  using GemmFunc = cutlass_patch::gemm::device::GemmUniversalWithVariadic<
      ElementInputA,
      cutlass::layout::RowMajor,
      ElementInputB,
      cutlass::layout::RowMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::
          TShape,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::
          WShape,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::
          IShape,
      EpilogueOutputOp,
      typename GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::
          SwizzleThreadBlock,
      GemmTuningConfigs<ElementT, SwizzleFactor, Batched, ConfigId>::kNumStages,
      AlignA,
      AlignB,
      typename GemmOperation<ElementT>::Type>;

  CHECK_CUTLASS(SetMaxDynamicSharedMemorySize<GemmFunc>());

  /// Arguments
  cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};

  const ElementInputA *input =
      reinterpret_cast<const ElementInputA *>(params.input);
  const ElementInputB *weight =
      reinterpret_cast<const ElementInputB *>(params.weight);
  const ElementOutput *bias =
      reinterpret_cast<const ElementOutput *>(params.bias);
  ElementOutput *output = reinterpret_cast<ElementOutput *>(params.output);

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = bias ? static_cast<ElementComputeEpilogue>(1)
                                     : static_cast<ElementComputeEpilogue>(0);

  typename GemmFunc::Arguments arguments{
      GetGemmMode(params.batch_count),
      problem_size,        // <- problem size of matrix multiplication
      params.batch_count,  // <- batch_count or k-dimension split factor
      {alpha, beta, variadic_args},  // <- epilogue params, alpha, beta
      input,                         // <- input, ptr_A, A, shape={M, K}
      weight,                        // <- input, ptr_B, B, shape={K, N}
      bias,                          // <- input, ptr_C, shape={M, N} or {1, N}
      output,                        // <- output, ptr_D, Z, shape={M, N}
      params.shape_args.batch_stride_A,
      params.shape_args.batch_stride_B,
      params.shape_args.batch_stride_C,
      params.shape_args.batch_stride_D,
      params.shape_args.lda,
      params.shape_args.ldb,
      params.shape_args.ldc_bias,
      params.shape_args.ldd};

  size_t workspace_size = GemmFunc::get_workspace_size(arguments);
  void *workspace = workspace_size > 0 ? GetWorkspace(workspace_size) : nullptr;

  GemmFunc device_gemm;

  cudaStream_t *stream_ptr =
      reinterpret_cast<cudaStream_t *>(params.stream_ptr);

  CHECK_CUTLASS(device_gemm.can_implement(arguments));
  CHECK_CUTLASS(device_gemm.initialize(arguments, workspace, *stream_ptr));

  //
  // Run the GEMM
  //
  CHECK_CUTLASS(device_gemm(*stream_ptr));
#if AP_ENABLE_DEBUG
  CHECK_CUDA(cudaStreamSynchronize(*stream_ptr));
#endif
}

}  // namespace ap
