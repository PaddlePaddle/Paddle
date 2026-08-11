// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "hytlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "hytlass/gemm/device/gemm_universal.h"
#include "hytlass/gemm/device/gemm_universal_with_broadcast.h"
#include "hytlass/util/device_memory.h"

#include "cutlass_patch/batched_matrix_coord.h"
#include "cutlass_patch/epilogue/thread/linear_combination_unary.h"
#include "cutlass_patch/epilogue/thread/linear_combination_variadic.h"
#include "cutlass_patch/gemm/device/gemm_universal_with_variadic.h"
#include "cutlass_patch/hip/all_tuning_configs.h"

#include "params.h"  // NOLINT

#define CHECK_HYTLASS(status)                               \
  {                                                         \
    auto error = status;                                    \
    if (error != hytlass::Status::kSuccess) {               \
      std::cerr << "HYTLASS error = " << int(error) << " (" \
                << hytlassGetStatusString(error) << ")"     \
                << " at line " << __LINE__ << std::endl;    \
      std::abort();                                         \
    }                                                       \
  }

namespace ap {

using MatrixCoord = cutlass_patch::BatchedMatrixCoord;
using bfloat16 = __hip_bfloat16;

// Operation performed by GEMM
template <typename ElementT>
struct GemmOperation {
  using Type = hytlass::arch::OpMultiplyAdd;
};

template <>
struct GemmOperation<float> {
  using Type = hytlass::arch::OpMultiplyAddFastF32;
};

static hytlass::gemm::GemmUniversalMode GetGemmMode(int batch_count) {
  return batch_count > 1 ? hytlass::gemm::GemmUniversalMode::kBatched
                         : hytlass::gemm::GemmUniversalMode::kGemm;
}

static void *GetWorkspace(size_t workspace_size) {
  static hytlass::device_memory::allocation<uint8_t> workspace;
  if (workspace.size() < workspace_size) {
    workspace.reset(workspace_size);
  }
  return workspace.get();
}

template <typename GemmFunc>
hytlass::Status SetMaxDynamicSharedMemorySize() {
  hipError_t hiprt_result;

  // If requires more than 48KB: configure for extended, dynamic shared memory
  if constexpr (GemmFunc::kSharedStorageSize >= (48 << 10)) {
    hiprt_result = hipFuncSetAttribute(
        (const void *)hytlass::Kernel2<typename GemmFunc::GemmKernel>,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        GemmFunc::kSharedStorageSize);
    if (hiprt_result != hipSuccess) {
      HYTLASS_TRACE_HOST("hipFuncSetAttribute() returned error "
                         << hipGetErrorString(hiprt_result));
      return hytlass::Status::kErrorInternal;
    }
  }

#if AP_ENABLE_DEBUG
  // Update SM occupancy member
  int sm_occupancy = -1;
  hiprt_result = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &sm_occupancy,
      hytlass::Kernel2<typename GemmFunc::GemmKernel>,
      GemmFunc::GemmKernel::kThreadCount,
      GemmFunc::kSharedStorageSize,
      hipOccupancyDisableCachingOverride);
  if (hiprt_result != hipSuccess) {
    HYTLASS_TRACE_HOST(
        "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() returned "
        "error "
        << hipGetErrorString(hiprt_result));
    return hytlass::Status::kErrorInternal;
  }
  HYTLASS_TRACE_HOST("sm_occupancy: (" << sm_occupancy
                                       << ") "
                                          "smem_size: ("
                                       << GemmFunc::kSharedStorageSize
                                       << ") "
                                          "GemmKernel::kThreadCount: ("
                                       << GemmFunc::GemmKernel::kThreadCount
                                       << ")");
#endif
  return hytlass::Status::kSuccess;
}

// Convert HIP data type to hytlass data type
template <typename T>
struct HytlassDataType {
  using Type = T;
};

template <>
struct HytlassDataType<half> {
  using Type = hytlass::half_t;
};

template <>
struct HytlassDataType<__hip_bfloat16> {
  using Type = hytlass::bfloat16_t;
};

// Convert to hytlass layout
template <bool Transposed>
struct MatrixLayout {
  using Type = hytlass::layout::RowMajor;
};

template <>
struct MatrixLayout<true> {
  using Type = hytlass::layout::ColumnMajor;
};

template <typename ElementT,
          typename ElementComputeT,
          template <typename T>
          class VariadicFunctor,
          int AlignA = 128 / hytlass::sizeof_bits<ElementT>::value,
          int AlignB = 128 / hytlass::sizeof_bits<ElementT>::value,
          int ConfigId = DefaultConfig::kConfigId,
          int SwizzleFactor = DefaultConfig::kSwizzleFactor,
          bool Batched = DefaultConfig::kBatched>
void MatmulAddVariadic(
    const GemmEpilogueParams &params,
    const typename VariadicFunctor<ElementComputeT>::Arguments &variadic_args) {
  // <- data type of accumulator
  using ElementAccumulator = typename HytlassDataType<ElementComputeT>::Type;
  // <- data type of epilogue operations
  using ElementComputeEpilogue = ElementAccumulator;
  // <- data type of elements in input matrix A
  using ElementInputA = typename HytlassDataType<ElementT>::Type;
  // <- data type of elements in input matrix B
  using ElementInputB = typename HytlassDataType<ElementT>::Type;
  // <- data type of elements in output matrix D
  using ElementOutput = typename HytlassDataType<ElementT>::Type;

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
          hytlass::epilogue::thread::ScaleType::NoBetaScaling>;

  using GemmFunc = cutlass_patch::gemm::device::GemmUniversalWithVariadic<
      ElementInputA,
      hytlass::layout::RowMajor,
      ElementInputB,
      hytlass::layout::RowMajor,
      ElementOutput,
      hytlass::layout::RowMajor,
      ElementAccumulator,
      hytlass::arch::OpClassTensorOp,
      hytlass::arch::Gfx928,
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

  CHECK_HYTLASS(SetMaxDynamicSharedMemorySize<GemmFunc>());

  /// Arguments
  hytlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};

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

  hipStream_t *stream_ptr = reinterpret_cast<hipStream_t *>(params.stream_ptr);

  CHECK_HYTLASS(device_gemm.can_implement(arguments));
  CHECK_HYTLASS(device_gemm.initialize(arguments, workspace, *stream_ptr));

  // Run the GEMM
  CHECK_HYTLASS(device_gemm(*stream_ptr));
#if AP_ENABLE_DEBUG
  CHECK_HIP(hipStreamSynchronize(*stream_ptr));
#endif
}

}  // namespace ap
