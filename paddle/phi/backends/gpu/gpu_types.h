// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#else  // PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#endif

namespace phi {

// Note(qili93): CUDA Runtime API supported by HIP
// https://github.com/ROCm/HIPIFY/blob/master/doc/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md

#ifdef PADDLE_WITH_HIP
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
  using GPU_TYPE = ROCM_TYPE;

#else  // PADDLE_WITH_CUDA

#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
  using GPU_TYPE = CUDA_TYPE;
#endif

DECLARE_TYPE_FOR_GPU(gpuError_t, cudaError_t, hipError_t);
DECLARE_TYPE_FOR_GPU(gpuMemcpyKind, cudaMemcpyKind, hipMemcpyKind);
DECLARE_TYPE_FOR_GPU(gpuDeviceProp, cudaDeviceProp, hipDeviceProp_t);
DECLARE_TYPE_FOR_GPU(dnnDataType_t, cudnnDataType_t, miopenDataType_t);
DECLARE_TYPE_FOR_GPU(dnnPoolingMode_t, cudnnPoolingMode_t, miopenPoolingMode_t);
DECLARE_TYPE_FOR_GPU(dnnTensorFormat_t,
                     cudnnTensorFormat_t,
                     miopenTensorFormat_t);
DECLARE_TYPE_FOR_GPU(dnnActivationMode_t,
                     cudnnActivationMode_t,
                     miopenActivationMode_t);
DECLARE_TYPE_FOR_GPU(gpuGraph_t, cudaGraph_t, hipGraph_t);
DECLARE_TYPE_FOR_GPU(gpuFunction_t, cudaFunction_t, hipFunction_t);
DECLARE_TYPE_FOR_GPU(gpuGraphExec_t, cudaGraphExec_t, hipGraphExec_t);
DECLARE_TYPE_FOR_GPU(gpuGraphNode_t, cudaGraphNode_t, hipGraphNode_t);
DECLARE_TYPE_FOR_GPU(gpuGraphNodeType, cudaGraphNodeType, hipGraphNodeType);
DECLARE_TYPE_FOR_GPU(gpuKernelNodeParams,
                     cudaKernelNodeParams,
                     hipKernelNodeParams);
DECLARE_TYPE_FOR_GPU(gpuStreamCaptureMode,
                     cudaStreamCaptureMode,
                     hipStreamCaptureMode);
DECLARE_TYPE_FOR_GPU(gpuStreamCaptureStatus,
                     cudaStreamCaptureStatus,
                     hipStreamCaptureStatus);

#undef DECLARE_TYPE_FOR_GPU

#ifdef PADDLE_WITH_HIP
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV) \
  constexpr auto GPU_CV = ROCM_CV;
#else  // PADDLE_WITH_CUDA
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV) \
  constexpr auto GPU_CV = CUDA_CV;
#endif

DECLARE_CONSTANT_FOR_GPU(gpuErrorOutOfMemory,
                         cudaErrorMemoryAllocation,
                         hipErrorOutOfMemory);
DECLARE_CONSTANT_FOR_GPU(gpuErrorNotReady, cudaErrorNotReady, hipErrorNotReady);
DECLARE_CONSTANT_FOR_GPU(gpuSuccess, cudaSuccess, hipSuccess);

DECLARE_CONSTANT_FOR_GPU(gpuMemcpyHostToDevice,
                         cudaMemcpyKind::cudaMemcpyHostToDevice,
                         hipMemcpyKind::hipMemcpyHostToDevice);
DECLARE_CONSTANT_FOR_GPU(gpuMemcpyDeviceToHost,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost,
                         hipMemcpyKind::hipMemcpyDeviceToHost);
DECLARE_CONSTANT_FOR_GPU(gpuMemcpyDeviceToDevice,
                         cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                         hipMemcpyKind::hipMemcpyDeviceToDevice);
DECLARE_CONSTANT_FOR_GPU(gpuEventDisableTiming,
                         cudaEventDisableTiming,
                         hipEventDisableTiming);
DECLARE_CONSTANT_FOR_GPU(gpuStreamNonBlocking,
                         cudaStreamNonBlocking,
                         hipStreamNonBlocking);
DECLARE_CONSTANT_FOR_GPU(gpuStreamCaptureModeThreadLocal,
                         cudaStreamCaptureModeThreadLocal,
                         hipStreamCaptureModeThreadLocal);
DECLARE_CONSTANT_FOR_GPU(gpuStreamCaptureModeRelaxed,
                         cudaStreamCaptureModeRelaxed,
                         hipStreamCaptureModeRelaxed);
DECLARE_CONSTANT_FOR_GPU(gpuStreamCaptureStatusActive,
                         cudaStreamCaptureStatusActive,
                         hipStreamCaptureStatusActive);
DECLARE_CONSTANT_FOR_GPU(gpuGraphNodeTypeKernel,
                         cudaGraphNodeTypeKernel,
                         hipGraphNodeTypeKernel);

#undef DECLARE_CONSTANT_FOR_GPU

#ifdef PADDLE_WITH_HIP
#define DECLARE_FUNCTION_FOR_GPU(GPU_FUNC, CUDA_FUNC, ROCM_FUNC) \
  const auto GPU_FUNC = ROCM_FUNC;
#else  // PADDLE_WITH_CUDA
#define DECLARE_FUNCTION_FOR_GPU(GPU_FUNC, CUDA_FUNC, ROCM_FUNC) \
  const auto GPU_FUNC = CUDA_FUNC;
#endif

DECLARE_FUNCTION_FOR_GPU(gpuGraphGetNodes, cudaGraphGetNodes, hipGraphGetNodes);
DECLARE_FUNCTION_FOR_GPU(gpuGraphGetEdges, cudaGraphGetEdges, hipGraphGetEdges);
DECLARE_FUNCTION_FOR_GPU(gpuGraphLaunch, cudaGraphLaunch, hipGraphLaunch);
DECLARE_FUNCTION_FOR_GPU(gpuGraphDestroy, cudaGraphDestroy, hipGraphDestroy);
DECLARE_FUNCTION_FOR_GPU(gpuGraphExecDestroy,
                         cudaGraphExecDestroy,
                         hipGraphExecDestroy);
DECLARE_FUNCTION_FOR_GPU(gpuGraphNodeGetType,
                         cudaGraphNodeGetType,
                         hipGraphNodeGetType);
DECLARE_FUNCTION_FOR_GPU(gpuGraphExecKernelNodeSetParams,
                         cudaGraphExecKernelNodeSetParams,
                         hipGraphExecKernelNodeSetParams);
DECLARE_FUNCTION_FOR_GPU(gpuGraphKernelNodeGetParams,
                         cudaGraphKernelNodeGetParams,
                         hipGraphKernelNodeGetParams);
DECLARE_FUNCTION_FOR_GPU(gpuStreamCreateWithPriority,
                         cudaStreamCreateWithPriority,
                         hipStreamCreateWithPriority);
DECLARE_FUNCTION_FOR_GPU(gpuStreamBeginCapture,
                         cudaStreamBeginCapture,
                         hipStreamBeginCapture);
DECLARE_FUNCTION_FOR_GPU(gpuStreamEndCapture,
                         cudaStreamEndCapture,
                         hipStreamEndCapture);
DECLARE_FUNCTION_FOR_GPU(gpuStreamGetCaptureInfo,
                         cudaStreamGetCaptureInfo,
                         hipStreamGetCaptureInfo);
DECLARE_FUNCTION_FOR_GPU(gpuEventCreateWithFlags,
                         cudaEventCreateWithFlags,
                         hipEventCreateWithFlags);
DECLARE_FUNCTION_FOR_GPU(gpuEventRecord, cudaEventRecord, hipEventRecord);
DECLARE_FUNCTION_FOR_GPU(gpuEventDestroy, cudaEventDestroy, hipEventDestroy);
DECLARE_FUNCTION_FOR_GPU(gpuEventQuery, cudaEventQuery, hipEventQuery);
DECLARE_FUNCTION_FOR_GPU(gpuEventSynchronize,
                         cudaEventSynchronize,
                         hipEventSynchronize);

#undef DECLARE_FUNCTION_FOR_GPU

}  // namespace phi

#endif  // defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
