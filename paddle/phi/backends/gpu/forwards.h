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

// Forward-declares CUDA API types used in platform-agnostic wrapper headers.
#pragma once

/// Forward declaration of Eigen types.
namespace Eigen {
struct GpuDevice;
}  // namespace Eigen

/// Forward declaration of CUDA types.

// Forward declaration of CUDA runtime types.
using cudaStream_t = struct CUstream_st *;
using cudaEvent_t = struct CUevent_st *;

// Forward declaration of cuDNN types.
using cudnnHandle_t = struct cudnnContext *;
using cudnnTensorDescriptor_t = struct cudnnTensorStruct *;
using cudnnConvolutionDescriptor_t = struct cudnnConvolutionStruct *;
using cudnnPoolingDescriptor_t = struct cudnnPoolingStruct *;
using cudnnFilterDescriptor_t = struct cudnnFilterStruct *;
using cudnnLRNDescriptor_t = struct cudnnLRNStruct *;
using cudnnActivationDescriptor_t = struct cudnnActivationStruct *;
using cudnnSpatialTransformerDescriptor_t =
    struct cudnnSpatialTransformerStruct *;
using cudnnOpTensorDescriptor_t = struct cudnnOpTensorStruct *;
using cudnnReduceTensorDescriptor_t = struct cudnnReduceTensorStruct *;
using cudnnCTCLossDescriptor_t = struct cudnnCTCLossStruct *;
using cudnnTensorTransformDescriptor_t = struct cudnnTensorTransformStruct *;
using cudnnDropoutDescriptor_t = struct cudnnDropoutStruct *;
using cudnnRNNDescriptor_t = struct cudnnRNNStruct *;
using cudnnPersistentRNNPlan_t = struct cudnnPersistentRNNPlan *;
using cudnnRNNDataDescriptor_t = struct cudnnRNNDataStruct *;
using cudnnAlgorithmDescriptor_t = struct cudnnAlgorithmStruct *;
using cudnnAlgorithmPerformance_t = struct cudnnAlgorithmPerformanceStruct *;
using cudnnSeqDataDescriptor_t = struct cudnnSeqDataStruct *;
using cudnnAttnDescriptor_t = struct cudnnAttnStruct *;
using cudnnFusedOpsConstParamPack_t = struct cudnnFusedOpsConstParamStruct *;
using cudnnFusedOpsVariantParamPack_t =
    struct cudnnFusedOpsVariantParamStruct *;
using cudnnFusedOpsPlan_t = struct cudnnFusedOpsPlanStruct *;

// Forward declaration of cuBLAS types.
using cublasHandle_t = struct cublasContext *;

// Forward declaration of cuSOLVER types.
using cusolverDnHandle_t = struct cusolverDnContext *;

// Forward declaration of cuSparse types.
using cusparseHandle_t = struct cusparseContext *;

// Forward declaration of cuFFT types.
using cufftHandle = int;

// Forward declaration of NCCL types.
using ncclComm_t = struct ncclComm *;

/// Forward declaration of ROCM types.
#include <cstddef>

using hipDevice_t = int;
using hipCtx_t = struct ihipCtx_t *;
using hipModule_t = struct ihipModule_t *;
using hipStream_t = struct ihipStream_t *;
using hipEvent_t = struct ihipEvent_t *;
using hipFunction_t = struct ihipModuleSymbol_t *;

// Forward declaration of MIOpen types.
using miopenHandle_t = struct miopenHandle *;
using miopenAcceleratorQueue_t = hipStream_t;
using miopenFusionOpDescriptor_t = struct miopenFusionOpDescriptor *;
using miopenTensorDescriptor_t = struct miopenTensorDescriptor *;
using miopenConvolutionDescriptor_t = struct miopenConvolutionDescriptor *;
using miopenPoolingDescriptor_t = struct miopenPoolingDescriptor *;
using miopenLRNDescriptor_t = struct miopenLRNDescriptor *;
using miopenActivationDescriptor_t = struct miopenActivationDescriptor *;
using miopenRNNDescriptor_t = struct miopenRNNDescriptor *;
using miopenCTCLossDescriptor_t = struct miopenCTCLossDescriptor *;
using miopenDropoutDescriptor_t = struct miopenDropoutDescriptor *;
using miopenFusionPlanDescriptor_t = struct miopenFusionPlanDescriptor *;
using miopenOperatorDescriptor_t = struct miopenOperatorDescriptor *;
using miopenOperatorArgs_t = struct miopenOperatorArgs *;
using miopenAllocatorFunction = void *(*)(void *context, size_t sizeBytes);
// using miopenDeallocatorFunction = void *(*)(void *context, void *memory);
// struct miopenConvAlgoPerf_t;
// struct miopenConvSolution_t;

// Forward declaration of rocBLAS types.
using rocblas_handle = struct _rocblas_handle *;

// Forward declaration of hipfft types.
using hipfftHandle = struct hipfftHandle_t *;

// Forward declaration of rocSOLVER types.
using rocsolver_handle = rocblas_handle;

// Forward declaration of rocSparse types.
using rocsparse_handle = struct _rocsparse_handle *;
