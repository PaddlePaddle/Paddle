/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#ifdef PADDLE_WITH_CUDA
#include <cudnn.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

TEST_API extern std::once_flag cudnn_dso_flag;
TEST_API extern void* cudnn_dso_handle;
extern bool HasCUDNN();

TEST_API extern void EnforceCUDNNLoaded(const char* fn_name);
#define DECLARE_DYNAMIC_LOAD_CUDNN_WRAP(__name)                      \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using cudnn_func = decltype(&::__name);                        \
      std::call_once(cudnn_dso_flag, []() {                          \
        cudnn_dso_handle = phi::dynload::GetCUDNNDsoHandle();        \
      });                                                            \
      EnforceCUDNNLoaded(#__name);                                   \
      static void* p_##__name = dlsym(cudnn_dso_handle, #__name);    \
      return reinterpret_cast<cudnn_func>(p_##__name)(args...);      \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

/**
 * include all needed cudnn functions in HPPL
 * different cudnn version has different interfaces
 **/
#define CUDNN_DNN_ROUTINE_EACH(__macro)                    \
  __macro(cudnnSetCallback);                               \
  __macro(cudnnSetTensor4dDescriptor);                     \
  __macro(cudnnSetTensor4dDescriptorEx);                   \
  __macro(cudnnSetTensorNdDescriptor);                     \
  __macro(cudnnGetTensorNdDescriptor);                     \
  __macro(cudnnGetConvolutionNdForwardOutputDim);          \
  __macro(cudnnCreateTensorDescriptor);                    \
  __macro(cudnnDestroyTensorDescriptor);                   \
  __macro(cudnnCreateFilterDescriptor);                    \
  __macro(cudnnSetFilter4dDescriptor);                     \
  __macro(cudnnSetFilterNdDescriptor);                     \
  __macro(cudnnGetFilterNdDescriptor);                     \
  __macro(cudnnSetPooling2dDescriptor);                    \
  __macro(cudnnSetPoolingNdDescriptor);                    \
  __macro(cudnnGetPoolingNdDescriptor);                    \
  __macro(cudnnDestroyFilterDescriptor);                   \
  __macro(cudnnCreateConvolutionDescriptor);               \
  __macro(cudnnCreatePoolingDescriptor);                   \
  __macro(cudnnDestroyPoolingDescriptor);                  \
  __macro(cudnnSetConvolution2dDescriptor);                \
  __macro(cudnnDestroyConvolutionDescriptor);              \
  __macro(cudnnSetConvolutionNdDescriptor);                \
  __macro(cudnnGetConvolutionNdDescriptor);                \
  __macro(cudnnDeriveBNTensorDescriptor);                  \
  __macro(cudnnCreateSpatialTransformerDescriptor);        \
  __macro(cudnnSetSpatialTransformerNdDescriptor);         \
  __macro(cudnnDestroySpatialTransformerDescriptor);       \
  __macro(cudnnSpatialTfGridGeneratorForward);             \
  __macro(cudnnSpatialTfGridGeneratorBackward);            \
  __macro(cudnnSpatialTfSamplerForward);                   \
  __macro(cudnnSpatialTfSamplerBackward);                  \
  __macro(cudnnCreate);                                    \
  __macro(cudnnDestroy);                                   \
  __macro(cudnnSetStream);                                 \
  __macro(cudnnActivationForward);                         \
  __macro(cudnnActivationBackward);                        \
  __macro(cudnnConvolutionForward);                        \
  __macro(cudnnConvolutionBackwardBias);                   \
  __macro(cudnnGetConvolutionForwardWorkspaceSize);        \
  __macro(cudnnTransformTensor);                           \
  __macro(cudnnPoolingForward);                            \
  __macro(cudnnPoolingBackward);                           \
  __macro(cudnnSoftmaxBackward);                           \
  __macro(cudnnSoftmaxForward);                            \
  __macro(cudnnGetVersion);                                \
  __macro(cudnnFindConvolutionForwardAlgorithmEx);         \
  __macro(cudnnFindConvolutionBackwardFilterAlgorithmEx);  \
  __macro(cudnnFindConvolutionBackwardFilterAlgorithm);    \
  __macro(cudnnFindConvolutionBackwardDataAlgorithmEx);    \
  __macro(cudnnGetErrorString);                            \
  __macro(cudnnCreateDropoutDescriptor);                   \
  __macro(cudnnDropoutGetStatesSize);                      \
  __macro(cudnnSetDropoutDescriptor);                      \
  __macro(cudnnRestoreDropoutDescriptor);                  \
  __macro(cudnnCreateRNNDescriptor);                       \
  __macro(cudnnDestroyDropoutDescriptor);                  \
  __macro(cudnnDestroyRNNDescriptor);                      \
  __macro(cudnnSetTensorNdDescriptorEx);                   \
  __macro(cudnnAddTensor);                                 \
  __macro(cudnnConvolutionBackwardData);                   \
  __macro(cudnnConvolutionBackwardFilter);                 \
  __macro(cudnnGetConvolutionBackwardFilterWorkspaceSize); \
  __macro(cudnnGetConvolutionBackwardDataWorkspaceSize);   \
  __macro(cudnnBatchNormalizationForwardTraining);         \
  __macro(cudnnBatchNormalizationForwardInference);        \
  __macro(cudnnBatchNormalizationBackward);                \
  __macro(cudnnCreateActivationDescriptor);                \
  __macro(cudnnSetActivationDescriptor);                   \
  __macro(cudnnGetActivationDescriptor);                   \
  __macro(cudnnDestroyActivationDescriptor);
CUDNN_DNN_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)

#if CUDNN_VERSION >= 7000 && CUDNN_VERSION < 8000
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R7_LESS_R8(__macro) \
  __macro(cudnnGetConvolutionBackwardFilterAlgorithm);   \
  __macro(cudnnGetConvolutionForwardAlgorithm);          \
  __macro(cudnnGetConvolutionBackwardDataAlgorithm);     \
  __macro(cudnnSetRNNDescriptor);
CUDNN_DNN_ROUTINE_EACH_AFTER_R7_LESS_R8(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION >= 7001
#define CUDNN_DNN_ROUTINE_EACH_R7(__macro)                \
  __macro(cudnnSetConvolutionGroupCount);                 \
  __macro(cudnnSetConvolutionMathType);                   \
  __macro(cudnnConvolutionBiasActivationForward);         \
  __macro(cudnnCreateCTCLossDescriptor);                  \
  __macro(cudnnDestroyCTCLossDescriptor);                 \
  __macro(cudnnGetCTCLossDescriptor);                     \
  __macro(cudnnSetCTCLossDescriptor);                     \
  __macro(cudnnGetCTCLossWorkspaceSize);                  \
  __macro(cudnnCTCLoss);                                  \
  __macro(cudnnGetConvolutionBackwardDataAlgorithm_v7);   \
  __macro(cudnnGetConvolutionBackwardFilterAlgorithm_v7); \
  __macro(cudnnGetConvolutionForwardAlgorithm_v7);        \
  __macro(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
CUDNN_DNN_ROUTINE_EACH_R7(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION >= 7201
#define CUDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7(__macro) \
  __macro(cudnnCreateRNNDataDescriptor);             \
  __macro(cudnnDestroyRNNDataDescriptor);            \
  __macro(cudnnSetRNNDataDescriptor);
CUDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION >= 7401
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R7(__macro)                     \
  __macro(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize); \
  __macro(cudnnBatchNormalizationForwardTrainingEx);                 \
  __macro(cudnnGetBatchNormalizationBackwardExWorkspaceSize);        \
  __macro(cudnnBatchNormalizationBackwardEx);                        \
  __macro(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);
CUDNN_DNN_ROUTINE_EACH_AFTER_R7(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION >= 8000
#define CUDNN_DNN_ROUTINE_EACH_R8(__macro)            \
  __macro(cudnnSetRNNDescriptor_v8);                  \
  __macro(cudnnCreateFusedOpsPlan);                   \
  __macro(cudnnCreateFusedOpsConstParamPack);         \
  __macro(cudnnCreateFusedOpsVariantParamPack);       \
  __macro(cudnnDestroyFusedOpsPlan);                  \
  __macro(cudnnDestroyFusedOpsConstParamPack);        \
  __macro(cudnnDestroyFusedOpsVariantParamPack);      \
  __macro(cudnnFusedOpsExecute);                      \
  __macro(cudnnSetFusedOpsConstParamPackAttribute);   \
  __macro(cudnnSetFusedOpsVariantParamPackAttribute); \
  __macro(cudnnMakeFusedOpsPlan);
CUDNN_DNN_ROUTINE_EACH_R8(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#ifdef PADDLE_WITH_CUDNN_FRONTEND
#define CUDNN_DNN_ROUTINE_EACH_FRONTEND(__macro) \
  __macro(cudnnBackendCreateDescriptor);         \
  __macro(cudnnBackendDestroyDescriptor);        \
  __macro(cudnnBackendExecute);                  \
  __macro(cudnnBackendFinalize);                 \
  __macro(cudnnBackendGetAttribute);             \
  __macro(cudnnBackendSetAttribute);             \
  __macro(cudnnGetStream);                       \
  __macro(cudnnReorderFilterAndBias);
CUDNN_DNN_ROUTINE_EACH_FRONTEND(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION < 90000
#define CUDNN_DNN_ROUTINE_EACH_REMOVED_IN_E9(__macro) \
  __macro(cudnnGetRNNParamsSize);                     \
  __macro(cudnnGetRNNWorkspaceSize);                  \
  __macro(cudnnGetRNNTrainingReserveSize);            \
  __macro(cudnnSetRNNDescriptor_v6);                  \
  __macro(cudnnRNNForwardInference);                  \
  __macro(cudnnRNNForwardTraining);                   \
  __macro(cudnnRNNBackwardData);                      \
  __macro(cudnnRNNBackwardWeights);
CUDNN_DNN_ROUTINE_EACH_REMOVED_IN_E9(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION < 90000 && CUDNN_VERSION >= 7201
#define CUDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7_REMOVED_IN_E9(__macro) \
  __macro(cudnnSetRNNPaddingMode);                                 \
  __macro(cudnnRNNForwardInferenceEx);                             \
  __macro(cudnnRNNForwardTrainingEx);                              \
  __macro(cudnnRNNBackwardDataEx);                                 \
  __macro(cudnnRNNBackwardWeightsEx);
CUDNN_DNN_ROUTINE_EACH_AFTER_TWO_R7_REMOVED_IN_E9(
    DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION >= 90000
#define CUDNN_DNN_ROUTINE_EACH_R9(__macro) \
  __macro(cudnnGetLastErrorString);        \
  __macro(cudnnGetRNNWeightSpaceSize);     \
  __macro(cudnnGetRNNTempSpaceSizes);      \
  __macro(cudnnRNNForward);                \
  __macro(cudnnRNNBackwardData_v8);        \
  __macro(cudnnRNNBackwardWeights_v8);
CUDNN_DNN_ROUTINE_EACH_R9(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif
}  // namespace dynload
}  // namespace phi

#endif
