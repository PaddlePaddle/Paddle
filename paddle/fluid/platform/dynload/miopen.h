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

#include <miopen/miopen.h>
#include <dlfcn.h>
#include <mutex>  // NOLINT
#include "paddle/fluid/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag miopen_dso_flag;
extern void* miopen_dso_handle;
extern bool HasMIOpen();

inline const char* miopenGetErrorString(miopenStatus_t status) {
  switch (status) {
    case miopenStatusSuccess:
      return "MIOPEN_STATUS_SUCCESS";
    case miopenStatusNotInitialized:
      return "MIOPEN_STATUS_NOT_INITIALIZED";
    case miopenStatusInvalidValue:
      return "MIOPEN_STATUS_INVALID_VALUE";
    case miopenStatusBadParm:
      return "MIOPEN_STATUS_BAD_PARAM";
    case miopenStatusAllocFailed:
      return "MIOPEN_STATUS_ALLOC_FAILED";
    case miopenStatusInternalError:
      return "MIOPEN_STATUS_INTERNAL_ERROR";
    case miopenStatusNotImplemented:
      return "MIOPEN_STATUS_NOT_IMPLEMENTED";
    case miopenStatusUnknownError:
    default:
      return "MIOPEN_STATUS_UNKNOWN_ERROR";
  }
}

#ifdef PADDLE_USE_DSO

extern void EnforceMIOPENLoaded(const char* fn_name);
#define DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP(__name)                            \
  struct DynLoad__##__name {                                               \
    template <typename... Args>                                            \
    auto operator()(Args... args) -> decltype(__name(args...)) {           \
      using miopen_func = decltype(&::__name);                              \
      std::call_once(miopen_dso_flag, []() {                                \
        miopen_dso_handle = paddle::platform::dynload::GetMIOpenDsoHandle(); \
      });                                                                  \
      EnforceMIOPENLoaded(#__name);                                         \
      void* p_##__name = dlsym(miopen_dso_handle, #__name);                 \
      return reinterpret_cast<miopen_func>(p_##__name)(args...);            \
    }                                                                      \
  };                                                                       \
  extern struct DynLoad__##__name __name

#else

#define DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP(__name)                  \
  struct DynLoad__##__name {                                     \
    template <typename... Args>                                  \
    auto operator()(Args... args) -> decltype(__name(args...)) { \
      return __name(args...);                                    \
    }                                                            \
  };                                                             \
  extern DynLoad__##__name __name

#endif

/**
 * include all needed cudnn functions in HPPL
 * different cudnn version has different interfaces
 **/
#define MIOPEN_DNN_ROUTINE_EACH(__macro)                     \
  __macro(miopenSet4dTensorDescriptor);                     \
  __macro(miopenGet4dTensorDescriptor);                     \
  __macro(miopenFindConvolutionForwardAlgorithm);           \
  __macro(miopenGetConvolutionDescriptor);                  \
  __macro(miopenCreateTensorDescriptor);                    \
  __macro(miopenDestroyTensorDescriptor);                   \
  __macro(miopenSet2dPoolingDescriptor);                    \
  __macro(miopenGet2dPoolingDescriptor);                    \
  __macro(miopenCreateConvolutionDescriptor);               \
  __macro(miopenCreatePoolingDescriptor);                   \
  __macro(miopenDestroyPoolingDescriptor);                  \
  __macro(miopenInitConvolutionDescriptor);                 \
  __macro(miopenDestroyConvolutionDescriptor);              \
  __macro(miopenDeriveBNTensorDescriptor);                  \
  __macro(miopenCreate);                                    \
  __macro(miopenDestroy);                                   \
  __macro(miopenSetStream);                                 \
  __macro(miopenActivationForward);                         \
  __macro(miopenConvolutionForward);                        \
  __macro(miopenConvolutionBackwardBias);                   \
  __macro(miopenConvolutionForwardGetWorkSpaceSize);        \
  __macro(miopenPoolingGetWorkSpaceSize);                   \
  __macro(miopenPoolingForward);                            \
  __macro(miopenPoolingBackward);                           \
  __macro(miopenSoftmaxBackward);                           \
  __macro(miopenSoftmaxForward);                            \
  __macro(miopenConvolutionBackwardData);                   \
  __macro(miopenConvolutionBackwardWeights);                \
  __macro(miopenFindConvolutionBackwardDataAlgorithm);      \
  __macro(miopenFindConvolutionBackwardWeightsAlgorithm);   \
  __macro(miopenConvolutionBackwardWeightsGetWorkSpaceSize);\
  __macro(miopenConvolutionBackwardDataGetWorkSpaceSize);   \
  __macro(miopenBatchNormalizationForwardTraining);         \
  __macro(miopenBatchNormalizationForwardInference);        \
  __macro(miopenBatchNormalizationBackward);                \
  __macro(miopenCreateActivationDescriptor);                \
  __macro(miopenSetActivationDescriptor);                   \
  __macro(miopenGetActivationDescriptor);                   \
  __macro(miopenDestroyActivationDescriptor);
MIOPEN_DNN_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MIOPEN_WRAP)


}  // namespace dynload
}  // namespace platform
}  // namespace paddle
