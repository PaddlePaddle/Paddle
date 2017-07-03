#include <cudnn.h>
#include "paddle/platform/dynamic_loader.h"

namespace paddle {
namespace dyload {

std::once_flag cudnn_dso_flag;
void* cudnn_dso_handle = nullptr;

#ifdef PADDLE_USE_DSO

#define DYNAMIC_LOAD_CUDNN_WRAP(__name)                                     \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    auto operator()(Args... args) -> decltype(__name(args...)) {            \
      using cudnn_func = decltype(__name(args...)) (*)(Args...);            \
      std::call_once(cudnn_dso_flag, GetCudnnDsoHandle, &cudnn_dso_handle); \
      void* p_##__name = dlsym(cudnn_dso_handle, #__name);                  \
      return reinterpret_cast<cudnn_func>(p_##__name)(args...);             \
    }                                                                       \
  } __name; /* struct DynLoad__##__name */

#else

#define DYNAMIC_LOAD_CUDNN_WRAP(__name)                          \
  struct DynLoad__##__name {                                     \
    template <typename... Args>                                  \
    auto operator()(Args... args) -> decltype(__name(args...)) { \
      return __name(args...);                                    \
    }                                                            \
  } __name; /* struct DynLoad__##__name */

#endif

/**
 * include all needed cudnn functions in HPPL
 * different cudnn version has different interfaces
 **/
// clang-format off
#define CUDNN_DNN_ROUTINE_EACH(__macro)                   \
  __macro(cudnnSetTensor4dDescriptor)                     \
  __macro(cudnnSetTensor4dDescriptorEx)                   \
  __macro(cudnnGetConvolutionNdForwardOutputDim)          \
  __macro(cudnnGetConvolutionForwardAlgorithm)            \
  __macro(cudnnCreateTensorDescriptor)                    \
  __macro(cudnnDestroyTensorDescriptor)                   \
  __macro(cudnnCreateFilterDescriptor)                    \
  __macro(cudnnSetFilter4dDescriptor)                     \
  __macro(cudnnSetPooling2dDescriptor)                    \
  __macro(cudnnDestroyFilterDescriptor)                   \
  __macro(cudnnCreateConvolutionDescriptor)               \
  __macro(cudnnCreatePoolingDescriptor)                   \
  __macro(cudnnDestroyPoolingDescriptor)                  \
  __macro(cudnnSetConvolution2dDescriptor)                \
  __macro(cudnnDestroyConvolutionDescriptor)              \
  __macro(cudnnCreate)                                    \
  __macro(cudnnDestroy)                                   \
  __macro(cudnnSetStream)                                 \
  __macro(cudnnActivationForward)                         \
  __macro(cudnnConvolutionForward)                        \
  __macro(cudnnConvolutionBackwardBias)                   \
  __macro(cudnnGetConvolutionForwardWorkspaceSize)        \
  __macro(cudnnTransformTensor)                           \
  __macro(cudnnPoolingForward)                            \
  __macro(cudnnPoolingBackward)                           \
  __macro(cudnnSoftmaxBackward)                           \
  __macro(cudnnSoftmaxForward)                            \
  __macro(cudnnGetVersion)                                \
  __macro(cudnnGetErrorString)
CUDNN_DNN_ROUTINE_EACH(DYNAMIC_LOAD_CUDNN_WRAP)

#define CUDNN_DNN_ROUTINE_EACH_R2(__macro)                \
  __macro(cudnnAddTensor)                                 \
  __macro(cudnnConvolutionBackwardData)                   \
  __macro(cudnnConvolutionBackwardFilter)
CUDNN_DNN_ROUTINE_EACH_R2(DYNAMIC_LOAD_CUDNN_WRAP)

// APIs available after R3:
#if CUDNN_VERSION >= 3000
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R3(__macro)              \
  __macro(cudnnGetConvolutionBackwardFilterWorkspaceSize)     \
  __macro(cudnnGetConvolutionBackwardDataAlgorithm)           \
  __macro(cudnnGetConvolutionBackwardFilterAlgorithm)         \
  __macro(cudnnGetConvolutionBackwardDataWorkspaceSize)
CUDNN_DNN_ROUTINE_EACH_AFTER_R3(DYNAMIC_LOAD_CUDNN_WRAP)
#undef CUDNN_DNN_ROUTINE_EACH_AFTER_R3
#endif


// APIs available after R4:
#if CUDNN_VERSION >= 4007
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R4(__macro)             \
  __macro(cudnnBatchNormalizationForwardTraining)            \
  __macro(cudnnBatchNormalizationForwardInference)           \
  __macro(cudnnBatchNormalizationBackward)
CUDNN_DNN_ROUTINE_EACH_AFTER_R4(DYNAMIC_LOAD_CUDNN_WRAP)
#undef CUDNN_DNN_ROUTINE_EACH_AFTER_R4
#endif

// APIs in R5
#if CUDNN_VERSION >= 5000
#define CUDNN_DNN_ROUTINE_EACH_R5(__macro)                    \
  __macro(cudnnCreateActivationDescriptor)                    \
  __macro(cudnnSetActivationDescriptor)                       \
  __macro(cudnnGetActivationDescriptor)                       \
  __macro(cudnnDestroyActivationDescriptor)
CUDNN_DNN_ROUTINE_EACH_R5(DYNAMIC_LOAD_CUDNN_WRAP)
#undef CUDNN_DNN_ROUTINE_EACH_R5
#endif

#undef CUDNN_DNN_ROUTINE_EACH
// clang-format on
}  // namespace dyload
}  // namespace paddle
