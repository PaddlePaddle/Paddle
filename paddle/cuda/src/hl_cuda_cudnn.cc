/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "hl_cuda_cudnn.h"
#include <cudnn.h>
#include <gflags/gflags.h>
#include "hl_cuda_cudnn.ph"
#include "hl_thread.ph"
#include "paddle/utils/DynamicLoader.h"
#include "paddle/utils/Logging.h"

DEFINE_int32(cudnn_conv_workspace_limit_in_mb,
             4096,
             "Specify cuDNN max workspace limit, in units MB, "
             "4096MB=4GB by default.");

namespace dynload {

std::once_flag cudnn_dso_flag;
void* cudnn_dso_handle = nullptr;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cudbnn routine
 * via operator overloading: operator ()
 *
 * note: default dynamic linked libs
 **/

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
} /* namespace dynload */

/**
 * Check build-in cudnn function using glog and it **does not**
 * support << operator for more details error info.
 */
#define CHECK_CUDNN(cudnnFunc)                                         \
  do {                                                                 \
    cudnnStatus_t cudnnStat = cudnnFunc;                               \
    CHECK_EQ(CUDNN_STATUS_SUCCESS, cudnnStat)                          \
        << "Cudnn Error: " << dynload::cudnnGetErrorString(cudnnStat); \
  } while (0)

bool g_is_libcudnn_init = false;
int g_cudnn_lib_version = 0;

void hl_cudnn_desc_init(cudnnTensorDescriptor_t* cudnn_desc) {
  CHECK_CUDNN(dynload::cudnnCreateTensorDescriptor(cudnn_desc));
}

void hl_cudnn_init(cudnnHandle_t* cudnn_handle, cudaStream_t stream) {
  size_t cudnn_dso_ver = dynload::cudnnGetVersion();
  size_t cudnn_dso_major = cudnn_dso_ver / 1000;
  size_t cudnn_cuh_major = CUDNN_VERSION / 1000;

  // Compare cudnn header version with that of cudnn.so.
  CHECK((cudnn_cuh_major < 4 && cudnn_dso_major < 4) ||
        (cudnn_cuh_major == cudnn_dso_major))
      << "[cudnn init] libcudnn v" << cudnn_dso_major << " with header v"
      << cudnn_cuh_major << " unmatched!\n"
      << "PaddlePaddle Requirement: "
      << "(header v[2-3] with libcudnn v[2-3]) Or "
      << "(header v4 with libcudnn v4) Or "
      << "(header v5 with libcudnn v5) Or"
      << "(header v6 with libcudnn v6).";

  CHECK(!(CUDNN_VERSION < 6000 && CUDNN_VERSION >= 5000 && CUDA_VERSION < 7050))
      << "cudnn v5 requires cuda version >= 7.5";

  CHECK(!(CUDNN_VERSION >= 6000 && CUDA_VERSION < 8000))
      << "cudnn v6 requires cuda version >= 8.0";

  CHECK_CUDNN(dynload::cudnnCreate(cudnn_handle));
  CHECK_CUDNN(dynload::cudnnSetStream(*cudnn_handle, stream));

  g_is_libcudnn_init = true;
  g_cudnn_lib_version = cudnn_dso_ver;
}

int hl_get_cudnn_lib_version() { return g_cudnn_lib_version; }

void hl_conv_workspace(hl_tensor_descriptor input,
                       hl_tensor_descriptor output,
                       hl_filter_descriptor filter,
                       hl_convolution_descriptor conv,
                       int* convFwdAlgo,
                       size_t* fwdLimitBytes,
                       int* convBwdDataAlgo,
                       size_t* bwdDataLimitBytes,
                       int* convBwdFilterAlgo,
                       size_t* bwdFilterLimitBytes) {
#if CUDNN_VERSION >= 4000

  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(filter);
  CHECK_NOTNULL(conv);

  // Specify workspace limit directly
  size_t memoryLimitBytes =
      (1LL << 20) * FLAGS_cudnn_conv_workspace_limit_in_mb;

  // cudnn convolution forward configuration
  cudnnTensorDescriptor_t fwd_src_desc = GET_TENSOR_DESCRIPTOR(input);
  cudnnTensorDescriptor_t fwd_dest_desc = GET_TENSOR_DESCRIPTOR(output);
  cudnnFilterDescriptor_t fwd_filter_desc = GET_FILTER_DESCRIPTOR(filter);
  cudnnConvolutionDescriptor_t fwd_conv_desc = GET_CONVOLUTION_DESCRIPTOR(conv);

  CHECK_CUDNN(dynload::cudnnGetConvolutionForwardAlgorithm(
      t_resource.cudnn_handle,
      fwd_src_desc,
      fwd_filter_desc,
      fwd_conv_desc,
      fwd_dest_desc,
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      memoryLimitBytes,
      reinterpret_cast<cudnnConvolutionFwdAlgo_t*>(convFwdAlgo)));

  CHECK_CUDNN(dynload::cudnnGetConvolutionForwardWorkspaceSize(
      t_resource.cudnn_handle,
      fwd_src_desc,
      fwd_filter_desc,
      fwd_conv_desc,
      fwd_dest_desc,
      static_cast<cudnnConvolutionFwdAlgo_t>(*convFwdAlgo),
      fwdLimitBytes));

  // cudnn convolution backward data configuration
  cudnnFilterDescriptor_t bwd_data_filter_desc = GET_FILTER_DESCRIPTOR(filter);
  cudnnTensorDescriptor_t bwd_data_diff_desc = GET_TENSOR_DESCRIPTOR(output);
  cudnnTensorDescriptor_t bwd_data_grad_desc = GET_TENSOR_DESCRIPTOR(input);
  cudnnConvolutionDescriptor_t bwd_data_conv_desc =
      GET_CONVOLUTION_DESCRIPTOR(conv);

  CHECK_CUDNN(dynload::cudnnGetConvolutionBackwardDataAlgorithm(
      t_resource.cudnn_handle,
      bwd_data_filter_desc,
      bwd_data_diff_desc,
      bwd_data_conv_desc,
      bwd_data_grad_desc,
      CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
      memoryLimitBytes,
      reinterpret_cast<cudnnConvolutionBwdDataAlgo_t*>(convBwdDataAlgo)));

  CHECK_CUDNN(dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
      t_resource.cudnn_handle,
      bwd_data_filter_desc,
      bwd_data_diff_desc,
      bwd_data_conv_desc,
      bwd_data_grad_desc,
      static_cast<cudnnConvolutionBwdDataAlgo_t>(*convBwdDataAlgo),
      bwdDataLimitBytes));

  // cudnn convolution backward filter configuration
  cudnnTensorDescriptor_t bwd_filter_src_desc = GET_TENSOR_DESCRIPTOR(input);
  cudnnTensorDescriptor_t bwd_filter_diff_desc = GET_TENSOR_DESCRIPTOR(output);
  cudnnConvolutionDescriptor_t bwd_filter_conv_desc =
      GET_CONVOLUTION_DESCRIPTOR(conv);
  cudnnFilterDescriptor_t bwd_filter_grad_desc = GET_FILTER_DESCRIPTOR(filter);

  CHECK_CUDNN(dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
      t_resource.cudnn_handle,
      bwd_filter_src_desc,
      bwd_filter_diff_desc,
      bwd_filter_conv_desc,
      bwd_filter_grad_desc,
      CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
      memoryLimitBytes,
      reinterpret_cast<cudnnConvolutionBwdFilterAlgo_t*>(convBwdFilterAlgo)));

  CHECK_CUDNN(dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
      t_resource.cudnn_handle,
      bwd_filter_src_desc,
      bwd_filter_diff_desc,
      bwd_filter_conv_desc,
      bwd_filter_grad_desc,
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(*convBwdFilterAlgo),
      bwdFilterLimitBytes));

#endif
}

void hl_create_tensor_descriptor(hl_tensor_descriptor* image_desc,
                                 int batch_size,
                                 int feature_maps,
                                 int height,
                                 int width) {
  CHECK_NOTNULL(image_desc);

  cudnn_tensor_descriptor hl_desc =
      (cudnn_tensor_descriptor)malloc(sizeof(_cudnn_tensor_descriptor));
  CHECK_NOTNULL(hl_desc);

#ifndef PADDLE_TYPE_DOUBLE
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
#else
  cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::cudnnCreateTensorDescriptor(&hl_desc->desc));

  CHECK_CUDNN(dynload::cudnnSetTensor4dDescriptor(hl_desc->desc,
                                                  CUDNN_TENSOR_NCHW,
                                                  data_type,
                                                  batch_size,
                                                  feature_maps,
                                                  height,
                                                  width));

  hl_desc->format = CUDNN_TENSOR_NCHW;
  hl_desc->data_type = data_type;
  hl_desc->batch_size = batch_size;
  hl_desc->feature_maps = feature_maps;
  hl_desc->height = height;
  hl_desc->width = width;

  *image_desc = (hl_tensor_descriptor)hl_desc;
}

void hl_create_tensor_descriptor(hl_tensor_descriptor* image_desc) {
  CHECK_NOTNULL(image_desc);

  cudnn_tensor_descriptor hl_desc =
      (cudnn_tensor_descriptor)malloc(sizeof(_cudnn_tensor_descriptor));
  CHECK_NOTNULL(hl_desc);

#ifndef PADDLE_TYPE_DOUBLE
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
#else
  cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::cudnnCreateTensorDescriptor(&hl_desc->desc));

  hl_desc->data_type = data_type;

  *image_desc = (hl_tensor_descriptor)hl_desc;
}

void hl_tensor_reshape(hl_tensor_descriptor image_desc,
                       int batch_size,
                       int feature_maps,
                       int height,
                       int width) {
  const int stride_w = 1;
  const int stride_h = width * stride_w;
  const int stride_c = height * stride_h;
  const int stride_n = feature_maps * stride_c;
  return hl_tensor_reshape(image_desc,
                           batch_size,
                           feature_maps,
                           height,
                           width,
                           stride_n,
                           stride_c,
                           stride_h,
                           stride_w);
}

void hl_tensor_reshape(hl_tensor_descriptor image_desc,
                       int batch_size,
                       int feature_maps,
                       int height,
                       int width,
                       int nStride,
                       int cStride,
                       int hStride,
                       int wStride) {
  CHECK_NOTNULL(image_desc);

  cudnn_tensor_descriptor hl_desc = (cudnn_tensor_descriptor)image_desc;
  CHECK_NOTNULL(hl_desc->desc);

  CHECK_CUDNN(dynload::cudnnSetTensor4dDescriptorEx(hl_desc->desc,
                                                    hl_desc->data_type,
                                                    batch_size,
                                                    feature_maps,
                                                    height,
                                                    width,
                                                    nStride,
                                                    cStride,
                                                    hStride,
                                                    wStride));

  hl_desc->batch_size = batch_size;
  hl_desc->feature_maps = feature_maps;
  hl_desc->height = height;
  hl_desc->width = width;
}

void hl_destroy_tensor_descriptor(hl_tensor_descriptor image_desc) {
  CHECK_NOTNULL(image_desc);

  cudnn_tensor_descriptor hl_desc = (cudnn_tensor_descriptor)image_desc;
  CHECK_NOTNULL(hl_desc->desc);

  CHECK_CUDNN(dynload::cudnnDestroyTensorDescriptor(hl_desc->desc));

  hl_desc->desc = NULL;

  free(image_desc);
}

void hl_create_pooling_descriptor(hl_pooling_descriptor* pooling_desc,
                                  hl_pooling_mode_t mode,
                                  int height,
                                  int width,
                                  int height_padding,
                                  int width_padding,
                                  int stride_height,
                                  int stride_width) {
  cudnnPoolingMode_t cudnn_mode;
  switch (mode) {
    case HL_POOLING_MAX:
      cudnn_mode = CUDNN_POOLING_MAX;
      break;
    case HL_POOLING_AVERAGE:
      cudnn_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      break;
    case HL_POOLING_AVERAGE_EXCLUDE_PADDING:
      cudnn_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
      break;
    default:
      LOG(FATAL) << "parameter mode error";
  }

  CHECK_NOTNULL(pooling_desc);

  cudnn_pooling_descriptor hl_pooling_desc =
      (cudnn_pooling_descriptor)malloc(sizeof(_cudnn_pooling_descriptor));
  CHECK_NOTNULL(hl_pooling_desc);

  CHECK_CUDNN(dynload::cudnnCreatePoolingDescriptor(&hl_pooling_desc->desc));

  CHECK_CUDNN(dynload::cudnnSetPooling2dDescriptor(hl_pooling_desc->desc,
                                                   cudnn_mode,
#if CUDNN_VERSION >= 5000
                                                   CUDNN_PROPAGATE_NAN,
#endif
                                                   height,
                                                   width,
                                                   height_padding,
                                                   width_padding,
                                                   stride_height,
                                                   stride_width));

  hl_pooling_desc->mode = cudnn_mode;
  hl_pooling_desc->window_height = height;
  hl_pooling_desc->window_width = width;
  hl_pooling_desc->stride_height = stride_height;
  hl_pooling_desc->stride_width = stride_width;

  *pooling_desc = (hl_pooling_descriptor)hl_pooling_desc;
}

void hl_destroy_pooling_descriptor(hl_pooling_descriptor pooling_desc) {
  CHECK_NOTNULL(pooling_desc);

  cudnn_pooling_descriptor hl_pooling = (cudnn_pooling_descriptor)pooling_desc;

  CHECK_NOTNULL(hl_pooling->desc);
  CHECK_CUDNN(dynload::cudnnDestroyPoolingDescriptor(hl_pooling->desc));

  hl_pooling->desc = NULL;

  free(pooling_desc);
}

void hl_pooling_forward(hl_tensor_descriptor input,
                        real* input_image,
                        hl_tensor_descriptor output,
                        real* output_image,
                        hl_pooling_descriptor pooling) {
  cudnnPoolingDescriptor_t pooling_desc;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;

  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(pooling);
  CHECK_NOTNULL(input_image);
  CHECK_NOTNULL(output_image);

  real alpha = 1.0f;
  real beta = 1.0f;
  input_desc = ((cudnn_tensor_descriptor)input)->desc;
  output_desc = ((cudnn_tensor_descriptor)output)->desc;
  pooling_desc = ((cudnn_pooling_descriptor)pooling)->desc;
  CHECK_CUDNN(dynload::cudnnPoolingForward(t_resource.cudnn_handle,
                                           pooling_desc,
                                           &alpha,
                                           input_desc,
                                           input_image,
                                           &beta,
                                           output_desc,
                                           output_image));
  CHECK_SYNC("hl_pooling_forward failed");
}

void hl_pooling_backward(hl_tensor_descriptor input,
                         real* input_image,
                         real* input_image_grad,
                         hl_tensor_descriptor output,
                         real* output_image,
                         real* output_image_grad,
                         hl_pooling_descriptor pooling) {
  cudnnPoolingDescriptor_t pooling_desc;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;

  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(pooling);
  CHECK_NOTNULL(input_image);
  CHECK_NOTNULL(input_image_grad);
  CHECK_NOTNULL(output_image);
  CHECK_NOTNULL(output_image_grad);

  real alpha = 1.0f;
  real beta = 1.0f;
  input_desc = ((cudnn_tensor_descriptor)input)->desc;
  output_desc = ((cudnn_tensor_descriptor)output)->desc;
  pooling_desc = ((cudnn_pooling_descriptor)pooling)->desc;
  CHECK_CUDNN(dynload::cudnnPoolingBackward(t_resource.cudnn_handle,
                                            pooling_desc,
                                            &alpha,
                                            output_desc,
                                            output_image,
                                            output_desc,
                                            output_image_grad,
                                            input_desc,
                                            input_image,
                                            &beta,
                                            input_desc,
                                            input_image_grad));
  CHECK_SYNC("hl_pooling_backward failed");
}

void hl_create_filter_descriptor(hl_filter_descriptor* filter,
                                 int input_feature_maps,
                                 int output_feature_maps,
                                 int height,
                                 int width) {
  CHECK_NOTNULL(filter);

  cudnn_filter_descriptor hl_filter =
      (cudnn_filter_descriptor)malloc(sizeof(_cudnn_filter_descriptor));
  CHECK_NOTNULL(hl_filter);

  CHECK_CUDNN(dynload::cudnnCreateFilterDescriptor(&hl_filter->desc));

#ifndef PADDLE_TYPE_DOUBLE
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
#else
  cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::cudnnSetFilter4dDescriptor(hl_filter->desc,
                                                  data_type,
#if CUDNN_VERSION >= 5000
                                                  CUDNN_TENSOR_NCHW,
#endif
                                                  output_feature_maps,
                                                  input_feature_maps,
                                                  height,
                                                  width));

  hl_filter->data_type = data_type;
  hl_filter->output_feature_maps = output_feature_maps;
  hl_filter->input_feature_maps = input_feature_maps;
  hl_filter->filter_height = height;
  hl_filter->filter_width = width;

  *filter = (hl_filter_descriptor)hl_filter;
}

void hl_destroy_filter_descriptor(hl_filter_descriptor filter) {
  CHECK_NOTNULL(filter);

  cudnn_filter_descriptor hl_filter = (cudnn_filter_descriptor)filter;
  CHECK_NOTNULL(hl_filter->desc);

  CHECK_CUDNN(dynload::cudnnDestroyFilterDescriptor(hl_filter->desc));

  hl_filter->desc = NULL;

  free(filter);
}

void hl_create_convolution_descriptor(hl_convolution_descriptor* conv,
                                      hl_tensor_descriptor image,
                                      hl_filter_descriptor filter,
                                      int padding_height,
                                      int padding_width,
                                      int stride_height,
                                      int stride_width) {
  CHECK_NOTNULL(conv);

  cudnn_convolution_descriptor hl_conv = (cudnn_convolution_descriptor)malloc(
      sizeof(_cudnn_convolution_descriptor));

  CHECK_NOTNULL(hl_conv);
  CHECK_CUDNN(dynload::cudnnCreateConvolutionDescriptor(&hl_conv->desc));

  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

#if CUDNN_VERSION >= 6000
#ifndef PADDLE_TYPE_DOUBLE
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
#else
  cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::cudnnSetConvolution2dDescriptor(hl_conv->desc,
                                                       padding_height,
                                                       padding_width,
                                                       stride_height,
                                                       stride_width,
                                                       1,
                                                       1,
                                                       mode,
                                                       data_type));
#else
  CHECK_CUDNN(dynload::cudnnSetConvolution2dDescriptor(hl_conv->desc,
                                                       padding_height,
                                                       padding_width,
                                                       stride_height,
                                                       stride_width,
                                                       1,
                                                       1,
                                                       mode));
#endif

  hl_conv->input_image = image;
  hl_conv->filter = filter;
  hl_conv->padding_height = padding_height;
  hl_conv->padding_width = padding_width;
  hl_conv->stride_height = stride_height;
  hl_conv->stride_width = stride_width;
  hl_conv->upscalex = 1;
  hl_conv->upscaley = 1;
  hl_conv->mode = mode;

  *conv = (hl_convolution_descriptor)hl_conv;
}

void hl_reset_convolution_descriptor(hl_convolution_descriptor conv,
                                     hl_tensor_descriptor image,
                                     hl_filter_descriptor filter,
                                     int padding_height,
                                     int padding_width,
                                     int stride_height,
                                     int stride_width) {
  CHECK_NOTNULL(conv);
  CHECK_NOTNULL(image);
  CHECK_NOTNULL(filter);

  cudnnConvolutionDescriptor_t conv_desc = GET_CONVOLUTION_DESCRIPTOR(conv);
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

#if CUDNN_VERSION >= 6000
#ifndef PADDLE_TYPE_DOUBLE
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
#else
  cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::cudnnSetConvolution2dDescriptor(conv_desc,
                                                       padding_height,
                                                       padding_width,
                                                       stride_height,
                                                       stride_width,
                                                       1,
                                                       1,
                                                       mode,
                                                       data_type));
#else
  CHECK_CUDNN(dynload::cudnnSetConvolution2dDescriptor(conv_desc,
                                                       padding_height,
                                                       padding_width,
                                                       stride_height,
                                                       stride_width,
                                                       1,
                                                       1,
                                                       mode));
#endif

  cudnn_convolution_descriptor hl_conv = (cudnn_convolution_descriptor)conv;
  hl_conv->input_image = image;
  hl_conv->filter = filter;
  hl_conv->padding_height = padding_height;
  hl_conv->padding_width = padding_width;
  hl_conv->stride_height = stride_height;
  hl_conv->stride_width = stride_width;
  hl_conv->upscalex = 1;
  hl_conv->upscaley = 1;
  hl_conv->mode = mode;
}

void hl_destroy_convolution_descriptor(hl_convolution_descriptor conv) {
  CHECK_NOTNULL(conv);

  cudnn_convolution_descriptor hl_conv = (cudnn_convolution_descriptor)conv;
  CHECK_NOTNULL(hl_conv->desc);

  CHECK_CUDNN(dynload::cudnnDestroyConvolutionDescriptor(hl_conv->desc));
  hl_conv->desc = NULL;

  free(conv);
}

void hl_convolution_forward(hl_tensor_descriptor input,
                            real* input_data,
                            hl_tensor_descriptor output,
                            real* output_data,
                            hl_filter_descriptor filter,
                            real* filter_data,
                            hl_convolution_descriptor conv,
                            void* gpuWorkSpace,
                            size_t sizeInBytes,
                            int convFwdAlgo) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(filter);
  CHECK_NOTNULL(conv);
  CHECK_NOTNULL(input_data);
  CHECK_NOTNULL(output_data);
  CHECK_NOTNULL(filter_data);
  cudnnTensorDescriptor_t src_desc = GET_TENSOR_DESCRIPTOR(input);
  cudnnTensorDescriptor_t dest_desc = GET_TENSOR_DESCRIPTOR(output);
  cudnnFilterDescriptor_t filter_desc = GET_FILTER_DESCRIPTOR(filter);
  cudnnConvolutionDescriptor_t conv_desc = GET_CONVOLUTION_DESCRIPTOR(conv);
  real alpha = 1.0f;
  real beta = 1.0f;
  CHECK_CUDNN(dynload::cudnnConvolutionForward(
      t_resource.cudnn_handle,
      &alpha,
      src_desc,
      input_data,
      filter_desc,
      filter_data,
      conv_desc,
      static_cast<cudnnConvolutionFwdAlgo_t>(convFwdAlgo),
      gpuWorkSpace,
      sizeInBytes,
      &beta,
      dest_desc,
      output_data));
  CHECK_SYNC("hl_convolution_forward failed");
}

void hl_convolution_forward_add_bias(hl_tensor_descriptor bias,
                                     real* bias_data,
                                     hl_tensor_descriptor output,
                                     real* output_data) {
  CHECK_NOTNULL(bias);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(bias_data);
  CHECK_NOTNULL(output_data);

  cudnnTensorDescriptor_t output_desc = GET_TENSOR_DESCRIPTOR(output);
  cudnnTensorDescriptor_t bias_desc = GET_TENSOR_DESCRIPTOR(bias);
  real alpha = 1.0f;
  real beta = 1.0f;

  CHECK_CUDNN(dynload::cudnnAddTensor(t_resource.cudnn_handle,
#if CUDNN_VERSION < 4000
                                      CUDNN_ADD_SAME_C,
#endif
                                      &alpha,
                                      bias_desc,
                                      bias_data,
                                      &beta,
                                      output_desc,
                                      output_data));
  CHECK_SYNC("hl_convolution_forward_add_bias failed");
}

void hl_convolution_backward_bias(hl_tensor_descriptor bias,
                                  real* bias_grad_data,
                                  hl_tensor_descriptor output,
                                  real* output_grad_data) {
  CHECK_NOTNULL(bias);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(bias_grad_data);
  CHECK_NOTNULL(output_grad_data);

  real alpha = 1.0f;
  real beta = 1.0f;
  cudnnTensorDescriptor_t diff_desc = GET_TENSOR_DESCRIPTOR(output);
  cudnnTensorDescriptor_t bias_desc = GET_TENSOR_DESCRIPTOR(bias);
  CHECK_CUDNN(dynload::cudnnConvolutionBackwardBias(t_resource.cudnn_handle,
                                                    &alpha,
                                                    diff_desc,
                                                    output_grad_data,
                                                    &beta,
                                                    bias_desc,
                                                    bias_grad_data));
  CHECK_SYNC("hl_convolution_backward_bias failed");
}

void hl_convolution_backward_filter(hl_tensor_descriptor input,
                                    real* input_data,
                                    hl_tensor_descriptor output,
                                    real* output_grad_data,
                                    hl_filter_descriptor filter,
                                    real* filter_grad_data,
                                    hl_convolution_descriptor conv,
                                    void* gpuWorkSpace,
                                    size_t sizeInBytes,
                                    int convBwdFilterAlgo) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(filter);
  CHECK_NOTNULL(conv);
  CHECK_NOTNULL(input_data);
  CHECK_NOTNULL(output_grad_data);
  CHECK_NOTNULL(filter_grad_data);

  real alpha = 1.0f;
  real beta = 1.0f;
  cudnnTensorDescriptor_t src_desc = GET_TENSOR_DESCRIPTOR(input);
  cudnnTensorDescriptor_t diff_desc = GET_TENSOR_DESCRIPTOR(output);
  cudnnConvolutionDescriptor_t conv_desc = GET_CONVOLUTION_DESCRIPTOR(conv);
  cudnnFilterDescriptor_t grad_desc = GET_FILTER_DESCRIPTOR(filter);

  CHECK_CUDNN(dynload::cudnnConvolutionBackwardFilter(
      t_resource.cudnn_handle,
      &alpha,
      src_desc,
      input_data,
      diff_desc,
      output_grad_data,
      conv_desc,
#if CUDNN_VERSION >= 4000
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(convBwdFilterAlgo),
      gpuWorkSpace,
      sizeInBytes,
#endif
      &beta,
      grad_desc,
      filter_grad_data));
  CHECK_SYNC("hl_convolution_backward_filter failed");
}

void hl_convolution_backward_data(hl_tensor_descriptor input,
                                  real* input_data_grad,
                                  hl_tensor_descriptor output,
                                  real* output_grad_data,
                                  hl_filter_descriptor filter,
                                  real* filter_data,
                                  hl_convolution_descriptor conv,
                                  void* gpuWorkSpace,
                                  size_t sizeInBytes,
                                  int convBwdDataAlgo) {
  real alpha = 1.0f;
  real beta = 1.0f;
  cudnnFilterDescriptor_t filter_desc = GET_FILTER_DESCRIPTOR(filter);
  cudnnTensorDescriptor_t diff_desc = GET_TENSOR_DESCRIPTOR(output);
  cudnnTensorDescriptor_t grad_desc = GET_TENSOR_DESCRIPTOR(input);
  cudnnConvolutionDescriptor_t conv_desc = GET_CONVOLUTION_DESCRIPTOR(conv);

  CHECK_CUDNN(dynload::cudnnConvolutionBackwardData(
      t_resource.cudnn_handle,
      &alpha,
      filter_desc,
      filter_data,
      diff_desc,
      output_grad_data,
      conv_desc,
#if CUDNN_VERSION >= 4000
      static_cast<cudnnConvolutionBwdDataAlgo_t>(convBwdDataAlgo),
      gpuWorkSpace,
      sizeInBytes,
#endif
      &beta,
      grad_desc,
      input_data_grad));
  CHECK_SYNC("hl_convolution_backward_data failed");
}

void hl_softmax_forward(real* input, real* output, int height, int width) {
#ifndef PADDLE_TYPE_DOUBLE
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
#else
  cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::cudnnSetTensor4dDescriptor(t_resource.cudnn_desc,
                                                  CUDNN_TENSOR_NCHW,
                                                  data_type,
                                                  height,
                                                  width,
                                                  1,
                                                  1));

  real alpha = 1.0f;
  real beta = 0.0f;
  CHECK_CUDNN(dynload::cudnnSoftmaxForward(t_resource.cudnn_handle,
                                           CUDNN_SOFTMAX_ACCURATE,
                                           CUDNN_SOFTMAX_MODE_CHANNEL,
                                           &alpha,
                                           t_resource.cudnn_desc,
                                           input,
                                           &beta,
                                           t_resource.cudnn_desc,
                                           output));
  CHECK_SYNC("hl_softmax_forward failed");
}

void hl_softmax_backward(real* output_value,
                         real* output_grad,
                         int height,
                         int width) {
#ifndef PADDLE_TYPE_DOUBLE
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
#else
  cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::cudnnSetTensor4dDescriptor(t_resource.cudnn_desc,
                                                  CUDNN_TENSOR_NCHW,
                                                  data_type,
                                                  height,
                                                  width,
                                                  1,
                                                  1));

  real alpha = 1.0f;
  real beta = 0.0f;
  CHECK_CUDNN(dynload::cudnnSoftmaxBackward(t_resource.cudnn_handle,
                                            CUDNN_SOFTMAX_ACCURATE,
                                            CUDNN_SOFTMAX_MODE_CHANNEL,
                                            &alpha,
                                            t_resource.cudnn_desc,
                                            output_value,
                                            t_resource.cudnn_desc,
                                            output_grad,
                                            &beta,
                                            t_resource.cudnn_desc,
                                            output_grad));
  CHECK_SYNC("hl_softmax_backward failed");
}

void hl_batch_norm_forward_training(hl_tensor_descriptor inputDesc,
                                    real* input,
                                    hl_tensor_descriptor outputDesc,
                                    real* output,
                                    hl_tensor_descriptor bnParamDesc,
                                    real* scale,
                                    real* bias,
                                    double factor,
                                    real* runningMean,
                                    real* runningInvVar,
                                    double epsilon,
                                    real* savedMean,
                                    real* savedVar) {
#if CUDNN_VERSION >= 4007
  if ((NULL != runningMean && NULL == runningInvVar) ||
      (NULL == runningMean && NULL != runningInvVar)) {
    LOG(FATAL) << "runningMean and runningInvVar can be NULL "
               << "but only at the same time.";
  }
  if ((NULL != savedMean && NULL == savedVar) ||
      (NULL == savedMean && NULL != savedVar)) {
    LOG(FATAL) << "savedMean and savedVar can be NULL "
               << "but only at the same time.";
  }

  cudnnTensorDescriptor_t xDesc = GET_TENSOR_DESCRIPTOR(inputDesc);
  cudnnTensorDescriptor_t yDesc = GET_TENSOR_DESCRIPTOR(outputDesc);
  cudnnTensorDescriptor_t bnDesc = GET_TENSOR_DESCRIPTOR(bnParamDesc);
  real alpha = 1.0f;
  real beta = 1.0f;
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
  CHECK_CUDNN(
      dynload::cudnnBatchNormalizationForwardTraining(t_resource.cudnn_handle,
                                                      mode,
                                                      &alpha,
                                                      &beta,
                                                      xDesc,
                                                      input,
                                                      yDesc,
                                                      output,
                                                      bnDesc,
                                                      scale,
                                                      bias,
                                                      factor,
                                                      runningMean,
                                                      runningInvVar,
                                                      epsilon,
                                                      savedMean,
                                                      savedVar));

  CHECK_SYNC("hl_batch_norm_forward_training failed");
#else
  LOG(FATAL) << "CudnnBatchNorm requires cudnn version >= 4007. "
             << "But cudnn lib version is " << g_cudnn_lib_version;
#endif
}

void hl_batch_norm_forward_inference(hl_tensor_descriptor inputDesc,
                                     real* input,
                                     hl_tensor_descriptor outputDesc,
                                     real* output,
                                     hl_tensor_descriptor bnParamDesc,
                                     real* scale,
                                     real* bias,
                                     real* estimatedMean,
                                     real* estimatedInvVar,
                                     double epsilon) {
#if CUDNN_VERSION >= 4007
  cudnnTensorDescriptor_t xDesc = GET_TENSOR_DESCRIPTOR(inputDesc);
  cudnnTensorDescriptor_t yDesc = GET_TENSOR_DESCRIPTOR(outputDesc);
  cudnnTensorDescriptor_t bnDesc = GET_TENSOR_DESCRIPTOR(bnParamDesc);
  real alpha = 1.0f;
  real beta = 1.0f;
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
  CHECK_CUDNN(
      dynload::cudnnBatchNormalizationForwardInference(t_resource.cudnn_handle,
                                                       mode,
                                                       &alpha,
                                                       &beta,
                                                       xDesc,
                                                       input,
                                                       yDesc,
                                                       output,
                                                       bnDesc,
                                                       scale,
                                                       bias,
                                                       estimatedMean,
                                                       estimatedInvVar,
                                                       epsilon));

  CHECK_SYNC("hl_batch_norm_forward_inference failed");
#else
  LOG(FATAL) << "CudnnBatchNorm requires cudnn version >= 4007. "
             << "But cudnn lib version is " << g_cudnn_lib_version;
#endif
}

void hl_batch_norm_backward(hl_tensor_descriptor inputDesc,
                            real* input,
                            hl_tensor_descriptor outGradDesc,
                            real* outGrad,
                            hl_tensor_descriptor inGradDesc,
                            real* inGrad,
                            hl_tensor_descriptor dBnParamDesc,
                            real* scale,
                            real* scaleGrad,
                            real* biasGrad,
                            double epsilon,
                            real* savedMean,
                            real* savedInvVar) {
#if CUDNN_VERSION >= 4007
  if ((NULL != savedMean && NULL == savedInvVar) ||
      (NULL == savedMean && NULL != savedInvVar)) {
    LOG(FATAL) << "savedMean and savedVar can be NULL "
               << "but only at the same time.";
  }

  cudnnTensorDescriptor_t xDesc = GET_TENSOR_DESCRIPTOR(inputDesc);
  cudnnTensorDescriptor_t dyDesc = GET_TENSOR_DESCRIPTOR(outGradDesc);
  cudnnTensorDescriptor_t dxDesc = GET_TENSOR_DESCRIPTOR(inGradDesc);
  cudnnTensorDescriptor_t bnDesc = GET_TENSOR_DESCRIPTOR(dBnParamDesc);
  real alpha = 1.0f;
  real beta = 1.0f;
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
  CHECK_CUDNN(dynload::cudnnBatchNormalizationBackward(t_resource.cudnn_handle,
                                                       mode,
                                                       &alpha,
                                                       &beta,
                                                       &alpha,
                                                       &beta,
                                                       xDesc,
                                                       input,
                                                       dyDesc,
                                                       outGrad,
                                                       dxDesc,
                                                       inGrad,
                                                       bnDesc,
                                                       scale,
                                                       scaleGrad,
                                                       biasGrad,
                                                       epsilon,
                                                       savedMean,
                                                       savedInvVar));

  CHECK_SYNC("hl_batch_norm_backward failed");
#else
  LOG(FATAL) << "CudnnBatchNorm requires cudnn version >= 4007. "
             << "But cudnn lib version is " << g_cudnn_lib_version;
#endif
}
