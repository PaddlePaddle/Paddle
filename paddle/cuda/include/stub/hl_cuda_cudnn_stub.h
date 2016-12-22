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

#ifndef HL_CUDA_CUDNN_STUB_H_
#define HL_CUDA_CUDNN_STUB_H_

#include "hl_cuda_cudnn.h"

inline int hl_get_cudnn_lib_version() { return 0; }

inline void hl_create_tensor_descriptor(hl_tensor_descriptor* image_desc) {}

inline void hl_tensor_reshape(hl_tensor_descriptor image_desc,
                              int batch_size,
                              int feature_maps,
                              int height,
                              int width) {}

inline void hl_tensor_reshape(hl_tensor_descriptor image_desc,
                              int batch_size,
                              int feature_maps,
                              int height,
                              int width,
                              int nStride,
                              int cStride,
                              int hStride,
                              int wStride) {}

inline void hl_destroy_tensor_descriptor(hl_tensor_descriptor image_desc) {}

inline void hl_create_pooling_descriptor(hl_pooling_descriptor* pooling_desc,
                                         hl_pooling_mode_t mode,
                                         int height,
                                         int width,
                                         int height_padding,
                                         int width_padding,
                                         int stride_height,
                                         int stride_width) {}

inline void hl_destroy_pooling_descriptor(hl_pooling_descriptor pooling_desc) {}

inline void hl_pooling_forward(hl_tensor_descriptor input,
                               real* input_image,
                               hl_tensor_descriptor output,
                               real* output_image,
                               hl_pooling_descriptor pooling) {}

inline void hl_pooling_backward(hl_tensor_descriptor input,
                                real* input_image,
                                real* input_image_grad,
                                hl_tensor_descriptor output,
                                real* output_image,
                                real* output_image_grad,
                                hl_pooling_descriptor pooling) {}

inline void hl_create_filter_descriptor(hl_filter_descriptor* filter,
                                        int input_feature_maps,
                                        int output_feature_maps,
                                        int height,
                                        int width) {}

inline void hl_destroy_filter_descriptor(hl_filter_descriptor filter) {}

inline void hl_create_convolution_descriptor(hl_convolution_descriptor* conv,
                                             hl_tensor_descriptor image,
                                             hl_filter_descriptor filter,
                                             int padding_height,
                                             int padding_width,
                                             int stride_height,
                                             int stride_width) {}

inline void hl_reset_convolution_descriptor(hl_convolution_descriptor conv,
                                            hl_tensor_descriptor image,
                                            hl_filter_descriptor filter,
                                            int padding_height,
                                            int padding_width,
                                            int stride_height,
                                            int stride_width) {}

inline void hl_destroy_convolution_descriptor(hl_convolution_descriptor conv) {}

inline void hl_conv_workspace(hl_tensor_descriptor input,
                              hl_tensor_descriptor output,
                              hl_filter_descriptor filter,
                              hl_convolution_descriptor conv,
                              int* convFwdAlgo,
                              size_t* fwdLimitBytes,
                              int* convBwdDataAlgo,
                              size_t* bwdDataLimitBytes,
                              int* convBwdFilterAlgo,
                              size_t* bwdFilterLimitBytes) {}

inline void hl_convolution_forward(hl_tensor_descriptor input,
                                   real* input_data,
                                   hl_tensor_descriptor output,
                                   real* output_data,
                                   hl_filter_descriptor filter,
                                   real* filter_data,
                                   hl_convolution_descriptor conv,
                                   void* gpuWorkSpace,
                                   size_t sizeInBytes,
                                   int convFwdAlgo) {}

inline void hl_convolution_forward_add_bias(hl_tensor_descriptor bias,
                                            real* bias_data,
                                            hl_tensor_descriptor output,
                                            real* output_data) {}

inline void hl_convolution_backward_filter(hl_tensor_descriptor input,
                                           real* input_data,
                                           hl_tensor_descriptor output,
                                           real* output_grad_data,
                                           hl_filter_descriptor filter,
                                           real* filter_grad_data,
                                           hl_convolution_descriptor conv,
                                           void* gpuWorkSpace,
                                           size_t sizeInBytes,
                                           int convBwdFilterAlgo) {}

inline void hl_convolution_backward_data(hl_tensor_descriptor input,
                                         real* input_data_grad,
                                         hl_tensor_descriptor output,
                                         real* output_grad_data,
                                         hl_filter_descriptor filter,
                                         real* filter_data,
                                         hl_convolution_descriptor conv,
                                         void* gpuWorkSpace,
                                         size_t sizeInBytes,
                                         int convBwdDataAlgo) {}

inline void hl_convolution_backward_bias(hl_tensor_descriptor bias,
                                         real* bias_grad_data,
                                         hl_tensor_descriptor output,
                                         real* output_grad_data) {}

inline void hl_softmax_forward(real* input,
                               real* output,
                               int height,
                               int width) {}

inline void hl_softmax_backward(real* output_value,
                                real* output_grad,
                                int height,
                                int width) {}

inline void hl_batch_norm_forward_training(hl_tensor_descriptor inputDesc,
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
                                           real* savedVar) {}

inline void hl_batch_norm_forward_inference(hl_tensor_descriptor inputDesc,
                                            real* input,
                                            hl_tensor_descriptor outputDesc,
                                            real* output,
                                            hl_tensor_descriptor bnParamDesc,
                                            real* scale,
                                            real* bias,
                                            real* estimatedMean,
                                            real* estimatedVar,
                                            double epsilon) {}

inline void hl_batch_norm_backward(hl_tensor_descriptor inputDesc,
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
                                   real* savedInvVar) {}

#endif  // HL_CUDA_CUDNN_STUB_H_
