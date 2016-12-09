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

#ifndef HL_CUDA_CUDNN_PH_
#define HL_CUDA_CUDNN_PH_

#include "hl_base.h"

/*
 * @brief   hppl for cudnn tensor4d descriptor.
 */
typedef struct {
    cudnnTensorDescriptor_t     desc;
    cudnnTensorFormat_t         format;
    cudnnDataType_t             data_type;  // image data type
    int batch_size;                         // number of input batch size
    int feature_maps;                       // number of input feature maps
    int height;                             // height of input image
    int width;                              // width of input image
} _cudnn_tensor_descriptor, *cudnn_tensor_descriptor;

#define GET_TENSOR_DESCRIPTOR(image) (((cudnn_tensor_descriptor)image)->desc)

/*
 * @brief   hppl for cudnn pooling descriptor.
 */
typedef struct {
    cudnnPoolingDescriptor_t   desc;
    cudnnPoolingMode_t         mode;
    int window_height;
    int window_width;
    int stride_height;
    int stride_width;
} _cudnn_pooling_descriptor, *cudnn_pooling_descriptor;

/*
 * @brief   hppl for cudnn filter descriptor.
 */
typedef struct {
    cudnnFilterDescriptor_t   desc;
    cudnnDataType_t           data_type;    /* data type */
    int output_feature_maps;        /* number of output feature maps */
    int input_feature_maps;         /* number of input feature maps */
    int filter_height;              /* height of each input filter */
    int filter_width;               /* width of  each input fitler */
} _cudnn_filter_descriptor, *cudnn_filter_descriptor;

#define GET_FILTER_DESCRIPTOR(filter) (((cudnn_filter_descriptor)filter)->desc)

/*
 * @brief   hppl for cudnn convolution descriptor.
 */
typedef struct {
    cudnnConvolutionDescriptor_t    desc;
    hl_tensor_descriptor             input_image;
    hl_filter_descriptor            filter;
    int padding_height;                     // zero-padding height
    int padding_width;                      // zero-padding width
    int stride_height;                      // vertical filter stride
    int stride_width;                       // horizontal filter stride
    int upscalex;                           // upscale the input in x-direction
    int upscaley;                           // upscale the input in y-direction
    cudnnConvolutionMode_t          mode;
} _cudnn_convolution_descriptor, *cudnn_convolution_descriptor;

#define GET_CONVOLUTION_DESCRIPTOR(conv)    \
    (((cudnn_convolution_descriptor)conv)->desc)

#endif /* HL_CUDA_CUDNN_PH_ */
