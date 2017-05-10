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

#ifndef HL_CUDA_CUDNN_H_
#define HL_CUDA_CUDNN_H_

#include "hl_base.h"

/*
 *  hppl pooling mode
 */
typedef enum {
  HL_POOLING_MAX = 0,
  // average includes padded values
  HL_POOLING_AVERAGE = 1,
  // average does not include padded values
  HL_POOLING_AVERAGE_EXCLUDE_PADDING = 2,
  HL_POOLING_END
} hl_pooling_mode_t;

/**
 * @brief return cudnn lib version
 */

extern int hl_get_cudnn_lib_version();

/**
 * @brief   hppl image descriptor.
 */
typedef struct _hl_tensor_descriptor* hl_tensor_descriptor;

/**
 * @brief   hppl pooling descriptor.
 */
typedef struct _hl_pooling_descriptor* hl_pooling_descriptor;

/**
 * @brief   hppl filter descriptor.
 */
typedef struct _hl_filter_descriptor* hl_filter_descriptor;

/**
 * @brief   hppl filter descriptor.
 */
typedef struct _hl_convolution_descriptor* hl_convolution_descriptor;

/**
 * @brief   create image descriptor.
 *
 * @param[out]   image_desc     image descriptor.
 *
 */
extern void hl_create_tensor_descriptor(hl_tensor_descriptor* image_desc);

/**
 * @brief   reshape image descriptor.
 *
 * @param[in,out]   image_desc    image descriptor.
 * @param[in]       batch_size    input batch size.
 * @param[in]       feature_maps  image feature maps.
 * @param[in]       height        image height.
 * @param[in]       width         image width.
 */
extern void hl_tensor_reshape(hl_tensor_descriptor image_desc,
                              int batch_size,
                              int feature_maps,
                              int height,
                              int width);

/**
 * @brief   reshape image descriptor.
 *
 * @param[in,out]   image_desc    image descriptor.
 * @param[in]       batch_size    input batch size.
 * @param[in]       feature_maps  image feature maps.
 * @param[in]       height        image height.
 * @param[in]       width         image width.
 * @param[in]       nStride       stride between two consecutive images.
 * @param[in]       cStride       stride between two consecutive feature maps.
 * @param[in]       hStride       stride between two consecutive rows.
 * @param[in]       wStride       stride between two consecutive columns.
 *
 */
extern void hl_tensor_reshape(hl_tensor_descriptor image_desc,
                              int batch_size,
                              int feature_maps,
                              int height,
                              int width,
                              int nStride,
                              int cStride,
                              int hStride,
                              int wStride);

/**
 * @brief   destroy image descriptor.
 *
 * @param[in]   image_desc  hppl image descriptor.
 */
extern void hl_destroy_tensor_descriptor(hl_tensor_descriptor image_desc);

/**
 * @brief   create pooling descriptor.
 *
 * @param[out]  pooling_desc    pooling descriptor.
 * @param[in]   mode            pooling mode.
 * @param[in]   height          height of the pooling window.
 * @param[in]   width           width of the pooling window.
 * @param[in]   height_padding  padding height.
 * @param[in]   width_padding   padding width.
 * @param[in]   stride_height   pooling vertical stride.
 * @param[in]   stride_width    pooling horizontal stride.
 */
extern void hl_create_pooling_descriptor(hl_pooling_descriptor* pooling_desc,
                                         hl_pooling_mode_t mode,
                                         int height,
                                         int width,
                                         int height_padding,
                                         int width_padding,
                                         int stride_height,
                                         int stride_width);

/**
 * @brief   destroy pooling descriptor.
 *
 * @param[in]   pooling_desc  hppl pooling descriptor.
 *
 */
extern void hl_destroy_pooling_descriptor(hl_pooling_descriptor pooling_desc);

/**
 * @brief   pooling forward(calculate output image).
 *
 * @param[in]   input           input image descriptor.
 * @param[in]   input_image     input image data.
 * @param[in]   output          output image descriptor.
 * @param[out]  output_image    output image data.
 * @param[in]   pooling         pooling descriptor.
 *
 */
extern void hl_pooling_forward(hl_tensor_descriptor input,
                               real* input_image,
                               hl_tensor_descriptor output,
                               real* output_image,
                               hl_pooling_descriptor pooling);

/**
 * @brief   pooling backward(calculate input image gradient).
 *
 * @param[in]   input               input image descriptor.
 * @param[in]   input_image         input image data.
 * @param[in]   input_image_grad    input image gradient data.
 * @param[in]   output              output image descriptor.
 * @param[in]   output_image        output image data.
 * @param[out]  output_image_grad   output image gradient data.
 * @param[in]   pooling             pooling descriptor.
 *
 */
extern void hl_pooling_backward(hl_tensor_descriptor input,
                                real* input_image,
                                real* input_image_grad,
                                hl_tensor_descriptor output,
                                real* output_image,
                                real* output_image_grad,
                                hl_pooling_descriptor pooling);

/**
 * @brief   create filter descriptor.
 *
 * @param[out]  filter                  filter descriptor.
 * @param[in]   input_feature_maps      input image feature maps.
 * @param[in]   output_feature_maps     output image feature maps.
 * @param[in]   height                  filter height.
 * @param[in]   width                   filter width.
 *
 */
extern void hl_create_filter_descriptor(hl_filter_descriptor* filter,
                                        int input_feature_maps,
                                        int output_feature_maps,
                                        int height,
                                        int width);

/**
 * @brief    convolution workspace configuration
 *
 * @param[in]    input                image descriptor
 * @param[in]    output               image descriptor
 * @param[in]    filter               filter descriptor
 * @param[in]    conv                 convolution descriptor
 * @param[out]   convFwdAlgo          forward algorithm
 * @param[out]   fwdLimitBytes        forward workspace size
 * @param[out]   convBwdDataAlgo      backward data algorithm
 * @param[out]   bwdDataLimitBytes    backward data workspace size
 * @param[out]   convBwdFilterAlgo    backward filter algorithm
 * @param[out]   bwdFilterLimitBytes  backward filter workspace size
 *
 */
extern void hl_conv_workspace(hl_tensor_descriptor input,
                              hl_tensor_descriptor output,
                              hl_filter_descriptor filter,
                              hl_convolution_descriptor conv,
                              int* convFwdAlgo,
                              size_t* fwdLimitBytes,
                              int* convBwdDataAlgo,
                              size_t* bwdDataLimitBytes,
                              int* convBwdFilterAlgo,
                              size_t* bwdFilterLimitBytes);

/**
 * @brief   destroy filter descriptor.
 *
 * @param[in]   filter  hppl filter descriptor.
 *
 */
extern void hl_destroy_filter_descriptor(hl_filter_descriptor filter);

/**
 * @brief   create convolution descriptor.
 *
 * @param[out]  conv                    conv descriptor.
 * @param[in]   image                   input image descriptor.
 * @param[in]   filter                  filter descriptor.
 * @param[in]   padding_height          padding height.
 * @param[in]   padding_width           padding width.
 * @param[in]   stride_height           stride height.
 * @param[in]   stride_width            stride width.
 *
 */
extern void hl_create_convolution_descriptor(hl_convolution_descriptor* conv,
                                             hl_tensor_descriptor image,
                                             hl_filter_descriptor filter,
                                             int padding_height,
                                             int padding_width,
                                             int stride_height,
                                             int stride_width);

/**
 * @brief   reset convolution descriptor.
 *
 * @param[in,out]   conv                conv descriptor.
 * @param[in]       image               input image descriptor.
 * @param[in]       filter              filter descriptor.
 * @param[in]       padding_height      padding height.
 * @param[in]       padding_width       padding width.
 * @param[in]       stride_height       stride height.
 * @param[in]       stride_width        stride width.
 *
 */
extern void hl_reset_convolution_descriptor(hl_convolution_descriptor conv,
                                            hl_tensor_descriptor image,
                                            hl_filter_descriptor filter,
                                            int padding_height,
                                            int padding_width,
                                            int stride_height,
                                            int stride_width);

/**
 * @brief   destroy convolution descriptor.
 *
 * @param[in]   conv  hppl convolution descriptor.
 */
extern void hl_destroy_convolution_descriptor(hl_convolution_descriptor conv);

/**
 * @brief   convolution forward(calculate output image).
 *
 * @param[in]   input           input image descriptor.
 * @param[in]   input_data      input image data.
 * @param[in]   output          output image descriptor.
 * @param[out]  output_data     output image data.
 * @param[in]   filter          filter descriptor.
 * @param[in]   filter_data     filter data.
 * @param[in]   conv            convolution descriptor.
 * @param[in]   gpuWorkSpace    limited gpu workspace.
 * @param[in]   sizeInBytes     gpu workspace size (bytes).
 * @param[in]   convFwdAlgo     forward algorithm.
 */
extern void hl_convolution_forward(hl_tensor_descriptor input,
                                   real* input_data,
                                   hl_tensor_descriptor output,
                                   real* output_data,
                                   hl_filter_descriptor filter,
                                   real* filter_data,
                                   hl_convolution_descriptor conv,
                                   void* gpuWorkSpace,
                                   size_t sizeInBytes,
                                   int convFwdAlgo);

/**
 * @brief   convolution forward add bias(calculate output add bias).
 *
 * @param[in]   bias                bias descriptor.
 * @param[in]   bias_data           bias data.
 * @param[in]   output              output image descriptor.
 * @param[out]  output_data         output image data.
 */
extern void hl_convolution_forward_add_bias(hl_tensor_descriptor bias,
                                            real* bias_data,
                                            hl_tensor_descriptor output,
                                            real* output_data);

/**
 * @brief   convolution backward filter(calculate filter grad data).
 *
 * @param[in]   input               input image descriptor.
 * @param[in]   input_data          input image data.
 * @param[in]   output              output image descriptor.
 * @param[in]   output_grad_data    output image grad data.
 * @param[in]   filter              filter descriptor.
 * @param[out]  filter_grad_data    filter grad data.
 * @param[in]   conv                convolution descriptor.
 * @param[in]   gpuWorkSpace        limited gpu workspace.
 * @param[in]   sizeInBytes         gpu workspace size (bytes).
 * @param[in]   convBwdFilterAlgo   backward filter algorithm.
 */
extern void hl_convolution_backward_filter(hl_tensor_descriptor input,
                                           real* input_data,
                                           hl_tensor_descriptor output,
                                           real* output_grad_data,
                                           hl_filter_descriptor filter,
                                           real* filter_grad_data,
                                           hl_convolution_descriptor conv,
                                           void* gpuWorkSpace,
                                           size_t sizeInBytes,
                                           int convBwdFilterAlgo);

/**
 * @brief   convolution backward data(calculate input image grad data).
 *
 * @param[in]   input               input image descriptor.
 * @param[out]  input_data_grad     input image grad data.
 * @param[in]   output              output image descriptor.
 * @param[in]   output_grad_data    output image grad data.
 * @param[in]   filter              filter descriptor.
 * @param[in]   filter_data         filter data.
 * @param[in]   conv                convolution descriptor.
 * @param[in]   gpuWorkSpace        limited gpu workspace.
 * @param[in]   sizeInBytes         gpu workspace size (bytes).
 * @param[in]   convBwdDataAlgo     backward data algorithm.
 */
extern void hl_convolution_backward_data(hl_tensor_descriptor input,
                                         real* input_data_grad,
                                         hl_tensor_descriptor output,
                                         real* output_grad_data,
                                         hl_filter_descriptor filter,
                                         real* filter_data,
                                         hl_convolution_descriptor conv,
                                         void* gpuWorkSpace,
                                         size_t sizeInBytes,
                                         int convBwdDataAlgo);

/**
 * @brief   convolution backward bias(calculate bias grad data).
 *
 * @param[in]   bias                bias descriptor.
 * @param[out]  bias_grad_data      bias grad data.
 * @param[in]   output              output image descriptor.
 * @param[in]   output_grad_data    output image grad data.
 */
extern void hl_convolution_backward_bias(hl_tensor_descriptor bias,
                                         real* bias_grad_data,
                                         hl_tensor_descriptor output,
                                         real* output_grad_data);

/**
 * @brief   softmax forward.
 *
 * @param[in]   input               input value.
 * @param[out]  output              output value.
 * @param[in]   height              matrix height.
 * @param[in]   width               matrix width.
 */
extern void hl_softmax_forward(real* input,
                               real* output,
                               int height,
                               int width);

/**
 * @brief   softmax backward.
 *
 * @param[in]   output_value        output value data.
 * @param[out]  output_grad         output grad data.
 * @param[in]   height              matrix height.
 * @param[in]   width               matrix width.
 */
extern void hl_softmax_backward(real* output_value,
                                real* output_grad,
                                int height,
                                int width);

/**
 * @brief   cudnn batch norm forward.
 *
 * @param[in]   inputDesc     input tensor descriptor desc.
 * @param[in]   input         input data.
 * @param[in]   outputDesc    output tensor descriptor desc.
 * @param[out]  output        output data.
 * @param[in]   bnParamDesc   tensor descriptor desc.
 *                            bnScale, bnBias, running mean/var, save_mean/var.
 * @param[in]   scale         batch normalization scale parameter (in original
 *                            paper scale is referred to as gamma).
 * @param[in]   bias          batch normalization bias parameter (in original
 *                            paper scale is referred to as beta).
 * @param[in]   factor        Factor used in the moving average computation.
 *                            runningMean = newMean * factor
 *                                         + runningMean * (1 - factor)
 * @param[in]   runningMean   running mean.
 * @param[in]   runningInvVar running variance.
 * @param[in]   epsilon       Epsilon value used in the batch normalization
 *                            formula.
 * @param[out]  savedMean     optional cache to save intermediate results.
 * @param[out]  savedVar      optional cache to save intermediate results.
 *
 */
extern void hl_batch_norm_forward_training(hl_tensor_descriptor inputDesc,
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
                                           real* savedVar);

/**
 * @brief   cudnn batch norm forward.
 *
 * @param[in]   inputDesc    input tensor descriptor desc.
 * @param[in]   input        input data.
 * @param[in]   outputDesc   output tensor descriptor desc.
 * @param[out]  output       output data.
 * @param[in]   bnParamDesc  tensor descriptor desc.
 *                           bnScale, bnBias, running mean/var, save_mean/var.
 * @param[in]   scale        batch normalization scale parameter (in original
 *                           paper scale is referred to as gamma).
 * @param[in]   bias         batch normalization bias parameter (in original
 *                           paper scale is referred to as beta).
 * @param[in]   estimatedMean
 * @param[in]   estimatedVar It is suggested that resultRunningMean,
 *                           resultRunningVariance from the
 *                           cudnnBatchNormalizationForwardTraining call
 *                           accumulated during the training phase are passed
 *                           as inputs here.
 * @param[in]   epsilon      Epsilon value used in the batch
 *                           normalization formula.
 *
 */
extern void hl_batch_norm_forward_inference(hl_tensor_descriptor inputDesc,
                                            real* input,
                                            hl_tensor_descriptor outputDesc,
                                            real* output,
                                            hl_tensor_descriptor bnParamDesc,
                                            real* scale,
                                            real* bias,
                                            real* estimatedMean,
                                            real* estimatedVar,
                                            double epsilon);

/**
 * @brief   cudnn batch norm forward.
 *
 * @param[in]   inputDesc       input tensor descriptor desc.
 * @param[in]   input           input data.
 * @param[in]   outGradDesc     output tensor descriptor desc.
 * @param[out]  outGrad         output data.
 * @param[in]   inGradDesc      input tensor descriptor desc.
 * @param[in]   inGrad          input data.
 * @param[in]   dBnParamDesc    tensor descriptor desc.
 *                              bnScale, bnBias, running mean/var,
 * save_mean/var.
 * @param[in]   scale           batch normalization scale parameter (in original
 *                              paper scale is referred to as gamma).
 * @param[in]   scaleGrad       batch normalization scale parameter (in original
 *                              paper scale is referred to as gamma) gradient.
 * @param[in]   biasGrad        batch normalization bias parameter (in original
 *                              paper scale is referred to as beta) gradient.
 * @param[in]   epsilon         Epsilon value used in the batch
 *                              normalization formula.
 * @param[out]  savedMean       optional cache to save intermediate results.
 * @param[out]  savedInvVar     optional cache to save intermediate results.
 *
 */
extern void hl_batch_norm_backward(hl_tensor_descriptor inputDesc,
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
                                   real* savedInvVar);

#endif  // HL_CUDA_CUDNN_H_
