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

#ifndef HL_CNN_H_
#define HL_CNN_H_

#include "hl_base.h"

/**
 * @brief   Shrink column to feature.
 *
 * @param[in]   dataCol     expand data.
 * @param[in]   channels    number of channel.
 * @param[in]   height      image height.
 * @param[in]   width       image width.
 * @param[in]   blockH      filter height.
 * @param[in]   blockW      filter width.
 * @param[in]   strideH     stride height.
 * @param[in]   strideW     stride width.
 * @param[in]   paddingH    padding height.
 * @param[in]   paddingW    padding width.
 * @param[in]   outputH     output height.
 * @param[in]   outputW     output width.
 * @param[out]  dataIm      output image data.
 * @param[in]   alpha
 * @param[in]   beta
 */
extern void hl_shrink_col2feature(const real* dataCol,
                                  size_t channels,
                                  size_t height,
                                  size_t width,
                                  size_t blockH,
                                  size_t blockW,
                                  size_t strideH,
                                  size_t strideW,
                                  size_t paddingH,
                                  size_t paddingW,
                                  size_t outputH,
                                  size_t outputW,
                                  real* dataIm,
                                  real alpha = 1.0f,
                                  real beta = 0.0f);

/**
 * @brief   Expand feature to column.
 *
 * @param[in]   dataIm      input image data.
 * @param[in]   channels    number of channel.
 * @param[in]   height      image height.
 * @param[in]   width       image width.
 * @param[in]   blockH      filter height.
 * @param[in]   blockW      filter width.
 * @param[in]   strideH     stride height.
 * @param[in]   strideW     stride width.
 * @param[in]   paddingH    padding height.
 * @param[in]   paddingW    padding width.
 * @param[in]   outputH     output height.
 * @param[in]   outputW     output width.
 * @param[out]  dataCol     expand data.
 *
 */
extern void hl_expand_feature2col(const real* dataIm,
                                  size_t channels,
                                  size_t height,
                                  size_t width,
                                  size_t blockH,
                                  size_t blockW,
                                  size_t strideH,
                                  size_t strideW,
                                  size_t paddingH,
                                  size_t paddingW,
                                  size_t outputH,
                                  size_t outputW,
                                  real* dataCol);

/**
 * @brief   Maximum pool forward.
 *
 * @param[in]   frameCnt    batch size of input image.
 * @param[in]   inputData   input data.
 * @param[in]   channels    number of channel.
 * @param[in]   height      image height.
 * @param[in]   width       image width.
 * @param[in]   pooledH     output image height.
 * @param[in]   pooledW     output image width.
 * @param[in]   sizeX       width of pooling window.
 * @param[in]   sizeY       height of pooling window.
 * @param[in]   strideH     pooling stride height.
 * @param[in]   strideW     pooling stride width.
 * @param[in]   paddingH    padding height.
 * @param[in]   paddingW    padding width.
 * @param[out]  tgtData     output data.
 * @param[in]   tgtStride   stride between output data samples.
 *
 */
extern void hl_maxpool_forward(const int frameCnt,
                               const real* inputData,
                               const int channels,
                               const int height,
                               const int width,
                               const int pooledH,
                               const int pooledW,
                               const int sizeX,
                               const int sizeY,
                               const int strideH,
                               const int strideW,
                               const int paddingH,
                               const int paddingW,
                               real* tgtData,
                               const int tgtStride);

/**
 * @brief   Maximum pool backward.
 *
 * @param[in]   frameCnt    batch size of input image.
 * @param[in]   inputData   input data.
 * @param[out]  outData     output data.
 * @param[out]  outGrad     output grad data.
 * @param[in]   channels    number of channel.
 * @param[in]   height      image height.
 * @param[in]   width       image width.
 * @param[in]   pooledH     output image height.
 * @param[in]   pooledW     output image width.
 * @param[in]   sizeX       width of pooling window.
 * @param[in]   sizeY       height of pooling window.
 * @param[in]   strideH     pooling stride height.
 * @param[in]   strideW     pooling stride width.
 * @param[in]   scaleA      scale.
 * @param[in]   scaleB      scale.
 * @param[in]   paddingH    padding height.
 * @param[in]   paddingW    padding width.
 * @param[out]  targetGrad  output grad.
 * @param[in]   outStride   stride between output data samples.
 *
 */
extern void hl_maxpool_backward(const int frameCnt,
                                const real* inputData,
                                const real* outData,
                                const real* outGrad,
                                const int channels,
                                const int height,
                                const int width,
                                const int pooledH,
                                const int pooledW,
                                const int sizeX,
                                const int sizeY,
                                const int strideH,
                                const int strideW,
                                const int paddingH,
                                const int paddingW,
                                real scaleA,
                                real scaleB,
                                real* targetGrad,
                                const int outStride);

/**
 * @brief   Averge pool forward.
 *
 * @param[in]   frameCnt    batch size of input image.
 * @param[in]   inputData   input data.
 * @param[in]   channels    number of channel.
 * @param[in]   height      image height.
 * @param[in]   width       image width.
 * @param[in]   pooledH     output image height.
 * @param[in]   pooledW     output image width.
 * @param[in]   sizeX       width of pooling window.
 * @param[in]   sizeY       height of pooling window.
 * @param[in]   strideH     pooling stride height.
 * @param[in]   strideW     pooling stride width.
 * @param[in]   paddingH    padding height.
 * @param[in]   paddingW    padding width.
 * @param[out]  tgtData     output data.
 * @param[in]   tgtStride   stride between output data samples.
 *
 */
extern void hl_avgpool_forward(const int frameCnt,
                               const real* inputData,
                               const int channels,
                               const int height,
                               const int width,
                               const int pooledH,
                               const int pooledW,
                               const int sizeX,
                               const int sizeY,
                               const int strideH,
                               const int strideW,
                               const int paddingH,
                               const int paddingW,
                               real* tgtData,
                               const int tgtStride);

/**
 * @brief   Maximum pool backward.
 *
 * @param[in]   frameCnt    batch size of input image.
 * @param[in]   outGrad     output grad data.
 * @param[in]   channels    number of channel.
 * @param[in]   height      image height.
 * @param[in]   width       image width.
 * @param[in]   pooledH     output image height.
 * @param[in]   pooledW     output image width.
 * @param[in]   sizeX       width of pooling window.
 * @param[in]   sizeY       height of pooling window.
 * @param[in]   strideH     pooling stride height.
 * @param[in]   strideW     pooling stride width.
 * @param[in]   paddingH    padding height.
 * @param[in]   paddingW    padding width.
 * @param[in]   scaleA      scale.
 * @param[in]   scaleB      scale.
 * @param[out]  backGrad    output grad.
 * @param[in]   outStride   stride between output data samples.
 *
 */
extern void hl_avgpool_backward(const int frameCnt,
                                const real* outGrad,
                                const int channels,
                                const int height,
                                const int width,
                                const int pooledH,
                                const int pooledW,
                                const int sizeX,
                                const int sizeY,
                                const int strideH,
                                const int strideW,
                                int paddingH,
                                int paddingW,
                                real scaleA,
                                real scaleB,
                                real* backGrad,
                                const int outStride);

/**
 * @brief   Bilinear interpolation forward.
 *
 * @param[in]   inData      input value.
 * @param[in]   inImgH      input image height.
 * @param[in]   inImgW      input image width.
 * @param[in]   inputH      input batchSize.
 * @param[in]   inputW      input image data dim.
 * @param[out]  outData     output value.
 * @param[in]   outImgH     output image height.
 * @param[in]   outImgW     output image width.
 * @param[in]   outputH     output batchSize.
 * @param[in]   outputW     output image data dim.
 * @param[in]   numChannels number of channels.
 * @param[in]   ratioH      inImgH / outImgH.
 * @param[in]   ratioW      inImgW / outImgW.
 *
 */
extern void hl_bilinear_forward(const real* inData,
                                const size_t inImgH,
                                const size_t inImgW,
                                const size_t inputH,
                                const size_t inputW,
                                real* outData,
                                const size_t outImgH,
                                const size_t outImgW,
                                const size_t outputH,
                                const size_t outputW,
                                const size_t numChannels,
                                const real ratioH,
                                const real ratioW);

/**
 * @brief   Bilinear interpolation backward.
 *
 * @param[out]  inGrad      input gradient.
 * @param[in]   inImgH      input image height.
 * @param[in]   inImgW      input image width.
 * @param[in]   inputH      input batchSize.
 * @param[in]   inputW      input image data dim.
 * @param[in]   outGrad     output gradient.
 * @param[in]   outImgH     output image height.
 * @param[in]   outImgW     output image width.
 * @param[in]   outputH     output batchSize.
 * @param[in]   outputW     output image data dim.
 * @param[in]   numChannels number of channels.
 * @param[in]   ratioH      inImgH / outImgH.
 * @param[in]   ratioW      inImgW / outImgW.
 *
 */
extern void hl_bilinear_backward(real* inGrad,
                                 const size_t inImgH,
                                 const size_t inImgW,
                                 const size_t inputH,
                                 const size_t inputW,
                                 const real* outGrad,
                                 const size_t outImgH,
                                 const size_t outImgW,
                                 const size_t outputH,
                                 const size_t outputW,
                                 const size_t numChannels,
                                 const real ratioH,
                                 const real ratioW);

/**
 * @brief   MaxOut forward.
 *
 * @param[in]   inData      input data.
 * @param[out]  outData     output data.
 * @param[out]  idData      output maxId.
 * @param[in]   batchSize   batchSize.
 * @param[in]   size        number of channels * image height * image width.
 * @param[in]   featLen     feature length = image height * image width.
 * @param[in]   groups      number of groups.
 */
extern void hl_maxout_forward(const real* inData,
                              real* outData,
                              int* idData,
                              size_t batchSize,
                              size_t size,
                              size_t featLen,
                              size_t groups);

/**
 * @brief   MaxOut backward.
 *
 * @param[out]  inGrad      input grad data.
 * @param[in]   outGrad     output grad data.
 * @param[in]   idData      output maxId.
 * @param[in]   batchSize   batchSize.
 * @param[in]   size        number of channels * image height * image width.
 * @param[in]   featLen     feature length = image height * image width.
 * @param[in]   groups      number of groups.
 */
extern void hl_maxout_backward(real* inGrad,
                               const real* outGrad,
                               const int* idData,
                               size_t batchSize,
                               size_t size,
                               size_t featLen,
                               size_t groups);

#endif /* HL_CNN_H_ */
