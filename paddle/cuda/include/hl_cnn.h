/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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
extern void hl_shrink_col2feature(
    const real * dataCol, size_t channels,
    size_t height, size_t width,
    size_t blockH, size_t blockW,
    size_t strideH, size_t strideW,
    size_t paddingH, size_t paddingW,
    size_t outputH, size_t outputW,
    real* dataIm,
    real alpha = 1.0f, real beta = 0.0f);

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
extern void hl_expand_feature2col(
    const real* dataIm, size_t channels,
    size_t height, size_t width,
    size_t blockH, size_t blockW,
    size_t strideH, size_t strideW,
    size_t paddingH, size_t paddingW,
    size_t outputH, size_t outputW,
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
 * @param[in]   sizeX       size of pooling window.
 * @param[in]   stride      pooling stride.
 * @param[in]   start       pooling start.
 * @param[out]  tgtData     output data.
 *
 */
extern void hl_maxpool_forward(
    int frameCnt, const real* inputData, int channels,
    int height, int width, int pooledH, int pooledW,
    int sizeX, int stride, int start, real* tgtData);

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
 * @param[in]   sizeX       size of pooling window.
 * @param[in]   stride      pooling stride.
 * @param[in]   start       pooling start.
 * @param[out]  targetGrad  output grad.
 * @param[in]   scaleA      scale.
 * @param[in]   scaleB      scale.
 *
 */
extern void hl_maxpool_backward(
    int frameCnt, const real* inputData,
    const real* outData, const real* outGrad,
    int channels, int height, int width,
    int pooledH, int pooledW, int sizeX,
    int stride, int start, real* targetGrad,
    real scaleA, real scaleB);

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
 * @param[in]   sizeX       size of pooling window.
 * @param[in]   stride      pooling stride.
 * @param[in]   start       pooling start.
 * @param[out]  tgtData     output data.
 *
 */
extern void hl_avgpool_forward(
    int frameCnt, const real* inputData, int channels,
    int height, int width, int pooledH, int pooledW,
    int sizeX, int stride, int start, real* tgtData);

/**
 * @brief   Maximum pool backward.
 *
 * @param[in]   frameCnt    batch size of input image.
 * @param[in]   outGrad     input data.
 * @param[in]   channels    number of channel.
 * @param[in]   height      image height.
 * @param[in]   width       image width.
 * @param[in]   pooledH     output image height.
 * @param[in]   pooledW     output image width.
 * @param[in]   sizeX       size of pooling window.
 * @param[in]   stride      pooling stride.
 * @param[in]   start       pooling start.
 * @param[out]  backGrad    output grad.
 * @param[in]   scaleA      scale.
 * @param[in]   scaleB      scale.
 *
 */
extern void hl_avgpool_backward(
    int frameCnt, const real* outGrad,
    int channels, int height, int width,
    int pooledH, int pooledW, int sizeX,
    int stride, int start, real* backGrad,
    real scaleA, real scaleB);

/**
 * @brief   Cross-map-respose normalize forward.
 *
 * @param[in]   frameCnt    batch size of input image.
 * @param[in]   in          input data.
 * @param[in]   scale       buffer.
 * @param[out]  out         output data.
 * @param[in]   channels    number of channel.
 * @param[in]   height      image height.
 * @param[in]   width       image width.
 * @param[in]   sizeX       size.
 * @param[in]   alpha       scale.
 * @param[in]   beta        scale.
 *
 */
extern void hl_CMRNorm_forward(
    size_t frameCnt, const real* in, real* scale, real* out,
    size_t channels, size_t height, size_t width, size_t sizeX,
    real alpha, real beta);

/**
 * @brief   Cross-map-respose normalize backward.
 *
 * @param[in]   frameCnt    batch size of input image.
 * @param[in]   inV         input data.
 * @param[in]   scale       buffer.
 * @param[out]  outV        output value.
 * @param[out]  outDiff     output grad.
 * @param[out]  inDiff      input grad.
 * @param[in]   channels    number of channel.
 * @param[in]   height      image height.
 * @param[in]   width       image width.
 * @param[in]   sizeX       size.
 * @param[in]   alpha       scale.
 * @param[in]   beta        scale.
 *
 */
extern void hl_CMRNorm_backward(
    size_t frameCnt, const real* inV, const real* scale,
    const real* outV, const real* outDiff, real *inDiff,
    size_t channels, size_t height, size_t width, size_t sizeX,
    real alpha, real beta);

#endif /* HL_CNN_H_ */
