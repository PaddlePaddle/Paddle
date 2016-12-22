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

#ifndef HL_CNN_STUB_H_
#define HL_CNN_STUB_H_

#include "hl_cnn.h"

inline void hl_shrink_col2feature(const real* dataCol,
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
                                  real alpha,
                                  real beta) {}

inline void hl_expand_feature2col(const real* dataIm,
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
                                  real* dataCol) {}

inline void hl_maxpool_forward(const int frameCnt,
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
                               const int tgtStride) {}

inline void hl_maxpool_backward(const int frameCnt,
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
                                const int outStride) {}

inline void hl_avgpool_forward(const int frameCnt,
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
                               const int tgtStride) {}

inline void hl_avgpool_backward(const int frameCnt,
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
                                const int outStride) {}

inline void hl_bilinear_forward(const real* inData,
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
                                const real ratioW) {}

inline void hl_bilinear_backward(real* inGrad,
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
                                 const real ratioW) {}

inline void hl_maxout_forward(const real* inData,
                              real* outData,
                              int* idData,
                              size_t batchSize,
                              size_t size,
                              size_t featLen,
                              size_t group) {}

inline void hl_maxout_backward(real* inGrad,
                               const real* outGrad,
                               const int* idData,
                               size_t batchSize,
                               size_t size,
                               size_t featLen,
                               size_t group) {}

#endif  // HL_CNN_STUB_H_
