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


#ifndef HL_CNN_STUB_H_
#define HL_CNN_STUB_H_

#include "hl_cnn.h"

inline void hl_shrink_col2feature(
    const real * dataCol, size_t channels,
    size_t height, size_t width,
    size_t blockH, size_t blockW,
    size_t strideH, size_t strideW,
    size_t paddingH, size_t paddingW,
    size_t outputH, size_t outputW,
    real* dataIm,
    real alpha, real beta) {}

inline void hl_expand_feature2col(
    const real* dataIm, size_t channels,
    size_t height, size_t width,
    size_t blockH, size_t blockW,
    size_t strideH, size_t strideW,
    size_t paddingH, size_t paddingW,
    size_t outputH, size_t outputW,
    real* dataCol) {}

inline void hl_maxpool_forward(
    int frameCnt, const real* inputData, int channels,
    int height, int width, int pooledH, int pooledW,
    int sizeX, int stride, int start, real* tgtData) {}

inline void hl_maxpool_backward(
    int frameCnt, const real* inputData,
    const real* outData, const real* outGrad,
    int channels, int height, int width,
    int pooledH, int pooledW, int sizeX,
    int stride, int start, real* targetGrad,
    real scaleA, real scaleB) {}

inline void hl_avgpool_forward(
    int frameCnt, const real* inputData, int channels,
    int height, int width, int pooledH, int pooledW,
    int sizeX, int stride, int start, real* tgtData) {}

inline void hl_avgpool_backward(
    int frameCnt, const real* outGrad,
    int channels, int height, int width,
    int pooledH, int pooledW, int sizeX,
    int stride, int start, real* backGrad,
    real scaleA, real scaleB) {}

inline void hl_CMRNorm_forward(
    size_t frameCnt, const real* in, real* scale, real* out,
    size_t channels, size_t height, size_t width, size_t sizeX,
    real alpha, real beta) {}

inline void hl_CMRNorm_backward(
    size_t frameCnt, const real* inV, const real* scale,
    const real* outV, const real* outDiff, real *inDiff,
    size_t channels, size_t height, size_t width, size_t sizeX,
    real alpha, real beta) {}

#endif  // HL_CNN_STUB_H_
