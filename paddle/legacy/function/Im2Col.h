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

#include "TensorShape.h"
#include "TensorType.h"
#include "neon/neon_util.h"

namespace paddle {

/* The storage format of the coldata in the Im2ColFunctor and Col2ImFunctor. */
enum ColFormat { kCFO = 0, kOCF = 1 };

/*
 * \brief Converts the image data of three dimensions(CHW) into a colData of
 *        five dimensions in the Im2ColFunctor calculation,
 *        And in the Col2ImFunctor calculation, it is reversed.
 *
 * \param imData   Image data.
 * \param imShape  The shape of imData,
 *                 [inputChannels, inputHeight, inputWidth].
 * \param colData  Column data.
 * \param colShape The shape of colData.
 *
 * If the template argument Format is kCFO, the shape of colData is:
 * [inputChannels, filterHeight, filterWidth, outputHeight, outputWidth]
 * So, it is easy to reshape into a convolution matrix for convolution
 * calculation based on matrix multiplication.
 * The shape of convolution matrix is [height, width], where the height is equal
 * inputChannels * filterHeight * filterWidth, and the width is equal
 * outputHeight * outputWidth.
 *
 * Reshape:
 *     shape of colData           shape of convolution matrix
 *     [inputChannels,
 *      filterHeight,
 *      filterWidth,      ======>      [height, width]
 *      outputHeight,
 *      outputWidth]
 *
 * If the template argument Format is kOCF, the shape of colData is:
 * [outputHeight, outputWidth, inputChannels, filterHeight, filterWidth]
 * So, it is easy to reshape into a sequence matrix for rnn calculation.
 * The shape of sequence matrix is [seqLength, stepSize], where the seqLength
 * is equal outputHeight * outputWidth, and the stepSize is equal
 * inputChannels * filterHeight * filterWidth.
 *
 * Reshape:
 *     shape of colData             shape of sequence matrix
 *     [outputHeight,
 *      outputWidth,
 *      inputChannels,    ======>    [seqLength, stepSize]
 *      filterHeight,
 *      filterWidth]
 *
 * \note The caller needs to ensure that imShape.inputChannels is equal to
 *       colShape.inputChannels.
 */
template <ColFormat Format, DeviceType Device, class T>
class Im2ColFunctor {
 public:
  void operator()(const T* imData,
                  const TensorShape& imShape,
                  T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int dilationHeight = 1,
                  int dilationWidth = 1);
};

template <ColFormat Format, DeviceType Device, class T>
class Col2ImFunctor {
 public:
  void operator()(T* imData,
                  const TensorShape& imShape,
                  const T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int dilationHeight = 1,
                  int dilationWidth = 1);
};

template <class T>
class Im2ColMobileFunctor {
 public:
  void operator()(const T* imData,
                  const TensorShape& imShape,
                  T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int dilationHeight,
                  int dilationWidth,
                  int inputChannels,
                  int colOffset,
                  int colOutputHeight,
                  int colWidth) {
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[1];
    int filterWidth = colShape[2];
    int outputWidth = colShape[4];

    for (int ic = 0; ic < inputChannels; ic++) {
      for (int oh = 0; oh < colOutputHeight; oh++) {
        T* dstData = colData + oh * outputWidth;
        for (int fh = 0; fh < filterHeight; fh++) {
          for (int fw = 0; fw < filterWidth; fw++) {
            int imRowIdx = (oh + colOffset) * strideHeight +
                           fh * dilationHeight - paddingHeight;
            if (imRowIdx < 0 || imRowIdx >= inputHeight) {
              memset(dstData, 0, outputWidth * sizeof(T));
            } else {
              for (int ow = 0; ow < outputWidth; ow++) {
                int imColIdx =
                    ow * strideWidth + fw * dilationWidth - paddingWidth;
                if (imColIdx < 0 || imColIdx >= inputWidth) {
                  dstData[ow] = T(0);
                } else {
                  dstData[ow] = imData[imRowIdx * inputWidth + imColIdx];
                }
              }
            }
            dstData += colWidth;
          }
        }
      }
      colData += filterHeight * filterWidth * colWidth;
      imData += inputHeight * inputWidth;
    }
  }
};

}  // namespace paddle
