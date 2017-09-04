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
                  int paddingWidth);
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
                  int paddingWidth);
};

template <class T>
struct Padding {
  static void run(const T* src,
                  T* dest,
                  int channels,
                  int inputHeight,
                  int inputWidth,
                  int paddingHeight,
                  int paddingWidth) {
    const int destWidth = inputWidth + 2 * paddingWidth;
    for (int c = 0; c < channels; c++) {
      if (paddingHeight > 0) {
        memset(dest, 0, destWidth * paddingHeight * sizeof(T));
        dest += destWidth * paddingHeight;
      }

      for (int i = 0; i < inputHeight; i++) {
        // padding head
        for (int j = 0; j < paddingWidth; j++) {
          *dest++ = T(0);
        }

        memcpy(dest, src, inputWidth * sizeof(T));
        dest += inputWidth;
        src += inputWidth;

        // padding tail
        for (int j = 0; j < paddingWidth; j++) {
          *dest++ = T(0);
        }
      }

      if (paddingHeight > 0) {
        memset(dest, 0, destWidth * paddingHeight * sizeof(T));
        dest += destWidth * paddingHeight;
      }
    }
  }
};

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
template <>
struct Padding<float> {
  static void run(const float* src,
                  float* dest,
                  int channels,
                  int inputHeight,
                  int inputWidth,
                  int paddingHeight,
                  int paddingWidth) {
    const int destWidth = inputWidth + 2 * paddingWidth;
    for (int c = 0; c < channels; c++) {
      if (paddingHeight > 0) {
        memset(dest, 0, destWidth * paddingHeight * sizeof(float));
        dest += destWidth * paddingHeight;
      }

      for (int i = 0; i < inputHeight; i++) {
        // padding head
        for (int j = 0; j < paddingWidth; j++) {
          *dest++ = float(0);
        }

        int step = inputWidth >> 2;
        int remain = inputWidth & 3;
        for (int s = 0; s < step; s++) {
          float32x4_t s0 = vld1q_f32(src);
          vst1q_f32(dest, s0);
          src += 4;
          dest += 4;
        }
        for (int r = 0; r < remain; r++) {
          *dest++ = *src++;
        }

        // padding tail
        for (int j = 0; j < paddingWidth; j++) {
          *dest++ = float(0);
        }
      }

      if (paddingHeight > 0) {
        memset(dest, 0, destWidth * paddingHeight * sizeof(float));
        dest += destWidth * paddingHeight;
      }
    }
  }
};

#endif

}  // namespace paddle
