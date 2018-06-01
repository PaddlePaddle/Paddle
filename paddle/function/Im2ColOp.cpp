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

#include "Im2Col.h"

namespace paddle {

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [inputChannels, filterHeight, filterWidth, outputHeight, outputWidth]
 */
template <class T>
class Im2ColFunctor<kCFO, DEVICE_TYPE_CPU, T> {
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
                  int dilationWidth) {
    int inputChannels = imShape[0];
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[1];
    int filterWidth = colShape[2];
    int outputHeight = colShape[3];
    int outputWidth = colShape[4];
    int channelsCol = inputChannels * filterHeight * filterWidth;

    for (int c = 0; c < channelsCol; ++c) {
      int wOffset = c % filterWidth;
      int hOffset = (c / filterWidth) % filterHeight;
      int c_im = c / filterWidth / filterHeight;
      for (int h = 0; h < outputHeight; ++h) {
        for (int w = 0; w < outputWidth; ++w) {
          int imRowIdx = h * strideHeight + hOffset * dilationHeight;
          int imColIdx = w * strideWidth + wOffset * dilationWidth;
          if ((imRowIdx - paddingHeight) < 0 ||
              (imRowIdx - paddingHeight) >= inputHeight ||
              (imColIdx - paddingWidth) < 0 ||
              (imColIdx - paddingWidth) >= inputWidth) {
            colData[(c * outputHeight + h) * outputWidth + w] = T(0);
          } else {
            imRowIdx += c_im * inputHeight - paddingHeight;
            imColIdx -= paddingWidth;
            colData[(c * outputHeight + h) * outputWidth + w] =
                imData[imRowIdx * inputWidth + imColIdx];
          }
        }
      }
    }
  }
};

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [inputChannels, filterHeight, filterWidth, outputHeight, outputWidth]
 */
template <class T>
class Col2ImFunctor<kCFO, DEVICE_TYPE_CPU, T> {
 public:
  void operator()(T* imData,
                  const TensorShape& imShape,
                  const T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int dilationHeight,
                  int dilationWidth) {
    int inputChannels = imShape[0];
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[1];
    int filterWidth = colShape[2];
    int outputHeight = colShape[3];
    int outputWidth = colShape[4];
    int channelsCol = inputChannels * filterHeight * filterWidth;

    for (int c = 0; c < channelsCol; ++c) {
      int wOffset = c % filterWidth;
      int hOffset = (c / filterWidth) % filterHeight;
      int c_im = c / filterWidth / filterHeight;
      for (int h = 0; h < outputHeight; ++h) {
        for (int w = 0; w < outputWidth; ++w) {
          int imRowIdx = h * strideHeight + hOffset * dilationHeight;
          int imColIdx = w * strideWidth + wOffset * dilationWidth;
          if ((imRowIdx - paddingHeight) >= 0 &&
              (imRowIdx - paddingHeight) < inputHeight &&
              (imColIdx - paddingWidth) >= 0 &&
              (imColIdx - paddingWidth) < inputWidth) {
            imRowIdx += c_im * inputHeight - paddingHeight;
            imColIdx -= paddingWidth;
            imData[imRowIdx * inputWidth + imColIdx] +=
                colData[(c * outputHeight + h) * outputWidth + w];
          }
        }
      }
    }
  }
};

template class Im2ColFunctor<kCFO, DEVICE_TYPE_CPU, float>;
template class Im2ColFunctor<kCFO, DEVICE_TYPE_CPU, double>;
template class Col2ImFunctor<kCFO, DEVICE_TYPE_CPU, float>;
template class Col2ImFunctor<kCFO, DEVICE_TYPE_CPU, double>;

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [outputHeight, outputWidth, inputChannels, filterHeight, filterWidth]
 */
template <class T>
class Im2ColFunctor<kOCF, DEVICE_TYPE_CPU, T> {
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
                  int dilationWidth = 1) {
    int inputChannels = imShape[0];
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[3];
    int filterWidth = colShape[4];
    int outputHeight = colShape[0];
    int outputWidth = colShape[1];
    for (int outputH = 0; outputH < outputHeight; ++outputH) {
      for (int outputW = 0; outputW < outputWidth; ++outputW) {
        for (int channel = 0; channel < inputChannels; ++channel) {
          for (int filterH = 0; filterH < filterHeight; ++filterH) {
            for (int filterW = 0; filterW < filterWidth; ++filterW) {
              int imRowOffset = outputH * strideHeight +
                                filterH * dilationHeight - paddingHeight;
              int imColOffset = outputW * strideWidth +
                                filterW * dilationWidth - paddingWidth;
              int colDataOffset =
                  (((outputH * outputWidth + outputW) * inputChannels +
                    channel) *
                       filterHeight +
                   filterH) *
                      filterWidth +
                  filterW;
              if (imRowOffset < 0 || imRowOffset >= inputHeight ||
                  imColOffset < 0 || imColOffset >= inputWidth) {
                colData[colDataOffset] = float(0);
              } else {
                int imDataOffset =
                    (channel * inputHeight + imRowOffset) * inputWidth +
                    imColOffset;
                colData[colDataOffset] = imData[imDataOffset];
              }
            }
          }
        }
      }
    }
  }
};

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [outputHeight, outputWidth, inputChannels, filterHeight, filterWidth]
 */
template <class T>
class Col2ImFunctor<kOCF, DEVICE_TYPE_CPU, T> {
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
                  int dilationWidth = 1) {
    int inputChannels = imShape[0];
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[3];
    int filterWidth = colShape[4];
    int outputHeight = colShape[0];
    int outputWidth = colShape[1];
    for (int outputH = 0; outputH < outputHeight; ++outputH) {
      for (int outputW = 0; outputW < outputWidth; ++outputW) {
        for (int channel = 0; channel < inputChannels; ++channel) {
          for (int filterH = 0; filterH < filterHeight; ++filterH) {
            for (int filterW = 0; filterW < filterWidth; ++filterW) {
              int imRowOffset = outputH * strideHeight +
                                filterH * dilationHeight - paddingHeight;
              int imColOffset = outputW * strideWidth +
                                filterW * dilationWidth - paddingWidth;
              int colDataOffset =
                  (((outputH * outputWidth + outputW) * inputChannels +
                    channel) *
                       filterHeight +
                   filterH) *
                      filterWidth +
                  filterW;
              if (imRowOffset >= 0 && imRowOffset < inputHeight &&
                  imColOffset >= 0 && imColOffset < inputWidth) {
                int imDataOffset =
                    (channel * inputHeight + imRowOffset) * inputWidth +
                    imColOffset;
                imData[imDataOffset] += colData[colDataOffset];
              }
            }
          }
        }
      }
    }
  }
};

template class Im2ColFunctor<kOCF, DEVICE_TYPE_CPU, float>;
template class Im2ColFunctor<kOCF, DEVICE_TYPE_CPU, double>;
template class Col2ImFunctor<kOCF, DEVICE_TYPE_CPU, float>;
template class Col2ImFunctor<kOCF, DEVICE_TYPE_CPU, double>;

}  // namespace paddle
