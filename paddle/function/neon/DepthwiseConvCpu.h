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

#include "paddle/function/neon/DepthwiseConvNeon.h"

namespace paddle {

template <class T>
struct DepthwiseConvNaiveCPUKernel {
  static void run(const T* inputPaddedData,
                  const T* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputPaddedHeight,
                  int inputPaddedWidth,
                  int filterMultiplier,
                  int filterSize,
                  int stride,
                  T* outputData) {
    for (int n = 0; n < batchSize; ++n) {
      for (int cOut = 0; cOut < outputChannels; ++cOut) {
        for (int hOut = 0; hOut < outputHeight; ++hOut) {
          for (int wOut = 0; wOut < outputWidth; ++wOut) {
            const T* filterDataTemp =
                filterData + cOut * filterSize * filterSize;
            T value = 0;
            const int cIn = cOut / filterMultiplier;
            for (int fh = 0; fh < filterSize; ++fh) {
              for (int fw = 0; fw < filterSize; ++fw) {
                int hIn = hOut * stride + fh;
                int wIn = wOut * stride + fw;
                if ((hIn < inputPaddedHeight) && (wIn < inputPaddedWidth)) {
                  int offset =
                      ((n * inputChannels + cIn) * inputPaddedHeight + hIn) *
                          inputPaddedWidth +
                      wIn;
                  value += (*filterDataTemp) * inputPaddedData[offset];
                }
                ++filterDataTemp;
              }
            }
            *outputData++ = value;
          }
        }
      }
    }
  }
};

template <class T>
void ComputeDepthwiseConv(const T* inputPaddedData,
                          const T* filterData,
                          int batchSize,
                          int outputChannels,
                          int outputHeight,
                          int outputWidth,
                          int inputChannels,
                          int inputHeight,
                          int inputWidth,
                          int filterMultiplier,
                          int filterSize,
                          int stride,
                          int padding,
                          T* outputData) {
  typedef void (*DepthwiseConvRun)(const T* /*inputPaddedData */,
                                   const T* /*filterData*/,
                                   int /*batchSize*/,
                                   int /*outputChannels*/,
                                   int /*outputHeight*/,
                                   int /*outputWidth*/,
                                   int /*inputChannels*/,
                                   int /*inputPaddedHeight*/,
                                   int /*inputPaddedWidth*/,
                                   int /*filterMultiplier*/,
                                   int /*filterSize*/,
                                   int /*stride*/,
                                   T* /*outputData*/);
  DepthwiseConvRun func;
  func = DepthwiseConvNaiveCPUKernel<T>::run;

#ifdef HAVE_NEON
#define PADDLE_USE_DEPTHWISECONV_KERNEL(FILTERSIZE, STRIDE) \
  if (FILTERSIZE == filterSize && STRIDE == stride) {
  func = neon::DepthwiseConvNeonKernel<FILTERSIZE, STRIDE>::run;
}

PADDLE_USE_DEPTHWISECONV_KERNEL(3, 1)
PADDLE_USE_DEPTHWISECONV_KERNEL(3, 2)
#endif

int inputPaddedHeight = inputHeight + 2 * padding;
int inputPaddedWidth = inputWidth + 2 * padding;
func(inputPaddedData,
     filterData,
     batchSize,
     outputChannels,
     outputHeight,
     outputWidth,
     inputChannels,
     inputPaddedHeight,
     inputPaddedWidth,
     filterMultiplier,
     filterSize,
     stride,
     outputData);
}

}  // end namespace
