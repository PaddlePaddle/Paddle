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

#include "ConvOp.h"

namespace paddle {

/*
 * imData = [input_channels, input_height, input_width]
 * colData = [input_channels, filter_height, filter_width,
 *            output_height, output_width]
 */
template <DeviceType Device, class T>
class DepthwiseConvFunctor {
public:
  void operator()(int outputSize,
                  const T* inputData,
                  const T* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* outputData);
};

template <DeviceType Device, class T>
class DepthwiseConvGradInputFunctor {
public:
  void operator()(int inputSize,
                  const T* outputGrad,
                  const T* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputHeight,
                  int inputWidth,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* inputGrad);
};

template <DeviceType Device, class T>
class DepthwiseConvGradFilterFunctor {
public:
  void operator()(int num_i,
                  int colDataSize,
                  const T* outputGrad,
                  const T* inputData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputHeight,
                  int inputWidth,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* colData,
                  T* multiplierData,
                  T* filterGrad);

};  // namespace paddle

}  // namespace paddle
