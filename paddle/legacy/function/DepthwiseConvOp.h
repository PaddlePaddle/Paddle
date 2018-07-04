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

#include "TensorType.h"

namespace paddle {

/**
 *\brief   Depthwise convolution forward. The outputData
 *         of depthwise convolution is same with ExpandConvLayer
 *         when groups equals inputChannels in ExpandConvLayer.
 *
 * \param[in]   inputData         input data.
 * \param[in]   filterData        the Paramters of the depthwise conv layer..
 * \param[in]   batchSize         batch size of input data.
 * \param[in]   outputChannels    channels of outputData.
 * \param[in]   outputHeight      height of outputData.
 * \param[in]   outputWidth       width of outputData.
 * \param[in]   inputChannels     channels of inputData.
 * \param[in]   inputHeight       height of inputData.
 * \param[in]   inputWidth        width of inputData..
 * \param[in]   filterMultiplier  equals to outputChannels/groups_.
 * \param[in]   filterHeight      height of filter.
 * \param[in]   filterWidth       widht of filter.
 * \param[in]   strideH           stride size in height direction.
 * \param[in]   strideW           stride size in width direction.
 * \param[in]   paddingH          padding size in height direction.
 * \param[in]   paddingW          padding size in width direction.
 * \param[out]  outputData        outputData.
 *
 */
template <DeviceType Device, class T>
class DepthwiseConvFunctor {
 public:
  void operator()(const T* inputData,
                  const T* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterMultiplier,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* outputData);
};

/**
 *\brief  Functor tot compute the depthwise convolution backprop w.r.t input.
 *
 *
 * \param[in]   outputGradData    the grad data of output.
 * \param[in]   filterData        the Paramters of the depthwise conv layer..
 * \param[in]   batchSize         batch size of input data.
 * \param[in]   outputChannels    channels of outputData.
 * \param[in]   outputHeight      height of outputData.
 * \param[in]   outputWidth       width of outputData.
 * \param[in]   inputChannels     channels of input data.
 * \param[in]   inputHeight       height of inputData.
 * \param[in]   inputWidth        width of inputData.
 * \param[in]   filterMultiplier  equals to outputChannels/groups_.
 * \param[in]   filterHeight      height of filter.
 * \param[in]   filterWidth       widht of filter.
 * \param[in]   strideH           stride size in height direction.
 * \param[in]   strideW           stride size in width direction.
 * \param[in]   paddingH          padding size in height direction.
 * \param[in]   paddingW          padding size in width direction.
 * \param[out]  inputGrad         the grad data of input.
 *
 */
template <DeviceType Device, class T>
class DepthwiseConvGradInputFunctor {
 public:
  void operator()(const T* outputGrad,
                  const T* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterMultiplier,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* inputGrad);
};

/**
 *\brief  Functor tot compute the depthwise convolution backprop w.r.t filter.
 *
 * \param[in]   outputGradData    the grad data of output.
 * \param[in]   inputData         inputData.
 * \param[in]   batchSize         batch size of input data.
 * \param[in]   outputChannels    channels of outputData.
 * \param[in]   outputHeight      height of outputData.
 * \param[in]   outputWidth       width of outputData.
 * \param[in]   inputChannels     channels of input data.
 * \param[in]   inputHeight       height of inputData.
 * \param[in]   inputWidth        width of inputData.
 * \param[in]   filterMultiplier  equals to outputChannels/groups_.
 * \param[in]   filterHeight      height of filter.
 * \param[in]   filterWidth       widht of filter.
 * \param[in]   strideH           stride size in height direction.
 * \param[in]   strideW           stride size in width direction.
 * \param[in]   paddingH          padding size in height direction.
 * \param[in]   paddingW          padding size in width direction.
 * \param[in]   colData           Auxiliary data when calculating filterGrad.
 * \param[in]   multiplierData    Auxiliary data when calculating filterGrad.
 * \param[out]  filterGrad        the grad data of filter.
 *
 */
template <DeviceType Device, class T>
class DepthwiseConvGradFilterFunctor {
 public:
  void operator()(const T* outputGrad,
                  const T* inputData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterMultiplier,
                  int filterHeight,
                  int filterWidth,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  T* colData,
                  T* filterGrad);
};

}  // namespace paddle
