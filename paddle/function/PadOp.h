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

#include "Function.h"

namespace paddle {

/**
 * \brief  This funtion pads zeros to inputs according to the specify dimension.
 *         The data structure of image data is NCHW.
 *
 * \param[out]  outputs  save results.
 * \param[in]   inputs   input data.
 * \param[in]   num      batch size of input data.
 * \param[in]   inC      channel number of input data.
 * \param[in]   inH      height of input data.
 * \param[in]   inH      with of input data.
 * \param[in]   padc0    how many values to add before the data in dimension of
 * channel.
 * \param[in]   padc1    how many values to add after the data in dimension of
 * channel.
 * \param[in]   padh0    how many values to add before the data in dimension of
 * height.
 * \param[in]   padh1    how many values to add after the data in dimension of
 * height.
 * \param[in]   padw0    how many values to add before the data in dimension of
 * width.
 * \param[in]   padw1    how many values to add after the data in dimension of
 * width.
 *
 */
template <DeviceType Device>
void Pad(real* outputs,
         const real* inputs,
         const int num,
         const int inC,
         const int inH,
         const int inW,
         const int padc0,
         const int padc1,
         const int padh0,
         const int padh1,
         const int padw0,
         const int padw1);

/**
 * \brief   Padding operation backward.
 *          The data structure of image data is NCHW.
 *
 * \param[out]  inGrad   gradients of previous layer.
 * \param[in]   outGrad  output gradients.
 * \param[in]   num      batch size of input data.
 * \param[in]   inC      channel number of input data.
 * \param[in]   inH      height of input data.
 * \param[in]   inH      with of input data.
 * \param[in]   padc0    how many values to add before the data in dimension of
 * channel.
 * \param[in]   padc1    how many values to add after the data in dimension of
 * channel.
 * \param[in]   padh0    how many values to add before the data in dimension of
 * height.
 * \param[in]   padh1    how many values to add after the data in dimension of
 * height.
 * \param[in]   padw0    how many values to add before the data in dimension of
 * width.
 * \param[in]   padw1    how many values to add after the data in dimension of
 * width.
 *
 */
template <DeviceType Device>
void PadGrad(real* inGrad,
             const real* outGrad,
             const int num,
             const int inC,
             const int inH,
             const int inW,
             const int padc0,
             const int padc1,
             const int padh0,
             const int padh1,
             const int padw0,
             const int padw1);
}  // namespace paddle
