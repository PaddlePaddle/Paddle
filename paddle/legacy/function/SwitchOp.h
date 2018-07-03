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

#include "Function.h"

namespace paddle {

/**
 * \brief  This funtion switch dimension order of image input.
 *         The input and output is a 4D tensor. Switch order 'batch_size,
 *channels, height, width' to
 *         order 'batch_size, height, width, channels'.
 *
 * \param[out] outputs save results.
 * \param[in]  inputs  input data.
 * \param[in]  num     batch size of input data.
 * \param[in]  inC     channel number of input data.
 * \param[in]  inH     height of input data.
 * \param[in]  inH     with of input data.
 * \param[in]  argType     type of output argument.
 */
template <DeviceType Device>
void NCHW2NHWC(real* outputs,
               const real* inputs,
               const int num,
               const int inC,
               const int inH,
               const int inW,
               const int argtype);

/**
 * \brief  This funtion switch dimension order of image input.
 *         The input and output is a 4D tensor. Switch order 'batch_size,
 *height, width, channels' to
 *         order 'batch_size, channels, height, width'.
 *
 * \param[out] inGrad  gradients of previous layer.
 * \param[in]  outGrad output gradients.
 * \param[in]  num     batch size of input data.
 * \param[in]  inH     height of input data.
 * \param[in]  inW     with of input data.
 * \param[in]  inC     channel number of input data.
 * \param[in]  argType     type of output argument.
 */
template <DeviceType Device>
void NHWC2NCHW(real* inGrad,
               const real* outGrad,
               const int num,
               const int inH,
               const int inW,
               const int inC,
               const int argType);
}  // namespace paddle
