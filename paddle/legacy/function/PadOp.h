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

struct PadConf {
  /// how many values to add before/after the data along channel dimension.
  std::vector<uint32_t> channel;
  /// how many values to add before/after the data along height dimension.
  std::vector<uint32_t> height;
  /// how many values to add before/after the data along width dimension.
  std::vector<uint32_t> width;
};

/**
 * \brief  This funtion pads zeros to inputs according to the specify dimension.
 *         The input and output is a 4D tensor. Padding zeros from the 2nd to
 *         the 4th dimenstion according argument of pad.
 *
 * \param[out] outputs save results.
 * \param[in]  inputs  input data.
 * \param[in]  num     batch size of input data.
 * \param[in]  inC     channel number of input data.
 * \param[in]  inH     height of input data.
 * \param[in]  inH     with of input data.
 * \param[in]  pad     the padding config, contains the size along the
 *                     specify dimension.
 */
template <DeviceType Device>
void Pad(real* outputs,
         const real* inputs,
         const int num,
         const int inC,
         const int inH,
         const int inW,
         const PadConf& pad);

/**
 * \brief   Padding operation backward.
 *
 * \param[out] inGrad  gradients of previous layer.
 * \param[in]  outGrad output gradients.
 * \param[in]  num     batch size of input data.
 * \param[in]  inC     channel number of input data.
 * \param[in]  inH     height of input data.
 * \param[in]  inH     with of input data.
 * \param[in]  pad     the padding config, contains the size along the
 *                     specify dimension.
 */
template <DeviceType Device>
void PadGrad(real* inGrad,
             const real* outGrad,
             const int num,
             const int inC,
             const int inH,
             const int inW,
             const PadConf& pad);
}  // namespace paddle
