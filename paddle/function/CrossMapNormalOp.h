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
 * \brief   Cross map respose normalize forward.
 *          The data structure of image data is NCHW.
 *
 * \param[out]  outputs     output data.
 * \param[in]   denoms      denoms buffer.
 * \param[in]   inputs      input data.
 * \param[in]   numSamples  batch size of input image.
 * \param[in]   channels    number of channel.
 * \param[in]   height      image height.
 * \param[in]   width       image width.
 * \param[in]   size        size.
 * \param[in]   scale       scale.
 * \param[in]   pow         scale.
 *
 */
template <DeviceType Device>
void CrossMapNormal(real* outputs,
                    real* denoms,
                    const real* inputs,
                    size_t numSamples,
                    size_t channels,
                    size_t height,
                    size_t width,
                    size_t size,
                    real scale,
                    real pow);

/**
 * \brief   Cross map respose normalize backward.
 *          The data structure of image data is NCHW.
 *
 * \param[out]  inputsGrad      input grad.
 * \param[in]   inputsValue     input value.
 * \param[out]  outputsValue    output value.
 * \param[out]  outputsGrad     output grad.
 * \param[in]   denoms          denoms buffer.
 * \param[in]   numSamples      batch size of input image.
 * \param[in]   channels        number of channel.
 * \param[in]   height          image height.
 * \param[in]   width           image width.
 * \param[in]   size            size.
 * \param[in]   scale           scale.
 * \param[in]   pow             scale.
 *
 */
template <DeviceType Device>
void CrossMapNormalGrad(real* inputsGrad,
                        const real* inputsValue,
                        const real* outputsValue,
                        const real* outputsGrad,
                        const real* denoms,
                        size_t numSamples,
                        size_t channels,
                        size_t height,
                        size_t width,
                        size_t size,
                        real scale,
                        real pow);

}  // namespace paddle
