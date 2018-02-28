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
 * \brief Function to multiply a value to values in specified sub continuous
 *        region. Indices must be provided to indcate the location and shape of
 *        the region and the multiplied value is passed by configure variable.
 *
 *
 * \param[out] outputs  Output value.
 * \param[in]  inputs   Input data which contains NCHW information.
 * \param[in]  indices  Indices data to indcate the sub region.
 * \param[in]  shape    Tensor shape of input value.
 * \param[in]  conf     Configure variable which contains the multiplied value.
 */
template <DeviceType Device>
void ScaleSubRegion(real* outputs,
                    const real* inputs,
                    const real* indices,
                    const TensorShape shape,
                    const FuncConfig& conf);

/**
 * \brief Backward propagation function of ScaleSubRegion.
 *
 * \param[out] inGrad   Gradients of previous layer.
 * \param[in]  outGrad  Output gradient.
 * \param[in]  indices  Indices data.
 * \param[in]  shape    The Shape of input tensor.
 * \param[in]  conf     Configure variable.
 */
template <DeviceType Device>
void ScaleSubRegionGrad(const real* inGrad,
                        real* outGrad,
                        const real* indices,
                        const TensorShape shape,
                        const FuncConfig& conf);
}  // namespace paddle
