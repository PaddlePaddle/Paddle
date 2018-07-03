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
 * \brief  This funtion crops inputs according to the specify start point and
 *shape.
 *
 * \param[out] outputs	save results.
 * \param[in]  inputs	input data.
 * \param[in]  inShape  the shape of input tensor.
 * \param[in]  conf     the cropping config
 */
template <DeviceType Device>
void Crop(real* outputs,
          const real* inputs,
          const TensorShape inShape,
          const TensorShape outShape,
          const FuncConfig& conf);

/**
 * \brief   Cropping operation backward.
 *
 * \param[out] inGrad	gradients of previous layer
 * \param[in]  outGrad  output gradient
 * \param[in]  inShape  the shape of input tensor.
 * \param[in]  conf     the cropping config
 */
template <DeviceType Device>
void CropGrad(const real* inGrad,
              real* outGrad,
              const TensorShape inShape,
              const TensorShape outShape,
              const FuncConfig& conf);
}  // namespace paddle
