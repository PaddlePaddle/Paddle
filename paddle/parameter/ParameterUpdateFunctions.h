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

#include "paddle/math/Vector.h"
#include "paddle/utils/Common.h"

namespace paddle {

/**
 * Performs the following operations.
 *
 * momentumVec = momentum * momentumVec
 *               - learningRate * grad
 *               - learningRate * decayRate * value
 *
 * value = value + momentumVec
 * momentum = 0 or decayRate = 0 are specially handled to avoid unnecessary
 * computation.
 */
void sgdUpdate(real learningRate,
               real momentum,
               real decayRate,
               Vector* value,
               Vector* grad,
               Vector* momentumVec);

void sgdUpdateCpu(real learningRate,
                  real momentum,
                  real decayRate,
                  size_t size,
                  real* value,
                  const real* grad,
                  real* momentumVec);

void sgdUpdateAvx(float learningRate,
                  float momentum,
                  float decayRate,
                  size_t size,
                  float* value,
                  const float* grad,
                  float* momentumVec);

}  // namespace paddle
