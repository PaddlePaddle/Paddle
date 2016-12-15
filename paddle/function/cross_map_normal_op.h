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

template <DeviceType Device>
void CrossMapNormal(real* outputs,
                    real* denoms,
                    real* inputs,
                    size_t numSamples,
                    size_t channels,
                    size_t height,
                    size_t width,
                    size_t size,
                    real scale,
                    real pow);

template <DeviceType Device>
void CrossMapNormalGrad(real* inputsGrad,
                        real* inputsValue,
                        real* outputsValue,
                        real* outputsGrad,
                        real* denoms,
                        size_t numSamples,
                        size_t channels,
                        size_t height,
                        size_t width,
                        size_t size,
                        real scale,
                        real pow);

}  // namespace paddle
