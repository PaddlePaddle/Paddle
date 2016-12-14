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
#include "paddle/math/Matrix.h"

namespace paddle {

template <DeviceType Device>
struct CrossMapNormal {
  void operator()(typename MatrixT<Device>::type& outputs,
                  typename MatrixT<Device>::type& denoms,
                  typename MatrixT<Device>::type& inputs,
                  size_t channels,
                  size_t imgSizeH,
                  size_t imgSizeW,
                  size_t sizeX,
                  real scale,
                  real pow);
};

template <DeviceType Device>
struct CrossMapNormalGrad {
  void operator()(typename MatrixT<Device>::type& inputsGrad,
                  typename MatrixT<Device>::type& inputsValue,
                  typename MatrixT<Device>::type& outputsGrad,
                  typename MatrixT<Device>::type& outputsValue,
                  typename MatrixT<Device>::type& denoms,
                  size_t channels,
                  size_t imgSizeH,
                  size_t imgSizeW,
                  size_t sizeX,
                  real scale,
                  real pow);
};

}  // namespace paddle
