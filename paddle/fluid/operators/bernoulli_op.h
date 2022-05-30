/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/hostdevice.h"

namespace paddle {
namespace operators {

/**
 * Samples a bernoulli distribution given a probability input
 */

template <typename T>
inline HOSTDEVICE T BernoulliFunctor(T p, T rand) {
  PADDLE_ENFORCE_LE(p, 1.0,
                    platform::errors::OutOfRange(
                        "The probability should be <= 1, but got %f", p));
  PADDLE_ENFORCE_GE(p, 0.0,
                    platform::errors::OutOfRange(
                        "The probability should be >= 0, but got %f", p));
  return static_cast<T>(rand < p);
}

template <typename DeviceContext, typename T>
class BernoulliOpKernel;

}  // namespace operators
}  // namespace paddle
