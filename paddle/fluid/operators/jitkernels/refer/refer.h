/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once
#include "paddle/fluid/operators/jitkernels/kernel_base.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace jitkernels {
namespace refer {

template <typename T>
void VMul(const T* x, const T* y, T* z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}

template <typename T>
class VMulKernel
    : public ReferKernel<T, void (*)(const T*, const T*, T*, int), int> {
 public:
  VMulKernel() { this->func = VMul<T>; }
};

}  // namespace refer
}  // namespace jitkernels
}  // namespace operators
}  // namespace paddle
